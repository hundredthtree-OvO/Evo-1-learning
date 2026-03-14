# model/internvl3/internvl3_embedder.py
import torch
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import GenerationConfig
from torchvision.transforms.functional import to_pil_image
from typing import Union, List
from torch import nn
import logging
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img), # 转 RGB
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), # resize 到 448 x 448
        T.ToTensor(), # ToTensor
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # ImageNet mean/std normalize
    ])

# === Aspect Ratio Handling ===
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    '''
    function: 将输入图像切分成多个 tile，并调整每个 tile 的大小以适配 InternVL3 的输入要求。切分的方式根据图像的宽高比动态决定，以尽量减少信息损失。切分后的 tile 会被 resize 成指定的 image_size，最后返回一个包含所有 tile 的列表。
    input: image (PIL.Image) - 输入图像；min_num (int) - 最小切分块数；max_num (int) - 最大切分块数；image_size (int) - 每个 tile 的目标大小；use_thumbnail (bool) - 是否在切分后添加一个缩略图。
    output: List[PIL.Image] - 切分并调整大小后的图像块
    '''
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVL3Embedder(nn.Module):
    def __init__(self, model_name="OpenGVLab/InternVL3-1B", image_size=448, device="cuda"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.transform = build_transform(image_size)
        # 加载预训练模型和分词器，设置模型为评估模式，并冻结所有参数以节省内存和计算资源。
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=True,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        # hasattr：检查模型是否具有 language_model 属性，如果有则进一步检查是否具有 model 属性，以适配不同版本的 InternVL3 模型结构。
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        # 只保留 language model 前 14 层
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        # nn.Identity() 是一个占位符模块，它的 forward 方法直接返回输入，不进行任何修改。
        # 这里把语言模型的 lm_head 替换成 Identity，意味着我们不使用预训练语言模型的输出层，而是直接使用最后一层的隐藏状态作为视觉语言融合后的特征表示。
        # 这通常用于下游任务中，我们会在这些隐藏状态上添加自己的头部（比如动作头）来进行特定任务的预测。
        self.model.language_model.lm_head = torch.nn.Identity()

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = False
        
    # 图像预处理
    def _preprocess_images(
        self,
        image_tensors: List[Union[Image.Image, torch.Tensor]]
    ) -> (torch.Tensor, List[int]):

        pixel_values_list = []
        for i, image in enumerate(image_tensors):
            # 如果输入是 tensor，先转 PIL
            if isinstance(image, torch.Tensor):
                image = to_pil_image(image)
            # 对每张图像进行动态切分和预处理，得到一个包含所有 tile 的列表
            tiles = dynamic_preprocess(image, image_size=self.image_size)
            # transform()调用了build_transform 函数进行预定义的图像变换
            tile_tensors = torch.stack([self.transform(t) for t in tiles])  # (T_i, 3, 448, 448)
            pixel_values_list.append(tile_tensors)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=torch.bfloat16, device=self.device)
        num_tiles_list = [pv.shape[0] for pv in pixel_values_list]

        return pixel_values, num_tiles_list

    # prompt 构造：构造一个带图像占位 token 的语言序列，再用视觉特征替换这些占位 token 的 embedding
    def _build_multimodal_prompt(
        self,
        num_tiles_list: List[int],
        text_prompt: str
    ) -> str:

        prompt = ''
        # 拼出结构：
        # Image-1: <image>
        # Image-2: <image>
        # ...
        for i in range(len(num_tiles_list)):
            prompt += f"Image-{i+1}: <image>\n"
        # 把文本指令追加到 prompt 的末尾，strip() 去掉首尾空白，确保格式整洁。
        prompt += text_prompt.strip()

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        # 把每个 <image> 替换成：
        # <img><IMG_CONTEXT>...<IMG_CONTEXT></img>
        # 替换数量由 self.model.num_image_token * tile_count 决定。
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        for tile_count in num_tiles_list:
            token_count = self.model.num_image_token * tile_count
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
            prompt = prompt.replace("<image>", image_tokens, 1)

        return prompt
    
    # 视觉特征塞进语言输入
    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int]
    ) -> (torch.Tensor, torch.Tensor):
   
        # 先 tokenizer prompt
        untruncated_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        true_sequence_length = untruncated_ids.shape[1]

        if true_sequence_length > self.max_text_length:
            print("\n" + "="*80)
            print(f" WARNING: Input prompt was TRUNCATED!")
            print(f"   - Max Length Allowed    : {self.max_text_length}")
            print(f"   - Actual Length      : {true_sequence_length}")
            print(f"   - Truncated Prompt (first 100 chars): '{prompt[:100]}...'")
            print("="*80 + "\n")

        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_text_length).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

       
        img_token_mask = (input_ids == self.img_context_token_id)
     
        img_token_locations = torch.where(img_token_mask)[1]


        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        selected = (input_ids == self.img_context_token_id)

            
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

 
        tokens_per_tile = self.model.num_image_token 
 
        torch.set_printoptions(profile="full", threshold=float('inf'))
   
        torch.set_printoptions(profile="default")
        current_token_idx = 0
        for i in range(len(image_mask)):
           
            num_tiles_for_this_image = num_tiles_list[i]
            num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
       
            if not image_mask[i]:
                
                start_idx = img_token_locations[current_token_idx]
                end_idx = start_idx + num_tokens_for_this_image
               
                attention_mask[0, start_idx:end_idx] = 0
    
            current_token_idx += num_tokens_for_this_image

        input_embeds = input_embeds.reshape(B, N, C)
    
        torch.set_printoptions(profile="full", threshold=float('inf'))
     
        torch.set_printoptions(profile="default")
        return input_embeds, attention_mask


    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors: list[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        text_prompt: str,
        return_cls_only: bool = True,
    ):

   
        pixel_values, num_tiles_list = self._preprocess_images(image_tensors)

       
        if pixel_values.shape[0] == 0:
           
            print("Warning: No valid images to process after masking.")

        vit_embeds = self.model.extract_feature(pixel_values)
        fused_embeds = vit_embeds  
        prompt = self._build_multimodal_prompt(num_tiles_list, text_prompt)
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(prompt, fused_embeds, image_mask, num_tiles_list)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        fused_hidden = outputs.hidden_states[-1].to(torch.float32)

        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden
