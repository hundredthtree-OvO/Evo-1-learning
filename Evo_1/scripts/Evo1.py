# Evo1.py 是这个项目的“模型装配层”。
# 它自己不实现视觉语言编码细节，也不实现 flow matching 细节，而是把两部分接起来：
# InternVL3Embedder 负责 图像 + 文本 -> fused tokens
# FlowmatchingActionHead 负责 fused tokens + state -> 动作/速度
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from types import SimpleNamespace
from typing import List, Union, Tuple
from PIL import Image
import torch
import torch.nn as nn
from model.internvl3.internvl3_embedder import InternVL3Embedder
from model.action_head.flow_matching import FlowmatchingActionHead
import logging
class EVO1(nn.Module):
    def __init__(self, config: dict):
        super().__init__() 
        self.config = config
        # .get() 方法用于从配置字典中获取指定键的值，如果键不存在则返回默认值。
        self._device = config.get("device", "cuda")
        self.return_cls_only = config.get("return_cls_only", False)
        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        # 初始化视觉语言骨干
        self.embedder = InternVL3Embedder(model_name=vlm_name, device=self._device)

        action_head_type = config.get("action_head", "flowmatching").lower()
        
        if action_head_type == "flowmatching":
            
            # 统一整理动作空间参数
            horizon = config.get("action_horizon", config.get("horizon", 16))
            per_action_dim = config.get("per_action_dim", 7)
            # 说明 Evo1 的动作不是单步 action，而是一个 action chunk
            # e.g. horizon=50, per_action_dim=24，那一次就预测 50 x 24 的动作序列。
            action_dim = horizon * per_action_dim
            
            config["horizon"] = horizon
            config["per_action_dim"] = per_action_dim
            config["action_dim"] = action_dim
            
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            # 根据配置初始化动作头
            # SimpleNamespace 是一个简单的对象类型，可以动态添加属性，这里用它来传递一大堆参数给 FlowmatchingActionHead。
            self.action_head = FlowmatchingActionHead(config=SimpleNamespace(
                embed_dim=config.get("embed_dim", 896),    
                hidden_dim=config.get("hidden_dim", 1024),
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                state_dim=config.get("state_dim", 7),
                state_hidden_dim=config.get("state_hidden_dim", 1024),
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 8),
                dropout=config.get("dropout", 0.0),
                num_inference_timesteps=config.get("num_inference_timesteps", 50),
                num_categories=config.get("num_categories", 1)
            )).to(self._device)
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")
    
    # 把多模态输入变成统一上下文
    def get_vl_embeddings(
        self,
        images: List[Image.Image],
        image_mask: torch.Tensor,  
        prompt: str = "",
        return_cls_only: Union[bool, None] = None
    ) -> torch.Tensor:
        
        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image (PIL.Image). Got `images=None` or empty list.")
        # 把图像和文本指令编码成动作头可以消费的上下文 token
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,  # return_cls_only=True 时，只取一个聚合表示;否则返回整段 token sequence
        )
    
    # 把机器人状态标准化成批输入
    # 总结：state 在 Evo1 里被视为独立条件输入，不是混在 prompt 里的文本，也不是直接拼到图像 token 前面；后面动作头会把它编码成一个额外 token。
    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:
        
        # 支持 list 或 torch.Tensor 输入，如果是 list 就转换成 tensor，如果已经是 tensor 就直接用
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")
        
        # 如果是一维，就补 batch 维
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        # 搬到模型 device
        return state_tensor.to(self._device)
    
    # 训练和推理在这里分叉
    # 训练时，动作头输出的是 pred_velocity, noise
    # 推理时，动作头直接返回最终 action chunk
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # 如果 actions_gt is None，走推理路径
        if actions_gt is None:
            return self.action_head.get_action(fused_tokens, state=state, action_mask=action_mask, embodiment_id=embodiment_ids)
        else:
            return self.action_head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids)

    # 完整推理入口，线上调用的完整路径
    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:

        fused_tokens = self.get_vl_embeddings(
                        images=images,
                        image_mask=image_mask,
                        prompt=prompt,
                        return_cls_only=return_cls_only
                        
                    )

        state_tensor = self.prepare_state(state_input)  
        
        # action_gt 为 None: 推理
        return self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)
    

    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None):
   
        # forward ：训练时调用，输入已经是融合后的视觉语言 token 和状态，以及动作标签和掩码，输出预测的动作速度和噪声。
        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    def _freeze_module(self, module: nn.Module, name: str):
        print(f"Freezing {name} parameters...")
        for p in module.parameters():
            p.requires_grad = False

    def set_finetune_flags(self):
        config = self.config  
        if not config.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")
        else:
            print("Finetuning VLM (InternVL3)...")

        if not config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")
