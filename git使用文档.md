# clone
git clone xxxxx

# 设置 HTTP 代理
git config --global http.proxy http://127.0.0.1:7890
# 设置 HTTPS 代理
git config --global https.proxy http://127.0.0.1:7890

# atn：如果你以后关闭了代理软件，或者换了网络环境发现 git 命令报错，记得运行以下命令“清除”设置，否则 Git 会因为找不到代理而无法联网：
git config --global --unset http.proxy
git config --global --unset https.proxy

git checkout -b feature-my-update
git add .
git commit -m "...."
# 第一次推送时需要关联远程分支
git push -u origin feature-my-update

# 常用配套命令
# 查看当前在哪个分支： git branch（带星号的就是你现在的分支）。
# 切换回主分支： 
git checkout main
# 合并分支： 先切换回 main，然后运行 
git merge feature-my-update


# 代码改完了，仅仅在编辑器里 Ctrl+S 是不够的，你还需要告诉 Git 记录下这些变化：

# 第一步：查看改了什么
git status
# 你会看到红色的文件名，这些是你改动过但还没“登记”的文件。

# 第二步：添加到暂存区 (登记)
git add .

# 第三步：正式提交 (存档)
git commit -m "描述你改了什么，例如：优化了机械臂动作预测的扩散步数"

# 上传github
git push -u origin feature-my-update