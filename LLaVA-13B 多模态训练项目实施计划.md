# LLaVA-13B 多模态训练项目实施计划

## 项目周期
预计 6 周，目标是完成一个可展示的多模态大模型训练项目，并在简历中作为亮点。

---

## 第 1 周：环境搭建与模型理解
**目标**：熟悉 LLaVA 框架，完成基础运行环境搭建。  
**任务**：
- 搭建多机多卡训练环境（8×5090 32GB），安装 CUDA、PyTorch、DeepSpeed/FSDP。
- 阅读 LLaVA 官方代码（模型结构、训练流程、数据预处理）。
- 在单卡上跑通 LLaVA-7B 推理 Demo。  
**产出**：
- 环境安装脚本（requirements.txt / conda env）。
- 推理 Demo 截图。
Access Token: MfwFzzfddP8QgAnnTeTtdxBD
pip install torch torchvision -f https://pypi.tuna.tsinghua.edu.cn/pytorch-wheels/cu128/
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m llava.serve.cli \
    --model-path /root/autodl-tmp/model/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
aria2c -x16 -s16 -c \
  -o c0eb94eb3fa578a4c63567f2eca6b0acf6a44a891c257aca0df88ffa891de72e \
  https://hf-mirror.com/liuhaotian/llava-v1.5-7b/resolve/main/pytorch_model-00002-of-00002.bin

ssh -p 39815 root@connect.westd.seetacloud.com

QWERTYU
---

## 第 2 周：数据准备与小规模微调
**目标**：跑通最小可行的 LoRA 微调。  
**任务**：
- 准备公开小数据集（COCO Caption / Flickr30k / Visual Genome QA）。
- 编写数据预处理脚本（图像编码、文本 tokenization）。
- 使用 LoRA 对 LLaVA-7B 或 13B 进行小规模微调（几千步）。  
**产出**：
- 可重复的数据预处理 pipeline。
- 微调 loss 曲线与日志。

---

## 第 3 周：分布式训练与规模扩展
**目标**：在 8 张 GPU 上稳定训练 13B。  
**任务**：
- 配置 DeepSpeed ZeRO-3 或 FSDP，测试多卡吞吐性能。
- 在 LLaVA-13B 上进行 LoRA 微调，batch size ≥ 128。
- 记录 GPU 利用率和训练速度。  
**产出**：
- 分布式训练脚本（带配置文件）。
- 性能指标（吞吐量、显存占用）。

---

## 第 4 周：中文数据集构建与指令微调
**目标**：构造一个小规模中文图文 QA 数据集，提升简历亮点。  
**任务**：
- 收集/爬取少量中文图文数据（如百科插图、截图+问答）。
- 使用 GPT-4 生成中文指令式问答数据。
- 在中文数据上做指令微调。  
**产出**：
- 自建中文多模态 QA 数据集（1–2 万对）。
- 微调后支持中文问答的模型 checkpoint。

---

## 第 5 周：模型评测与 Demo 搭建
**目标**：评估模型性能并搭建交互式展示。  
**任务**：
- 在 VQA-v2、MME 等小规模 benchmark 上做评测。
- 搭建 Gradio / Streamlit Demo（输入图片+文本 → 模型回答）。
- 对比 LLaVA 原版与微调版本的效果。  
**产出**：
- Benchmark 结果表格。
- 在线或本地 Demo 页面。

---

## 第 6 周：结果整理与项目打磨
**目标**：形成完整的项目成果，适合写进简历。  
**任务**：
- 整理训练过程、实验结果，撰写技术报告（博客或 README）。
- 制作 Demo 视频或 GIF 展示。
- 总结遇到的问题与优化点（如显存优化、数据增强）。  
**产出**：
- 项目报告（Markdown / PDF）。
- Demo 视频 / 截图。
- 简历条目示例：
  - 「基于 LLaVA-13B 搭建多模态大模型训练 pipeline，完成中文图文 QA 指令微调，在 VQA-v2 上提升 X%，实现 Gradio Demo。」

---
