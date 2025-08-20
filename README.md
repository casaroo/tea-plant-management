# tea-plant-management
一个面向茶园精细化管理的开源项目：集生长模型（如 LAI/产量预测、WOFOST/PCSE 接口）、病虫害识别（YOLO/传统ML）、环境数据接入（NASA POWER / 气象站）与可视化与API服务于一体，支持本地与服务器部署。

🌱 作物生长建模：LAI/产量估计、环境驱动模拟（PCSE/WOFOST 接口）

🐛 病虫害识别：训练与推理脚本，支持批量推断与结果可视化

☁️ 数据接入：气象/土壤数据导入与清洗，示例 pipelines

🧩 API 服务：基于 Flask 的 RESTful 接口

📊 看板与可视化：关键指标与模型结果可视化

🚀 一键部署：本地开发与生产部署指引（Gunicorn/Nginx/Conda）

📦 环境要求

Python 3.9+
推荐使用 Anaconda / Miniconda
可选：GPU（CUDA 11.x）用于加速训练/推理

按照requirements安装后直接运行app.py即可使用
