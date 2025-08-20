#!/usr/bin/env python3
"""
YOLOv5 茶叶病虫害分类训练脚本 - C盘版本
解决跨盘符路径问题的版本
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path
import subprocess
import importlib
from typing import Optional
import platform

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 茶叶病虫害分类训练")
    parser.add_argument("--data", type=str, 
                       default=str(BASE_DIR / "static" / "tea-bug and disease-data"),
                       help="原始数据目录路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=224, help="输入图像大小")
    parser.add_argument("--batch", type=int, default=32, help="批次大小")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/0/1/...)")
    parser.add_argument("--name", type=str, default="tea_pests_cls_yolov5", help="实验名称")
    parser.add_argument("--workers", type=int, default=4, help="数据加载工作进程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--c-drive-work-dir", type=str, default="C:/tea_yolov5_training", 
                       help="C盘工作目录")
    parser.add_argument("--keep-temp", action="store_true", help="保留C盘临时文件")
    
    return parser.parse_args()

def ensure_package(package_name: str, import_name: Optional[str] = None) -> bool:
    """确保包已安装"""
    if import_name is None:
        import_name = package_name
        
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"正在安装 {package_name}...")
        
        # 特殊处理 opencv-python 安装问题
        if package_name == "opencv-python" or "opencv" in package_name:
            return install_opencv_with_fallback()
        
        # 特殊处理 yolov5 安装问题
        if package_name == "yolov5":
            return install_yolov5_with_fallback()
            
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            # 清除导入缓存
            if import_name in sys.modules:
                del sys.modules[import_name]
            importlib.invalidate_caches()
            importlib.import_module(import_name)
            return True
        except Exception as e:
            print(f"安装 {package_name} 失败: {e}")
            return False

def install_opencv_with_fallback() -> bool:
    """使用多种策略安装 opencv-python"""
    opencv_packages = [
        "opencv-python-headless",  # 无GUI版本，更容易安装
        "opencv-contrib-python-headless",  # 带额外模块的无GUI版本
        "opencv-python",  # 标准版本
    ]
    
    for package in opencv_packages:
        print(f"尝试安装 {package}...")
        try:
            # 先尝试使用预编译的wheel
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--only-binary=all", package
            ])
            
            # 验证安装
            try:
                import cv2
                print(f"✅ {package} 安装成功")
                return True
            except ImportError:
                continue
                
        except subprocess.CalledProcessError:
            print(f"❌ {package} 安装失败，尝试下一个...")
            continue
    
    # 如果所有预编译版本都失败，尝试conda安装
    if shutil.which("conda"):
        print("尝试使用 conda 安装 opencv...")
        try:
            subprocess.check_call(["conda", "install", "-y", "opencv", "-c", "conda-forge"])
            import cv2
            print("✅ conda 安装 opencv 成功")
            return True
        except:
            print("❌ conda 安装也失败")
    
    print("❌ 所有 opencv 安装方法都失败")
    return False

def install_yolov5_with_fallback() -> bool:
    """使用多种策略安装 yolov5"""
    install_methods = [
        # 方法1: 直接安装yolov5
        lambda: subprocess.check_call([sys.executable, "-m", "pip", "install", "yolov5"]),
        
        # 方法2: 从GitHub安装
        lambda: subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/ultralytics/yolov5.git"
        ]),
        
        # 方法3: 安装ultralytics（新版本）
        lambda: subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"]),
    ]
    
    for i, install_method in enumerate(install_methods, 1):
        print(f"尝试 yolov5 安装方法 {i}...")
        try:
            install_method()
            
            # 验证安装
            try:
                import yolov5
                print("✅ yolov5 安装成功")
                return True
            except ImportError:
                try:
                    import ultralytics
                    print("✅ ultralytics 安装成功（可替代 yolov5）")
                    return True
                except ImportError:
                    continue
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ 方法 {i} 失败: {e}")
            continue
    
    print("❌ 所有 yolov5 安装方法都失败")
    return False

def train_with_ultralytics(c_data_dir: Path, args, c_work_dir: Path):
    """使用 ultralytics 进行训练"""
    from ultralytics import YOLO
    
    # 创建运行目录
    runs_dir = c_work_dir / "runs" / "classify"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备参数
    device = detect_device() if args.device == "auto" else args.device
    
    # 加载预训练模型
    model = YOLO('yolov8n-cls.pt')  # 使用YOLOv8分类模型
    
    print("开始使用 ultralytics 训练...")
    print(f"数据目录: {c_data_dir}")
    print(f"设备: {device}")
    
    # 训练参数
    results = model.train(
        data=str(c_data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(runs_dir),
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        exist_ok=True,
    )
    
    print("ultralytics 训练完成！")
    
    # 返回训练结果目录
    result_dir = runs_dir / args.name
    return result_dir

def detect_device():
    """检测可用设备"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            print(f"使用 GPU 训练：{device_name}, compute capability={compute_capability}")
            return "0"
        else:
            print("CUDA 不可用，使用 CPU 训练")
            return "cpu"
    except ImportError:
        print("PyTorch 未安装，使用 CPU")
        return "cpu"

def prepare_data_on_c_drive(source_data_dir: Path, c_work_dir: Path):
    """在C盘准备训练数据"""
    print(f"正在准备数据到 C 盘: {c_work_dir}")
    
    # 创建C盘工作目录
    c_work_dir.mkdir(parents=True, exist_ok=True)
    c_data_dir = c_work_dir / "data"
    
    # 如果已经存在处理好的数据，直接返回
    if (c_data_dir / "train").exists() and (c_data_dir / "val").exists():
        print(f"发现已存在的训练数据: {c_data_dir}")
        return c_data_dir
    
    # 调用数据准备脚本
    prepare_script = BASE_DIR / "scripts" / "prepare_tea_pests_classification.py"
    if not prepare_script.exists():
        raise FileNotFoundError(f"数据准备脚本不存在: {prepare_script}")
    
    print("正在准备分类数据...")
    temp_prepared_dir = BASE_DIR / "datasets" / "tea_pests_classify"
    
    cmd = [sys.executable, str(prepare_script), "--source", str(source_data_dir), "--dest", str(temp_prepared_dir)]
    subprocess.check_call(cmd)
    
    # 复制到C盘
    print(f"正在复制数据到 C 盘: {c_data_dir}")
    if c_data_dir.exists():
        shutil.rmtree(c_data_dir)
    shutil.copytree(temp_prepared_dir, c_data_dir)
    
    # 验证数据
    train_dir = c_data_dir / "train"
    val_dir = c_data_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise RuntimeError(f"数据准备失败: {c_data_dir}")
    
    # 统计数据
    print("数据统计:")
    for class_dir in sorted(train_dir.iterdir()):
        if class_dir.is_dir():
            train_count = len(list(class_dir.glob("*.jpg")))
            val_count = len(list((val_dir / class_dir.name).glob("*.jpg")))
            print(f"  {class_dir.name}: train={train_count}, val={val_count}")
    
    return c_data_dir

def train_on_c_drive(c_work_dir: Path, c_data_dir: Path, args):
    """在C盘进行训练"""
    print(f"开始在 C 盘训练: {c_work_dir}")
    
    # 切换到C盘工作目录
    original_cwd = os.getcwd()
    os.chdir(str(c_work_dir))
    
    try:
        # 检查YOLOv5
        if not ensure_package("yolov5"):
            raise RuntimeError("无法安装 yolov5")
        
        # 修复YOLOv5的torch.load问题
        fix_script = BASE_DIR / "scripts" / "fix_yolov5_torch_load.py"
        if fix_script.exists():
            print("正在修复YOLOv5的torch.load问题...")
            subprocess.call([sys.executable, str(fix_script)])
        
        # 获取训练脚本路径
        train_py = None
        
        # 尝试使用 yolov5
        try:
            import yolov5
            yolov5_path = Path(yolov5.__file__).parent
            train_py = yolov5_path / "classify" / "train.py"
            print(f"使用 yolov5 训练脚本: {train_py}")
        except ImportError:
            # 尝试使用 ultralytics
            try:
                import ultralytics
                print("使用 ultralytics 进行训练")
                return train_with_ultralytics(c_data_dir, args, c_work_dir)
            except ImportError:
                raise RuntimeError("无法导入 yolov5 或 ultralytics")
        
        if not train_py or not train_py.exists():
            raise RuntimeError(f"未找到 YOLOv5 分类训练脚本: {train_py}")
        
        # 设备参数
        device = detect_device() if args.device == "auto" else args.device
        y5_device = "cpu" if device == "cpu" else "0"
        
        # 训练参数
        runs_dir = c_work_dir / "runs" / "classify"
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量解决PyTorch weights_only问题
        env = os.environ.copy()
        env['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'
        
        cmd = [
            sys.executable, str(train_py),
            "--model", "yolov5n-cls.pt",
            "--data", str(c_data_dir),
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--device", y5_device,
            "--project", str(runs_dir),
            "--name", args.name,
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--exist-ok",
        ]
        
        print("执行训练命令:")
        print(" ".join(cmd))
        print("-" * 60)
        
        # 开始训练，传递修改后的环境变量
        subprocess.check_call(cmd, env=env)
        
        print("-" * 60)
        print("训练完成！")
        
        # 返回训练结果目录
        result_dir = runs_dir / args.name
        return result_dir
        
    finally:
        # 恢复原工作目录
        os.chdir(original_cwd)

def copy_results_back(c_result_dir: Path, args):
    """将训练结果复制回E盘"""
    if not c_result_dir.exists():
        print(f"警告: 训练结果目录不存在: {c_result_dir}")
        return None
        
    # 目标目录
    target_runs_dir = BASE_DIR / "runs" / "classify"
    target_runs_dir.mkdir(parents=True, exist_ok=True)
    target_result_dir = target_runs_dir / args.name
    
    # 复制结果
    print(f"正在复制训练结果到: {target_result_dir}")
    if target_result_dir.exists():
        shutil.rmtree(target_result_dir)
    shutil.copytree(c_result_dir, target_result_dir)
    
    # 复制模型文件到 models 目录
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    best_weights = target_result_dir / "weights" / "best.pt"
    if best_weights.exists():
        target_model = models_dir / "tea_pests_cls.pt"
        shutil.copy2(best_weights, target_model)
        print(f"模型已保存到: {target_model}")
        
        # 创建类别名称文件
        class_names = []
        train_dir = Path(str(c_result_dir).replace("runs/classify", "data")) / "train"
        if train_dir.exists():
            class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        
        if class_names:
            names_file = models_dir / "tea_pests_cls.names"
            with open(names_file, 'w', encoding='utf-8') as f:
                for name in class_names:
                    f.write(f"{name}\n")
            print(f"类别名称已保存到: {names_file}")
    
    return target_result_dir

def main():
    args = parse_args()
    
    print(f"使用的 Python 解释器：{sys.executable}")
    print(f"原始数据目录：{args.data}")
    print(f"C盘工作目录：{args.c_drive_work_dir}")
    
    # 确保必要的包
    if not ensure_package("torch"):
        raise RuntimeError("无法安装 PyTorch")
    
    if not ensure_package("yolov5"):
        raise RuntimeError("无法安装 YOLOv5")
    
    # 路径设置
    source_data_dir = Path(args.data)
    c_work_dir = Path(args.c_drive_work_dir)
    
    if not source_data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {source_data_dir}")
    
    try:
        # 1. 在C盘准备数据
        c_data_dir = prepare_data_on_c_drive(source_data_dir, c_work_dir)
        
        # 2. 在C盘进行训练
        c_result_dir = train_on_c_drive(c_work_dir, c_data_dir, args)
        
        # 3. 复制结果回E盘
        target_result_dir = copy_results_back(c_result_dir, args)
        
        if target_result_dir:
            print(f"\n✅ 训练完成！结果保存在: {target_result_dir}")
            print(f"✅ 模型文件: {BASE_DIR / 'models' / 'tea_pests_cls.pt'}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise
    
    finally:
        # 清理C盘临时文件
        if not args.keep_temp and c_work_dir.exists():
            try:
                print(f"正在清理C盘临时文件: {c_work_dir}")
                shutil.rmtree(c_work_dir)
                print("✅ 清理完成")
            except Exception as e:
                print(f"⚠️  清理失败: {e}")

if __name__ == "__main__":
    main()
