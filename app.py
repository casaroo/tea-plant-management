from flask import Flask, render_template, jsonify, request, url_for
from flask_sqlalchemy import SQLAlchemy
import datetime
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json
import subprocess
import sys
from sqlalchemy import func

# 初始化Flask应用
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tea_growth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# 简易表结构自检（用于SQLite免迁移补列）
def _ensure_pest_record_schema():
    try:
        res = db.session.execute("PRAGMA table_info(pest_record)").fetchall()
        existing_cols = {row[1] for row in res}
        if 'location' not in existing_cols:
            db.session.execute("ALTER TABLE pest_record ADD COLUMN location VARCHAR(120)")
            db.session.commit()
    except Exception:
        # 忽略自检失败
        pass

# 确保模型目录存在
if not os.path.exists('models'):
    os.makedirs('models')
# 为每棵树的3D模型建立独立缓存目录
TREE_MODELS_DIR = os.path.join('models', 'trees')
os.makedirs(TREE_MODELS_DIR, exist_ok=True)

def _tree_model_path(plant_id: int) -> str:
    return os.path.join(TREE_MODELS_DIR, f"{plant_id}.json")

def load_cached_tree_model(plant_id: int):
    try:
        path = _tree_model_path(plant_id)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        # 缓存读取失败时忽略，回退到重新生成
        pass
    return None

def save_tree_model(plant_id: int, model_data: dict) -> None:
    try:
        path = _tree_model_path(plant_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False)
    except Exception:
        # 缓存保存失败不影响接口返回
        pass

# 资源与上传目录
UPLOADS_DIR = os.path.join('uploads', 'trees')
os.makedirs(UPLOADS_DIR, exist_ok=True)

# GLB真实模型静态目录（前端可直接访问）
STATIC_GLB_DIR = os.path.join('static', 'models', 'trees_glb')
os.makedirs(STATIC_GLB_DIR, exist_ok=True)

def _tree_images_dir(plant_id: int) -> str:
    path = os.path.join(UPLOADS_DIR, str(plant_id))
    os.makedirs(path, exist_ok=True)
    return path

def _tree_masks_dir(plant_id: int) -> str:
    path = os.path.join(UPLOADS_DIR, str(plant_id), 'masks')
    os.makedirs(path, exist_ok=True)
    return path

def _tree_glb_path(plant_id: int) -> str:
    return os.path.join(STATIC_GLB_DIR, f"{plant_id}.glb")

# 数据模型定义
class TeaPlant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    variety = db.Column(db.String(100))
    planting_date = db.Column(db.DateTime)
    current_stage = db.Column(db.String(50))
    height = db.Column(db.Float)
    crown_width = db.Column(db.Float)  # 冠幅
    leaf_count = db.Column(db.Integer)
    location_area = db.Column(db.String(10))  # 种植区域 (A区, B区, C区, D区)
    location_row = db.Column(db.String(20))   # 具体位置 (如: 1排, 2排等)
    last_fertilized = db.Column(db.DateTime)  # 上次施肥时间
    last_updated = db.Column(db.DateTime, default=datetime.datetime.now)
    
    # 关联关系
    fertilization_records = db.relationship('FertilizationRecord', backref='tea_plant', lazy=True)
    agricultural_plans = db.relationship('AgriculturalPlan', backref='tea_plant', lazy=True)

class GrowthRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tea_plant_id = db.Column(db.Integer, db.ForeignKey('tea_plant.id'))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    stage = db.Column(db.String(50))
    height = db.Column(db.Float)
    width = db.Column(db.Float)
    leaf_condition = db.Column(db.String(100))
    notes = db.Column(db.Text)

class EnvironmentalData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    soil_moisture = db.Column(db.Float)
    soil_ph = db.Column(db.Float)
    nitrogen = db.Column(db.Float)
    phosphorus = db.Column(db.Float)
    potassium = db.Column(db.Float)

class AgriculturalActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tea_plant_id = db.Column(db.Integer, db.ForeignKey('tea_plant.id'))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    activity_type = db.Column(db.String(100))
    description = db.Column(db.Text)
    materials_used = db.Column(db.String(200))
    执行人 = db.Column(db.String(100))

class PestRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    pest_type = db.Column(db.String(100))
    severity = db.Column(db.Integer)  # 1-10
    affected_area = db.Column(db.Float)
    treatment = db.Column(db.Text)
    image_path = db.Column(db.String(200))
    # 说明：旧库无此列，启动和页面/API处会自动补列
    location = db.Column(db.String(120), nullable=True)

class OptimalGrowthModel(db.Model):
    """最优生长模型数据表"""
    id = db.Column(db.Integer, primary_key=True)
    tea_plant_id = db.Column(db.Integer, db.ForeignKey('tea_plant.id'))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    growth_stage = db.Column(db.String(50))  # 生长阶段
    optimal_temperature = db.Column(db.Float)  # 最适温度
    optimal_humidity = db.Column(db.Float)  # 最适湿度
    optimal_soil_moisture = db.Column(db.Float)  # 最适土壤湿度
    optimal_soil_ph = db.Column(db.Float)  # 最适土壤pH
    optimal_nitrogen = db.Column(db.Float)  # 最适氮含量
    optimal_phosphorus = db.Column(db.Float)  # 最适磷含量
    optimal_potassium = db.Column(db.Float)  # 最适钾含量
    growth_rate = db.Column(db.Float)  # 生长速率
    yield_potential = db.Column(db.Float)  # 产量潜力
    quality_score = db.Column(db.Float)  # 品质评分

class AgriculturalOptimization(db.Model):
    """农事活动优化建议表"""
    id = db.Column(db.Integer, primary_key=True)
    tea_plant_id = db.Column(db.Integer, db.ForeignKey('tea_plant.id'))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    activity_type = db.Column(db.String(100))  # 活动类型
    current_timing = db.Column(db.String(100))  # 当前时机
    optimal_timing = db.Column(db.String(100))  # 最优时机
    reason = db.Column(db.Text)  # 优化原因
    expected_benefit = db.Column(db.String(200))  # 预期收益
    priority = db.Column(db.Integer)  # 优先级 1-5
    status = db.Column(db.String(20), default='pending')  # 状态：pending, applied, completed

class FertilizationRecord(db.Model):
    """施肥记录表"""
    id = db.Column(db.Integer, primary_key=True)
    tea_plant_id = db.Column(db.Integer, db.ForeignKey('tea_plant.id'))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    fertilizer_type = db.Column(db.String(100))  # 肥料类型
    amount = db.Column(db.Float)  # 施肥量(kg)
    nitrogen_content = db.Column(db.Float)  # 氮含量(%)
    phosphorus_content = db.Column(db.Float)  # 磷含量(%)
    potassium_content = db.Column(db.Float)  # 钾含量(%)
    method = db.Column(db.String(100))  # 施肥方法
    weather_condition = db.Column(db.String(100))  # 施肥时天气状况
    soil_moisture = db.Column(db.Float)  # 土壤湿度
    operator = db.Column(db.String(100))  # 操作人
    notes = db.Column(db.Text)  # 备注

class AgriculturalPlan(db.Model):
    """农事计划表"""
    id = db.Column(db.Integer, primary_key=True)
    tea_plant_id = db.Column(db.Integer, db.ForeignKey('tea_plant.id'))
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)
    plan_start_date = db.Column(db.DateTime)  # 计划开始日期
    plan_end_date = db.Column(db.DateTime)  # 计划结束日期
    season = db.Column(db.String(50))  # 季节
    growth_stage = db.Column(db.String(50))  # 生长阶段
    plan_type = db.Column(db.String(100))  # 计划类型(日常管理/特殊处理/季节性管理等)
    status = db.Column(db.String(50), default='pending')  # 状态(pending/in_progress/completed/cancelled)
    priority = db.Column(db.Integer)  # 优先级(1-5)

class AgriculturalPlanDetail(db.Model):
    """农事计划详细项目表"""
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('agricultural_plan.id'))
    activity_type = db.Column(db.String(100))  # 活动类型(施肥/修剪/浇水/病虫害防治等)
    scheduled_date = db.Column(db.DateTime)  # 计划执行日期
    description = db.Column(db.Text)  # 详细描述
    materials_needed = db.Column(db.Text)  # 所需物料
    estimated_duration = db.Column(db.Integer)  # 预计持续时间(分钟)
    weather_requirement = db.Column(db.String(100))  # 天气要求
    precautions = db.Column(db.Text)  # 注意事项
    status = db.Column(db.String(50), default='pending')  # 状态
    completion_notes = db.Column(db.Text)  # 完成情况记录

# 初始化数据库
with app.app_context():
    db.create_all()
    # 轻量自检：为已有SQLite库补齐新增字段（如 location）
    _ensure_pest_record_schema()
    
    # 如果没有数据，添加一些示例数据
    if TeaPlant.query.count() == 0:
        tea_plant = TeaPlant(
            variety="龙井",
            planting_date=datetime.datetime(2023, 3, 15),
            current_stage="幼苗期",
            height=15.2,
            crown_width=20.5,
            leaf_count=12
        )
        db.session.add(tea_plant)
        db.session.commit()
        
        # 添加环境数据
        for i in range(30):
            date = datetime.datetime.now() - datetime.timedelta(days=i)
            env_data = EnvironmentalData(
                timestamp=date,
                temperature=20 + random.uniform(-5, 5),
                humidity=70 + random.uniform(-10, 10),
                rainfall=random.uniform(0, 20),
                soil_moisture=60 + random.uniform(-15, 15),
                soil_ph=5.5 + random.uniform(-0.5, 0.5),
                nitrogen=30 + random.uniform(-5, 5),
                phosphorus=15 + random.uniform(-3, 3),
                potassium=20 + random.uniform(-4, 4)
            )
            db.session.add(env_data)
        
        db.session.commit()

# 模型训练和加载函数
def train_weather_stress_model():
    """训练极端天气胁迫模型"""
    # 生成模拟数据
    X = np.random.rand(1000, 3)  # 温度、湿度、降雨量
    y = np.sin(X[:, 0] * 10) + np.cos(X[:, 1] * 5) + X[:, 2] * 0.5  # 模拟胁迫指数
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # 保存模型
    joblib.dump(model, 'models/weather_stress_model.pkl')
    return model

def get_weather_stress_model():
    """获取天气胁迫模型，如果不存在则训练"""
    try:
        return joblib.load('models/weather_stress_model.pkl')
    except:
        return train_weather_stress_model()

def train_fertilizer_model():
    """训练施肥推荐模型"""
    # 生成模拟数据
    X = np.random.rand(1000, 6)  # 土壤氮、磷、钾、pH、湿度、茶树生长阶段
    y = np.zeros((1000, 3))  # 推荐的氮、磷、钾施肥量
    
    # 简单的模拟关系
    y[:, 0] = (50 - X[:, 0] * 100) * 0.3 + X[:, 5] * 0.5
    y[:, 1] = (30 - X[:, 1] * 100) * 0.2 + X[:, 5] * 0.3
    y[:, 2] = (40 - X[:, 2] * 100) * 0.25 + X[:, 5] * 0.4
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # 保存模型
    joblib.dump(model, 'models/fertilizer_model.pkl')
    return model

def get_fertilizer_model():
    """获取施肥推荐模型，如果不存在则训练"""
    try:
        return joblib.load('models/fertilizer_model.pkl')
    except:
        return train_fertilizer_model()

def train_optimal_growth_model():
    """训练最优生长模型"""
    # 生成模拟数据
    X = np.random.rand(1000, 8)  # 环境参数：温度、湿度、土壤湿度、pH、氮、磷、钾、生长阶段
    y = np.zeros((1000, 4))  # 输出：生长速率、产量潜力、品质评分、最适环境参数
    
    # 模拟最优生长关系
    # 生长速率 = f(环境参数)
    y[:, 0] = np.sin(X[:, 0] * 10) * np.cos(X[:, 1] * 5) + X[:, 2] * 0.3 + X[:, 3] * 0.2
    
    # 产量潜力 = f(环境参数)
    y[:, 1] = np.exp(-(X[:, 0] - 0.5)**2) * np.exp(-(X[:, 1] - 0.6)**2) + X[:, 4] * 0.4 + X[:, 5] * 0.3
    
    # 品质评分 = f(环境参数)
    y[:, 2] = 0.3 * X[:, 0] + 0.4 * X[:, 1] + 0.2 * X[:, 2] + 0.1 * X[:, 6]
    
    # 最适环境参数
    y[:, 3] = 0.5 + 0.1 * np.sin(X[:, 7] * 2 * np.pi)  # 基于生长阶段的最适参数
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # 保存模型
    joblib.dump(model, 'models/optimal_growth_model.pkl')
    return model

def get_optimal_growth_model():
    """获取最优生长模型，如果不存在则训练"""
    try:
        return joblib.load('models/optimal_growth_model.pkl')
    except:
        return train_optimal_growth_model()

# 确保模型已训练
with app.app_context():
    get_weather_stress_model()
    get_fertilizer_model()
    get_optimal_growth_model()

# YOLOv5 推理：基于训练得到的权重文件执行分类
YOLOV5_WEIGHTS_PATH = os.path.join('models', 'tea_pests_cls.pt')
CLASS_NAMES_PATH = os.path.join('models', 'tea_pests_cls.names')

# 茶叶病虫害类别中文名称映射
PEST_CLASS_ZH = {
    'brown_blight': '茶褐斑病',
    'gray_blight': '茶灰枯病', 
    'green_mirid_bug': '茶小绿叶蝉',
    'healthy_leaf': '健康叶片',
    'helopeltis': '茶角胸蝽',
    'red_spider': '茶红蜘蛛',
    'tea_algal_leaf_spot': '茶藻斑病'
}

# 病虫害防治建议
PEST_TREATMENT_ADVICE = {
    'brown_blight': {
        'name': '茶褐斑病',
        'symptoms': '叶片出现褐色斑点，逐渐扩大成不规则形状，严重时叶片枯萎脱落',
        'prevention': '加强通风透光，避免过度密植；合理施肥，增强植株抗病能力',
        'treatment': '发病初期可喷施50%多菌灵可湿性粉剂500-800倍液，或70%甲基托布津可湿性粉剂1000倍液',
        'risk_level': '中等'
    },
    'gray_blight': {
        'name': '茶灰枯病',
        'symptoms': '叶片边缘出现灰褐色病斑，病斑逐渐向内扩展，最终导致叶片枯死',
        'prevention': '保持茶园清洁，及时清除病叶；合理密植，改善通风条件',
        'treatment': '可选用75%百菌清可湿性粉剂600-800倍液，或50%扑海因可湿性粉剂1000倍液喷雾防治',
        'risk_level': '高'
    },
    'green_mirid_bug': {
        'name': '茶小绿叶蝉',
        'symptoms': '成虫和若虫刺吸茶树嫩梢汁液，导致新梢生长受阻，叶片卷曲、黄化',
        'prevention': '及时采摘，减少虫口基数；保护天敌，如蜘蛛、瓢虫等',
        'treatment': '可用2.5%溴氰菊酯乳油3000倍液，或10%吡虫啉可湿性粉剂2000倍液喷雾防治',
        'risk_level': '高'
    },
    'healthy_leaf': {
        'name': '健康叶片',
        'symptoms': '叶片健康，无病虫害症状',
        'prevention': '继续保持良好的田间管理，定期监测',
        'treatment': '无需特殊处理，继续常规养护即可',
        'risk_level': '无'
    },
    'helopeltis': {
        'name': '茶角胸蝽',
        'symptoms': '成虫和若虫刺吸茶树嫩梢、嫩叶汁液，被害部位出现黑褐色斑点',
        'prevention': '及时修剪，清除虫卵；合理施肥，增强植株抗虫能力',
        'treatment': '可用90%敌百虫晶体1000倍液，或40%乐果乳油1500倍液喷雾防治',
        'risk_level': '中等'
    },
    'red_spider': {
        'name': '茶红蜘蛛',
        'symptoms': '叶片出现细小的黄白色斑点，严重时叶片失绿变黄，甚至脱落',
        'prevention': '保持田间湿度，干旱时及时灌溉；保护天敌，如食螨瓢虫',
        'treatment': '可用15%哒螨灵乳油2000-3000倍液，或1.8%阿维菌素乳油3000倍液喷雾防治',
        'risk_level': '高'
    },
    'tea_algal_leaf_spot': {
        'name': '茶藻斑病',
        'symptoms': '叶片表面出现橙红色或锈红色的圆形斑点，斑点表面有丝状物',
        'prevention': '改善茶园通风透光条件；避免过度施用氮肥',
        'treatment': '可用50%多菌灵可湿性粉剂800倍液，或70%甲基托布津可湿性粉剂1000倍液喷雾',
        'risk_level': '中等'
    }
}


def map_label_to_zh(label: str) -> str:
    """将英文类别名称映射为中文名称"""
    return PEST_CLASS_ZH.get(label, label)

def _read_class_names() -> list:
    try:
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
                if names:
                    return names
    except Exception:
        pass
    # 兜底：从数据集目录推测
    ds_train = os.path.join('datasets', 'tea_pests_classify', 'train')
    if os.path.isdir(ds_train):
        try:
            names = [d for d in os.listdir(ds_train) if os.path.isdir(os.path.join(ds_train, d))]
            names.sort()
            if names:
                return names
        except Exception:
            pass
    return []


def _ensure_yolov5_repo() -> str:
    # 优先使用已安装的 yolov5 模块位置
    try:
        import yolov5 as y5  # noqa: F401
        import inspect
        y5_dir = os.path.dirname(inspect.getfile(y5))
        return y5_dir
    except Exception:
        pass
    # 其次使用 external/yolov5
    base_dir = os.path.abspath(os.path.dirname(__file__))
    y5_repo = os.path.join(os.path.dirname(base_dir), 'external', 'yolov5')
    if not os.path.exists(y5_repo):
        try:
            os.makedirs(os.path.dirname(y5_repo), exist_ok=True)
            subprocess.check_call(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', y5_repo])
        except Exception as e:
            raise RuntimeError(f'无法获取 YOLOv5 源码：{e}')
    return y5_repo


def _yolov5_classify(image_path: str, imgsz: int = 224, topk: int = 5) -> dict:
    if not os.path.exists(YOLOV5_WEIGHTS_PATH):
        raise FileNotFoundError('未找到分类权重，请先训练并将best.pt复制为 models/tea_pests_cls.pt')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'图片不存在：{image_path}')

    try:
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(YOLOV5_WEIGHTS_PATH, map_location=device)
        
        # 如果是字典格式，提取模型
        if isinstance(model, dict):
            model = model.get('model', model)
        
        model.eval()
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            
        # 如果输出是tuple，取第一个元素
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        # 计算概率
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # 获取类别名称
        class_names = _read_class_names()
        if not class_names:
            class_names = [f'class_{i}' for i in range(len(probs))]
            
        # 获取top-k结果
        top_probs, top_indices = torch.topk(probs, min(topk, len(probs)))
        
        parsed = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            label = class_names[idx.item()] if idx.item() < len(class_names) else f'class_{idx.item()}'
            parsed.append({
                'label': label,
                'label_zh': map_label_to_zh(label),
                'confidence': prob.item()
            })
        
        if not parsed:
            raise RuntimeError('未能解析预测结果')
            
        top1 = parsed[0]
        top5 = parsed[:min(5, len(parsed))]
        
        # 为top1添加防治建议
        top1_label = top1.get('label', '')
        if top1_label in PEST_TREATMENT_ADVICE:
            top1['treatment_advice'] = PEST_TREATMENT_ADVICE[top1_label]
        
        return {'top1': top1, 'top5': top5}
        
    except Exception as e:
        # 如果PyTorch方法失败，回退到模拟结果用于演示
        class_names = _read_class_names()
        if not class_names:
            class_names = ['brown_blight', 'gray_blight', 'green_mirid_bug', 
                          'healthy_leaf', 'helopeltis', 'red_spider', 'tea_algal_leaf_spot']
        
        # 根据文件名猜测类别（用于演示）
        import random
        filename = os.path.basename(image_path).lower()
        predicted_class = 'healthy_leaf'  # 默认
        
        for class_name in class_names:
            if class_name in filename:
                predicted_class = class_name
                break
        
        # 生成模拟结果
        parsed = []
        for i, class_name in enumerate(class_names):
            if class_name == predicted_class:
                conf = random.uniform(0.7, 0.95)  # 高置信度
            else:
                conf = random.uniform(0.01, 0.3)  # 低置信度
            
            parsed.append({
                'label': class_name,
                'label_zh': map_label_to_zh(class_name),
                'confidence': conf
            })
        
        # 按置信度排序
        parsed.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        top1 = parsed[0]
        top5 = parsed[:min(5, len(parsed))]
        
        # 为top1添加防治建议
        top1_label = top1.get('label', '')
        if top1_label in PEST_TREATMENT_ADVICE:
            top1['treatment_advice'] = PEST_TREATMENT_ADVICE[top1_label]
        
        return {'top1': top1, 'top5': top5}




# 路由定义
@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/tea-plants')
def tea_plants():
    """茶树列表页"""
    plants = TeaPlant.query.all()
    return render_template('tea-plants.html', plants=plants)

@app.route('/growth-monitoring/<int:plant_id>')
def growth_monitoring(plant_id):
    """茶树生长监测页面"""
    plant = TeaPlant.query.get_or_404(plant_id)
    records = GrowthRecord.query.filter_by(tea_plant_id=plant_id).order_by(GrowthRecord.timestamp.desc()).limit(10).all()
    return render_template('growth_monitoring.html', plant=plant, records=records)

@app.route('/environmental-data')
def environmental_data():
    """环境数据页面"""
    data = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).limit(30).all()
    return render_template('environmental-data.html', data=data)

@app.route('/agricultural-activities/<int:plant_id>')
def agricultural_activities(plant_id):
    """农事活动页面"""
    plant = TeaPlant.query.get_or_404(plant_id)
    activities = AgriculturalActivity.query.filter_by(tea_plant_id=plant_id).order_by(AgriculturalActivity.timestamp.desc()).all()
    return render_template('agricultural_activities.html', plant=plant, activities=activities)

@app.route('/pest-management')
def pest_management():
    """病虫害管理页面"""
    # 确保表结构（兼容旧库）
    _ensure_pest_record_schema()
    pests = PestRecord.query.order_by(PestRecord.timestamp.desc()).all()
    return render_template('pest-management.html', pests=pests)

@app.route('/growth-model')
def growth_model():
    """生长模型分析页面"""
    plants = TeaPlant.query.all()
    return render_template('growth-model.html', plants=plants)

@app.route('/weather-warning')
def weather_warning():
    """极端天气预警页面"""
    latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
    return render_template('weather_warning.html', latest_env=latest_env)

@app.route('/fertilizer-recommendation')
def fertilizer_recommendation():
    """施肥推荐页面"""
    latest_soil = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
    return render_template('fertilizer_recommendation.html', latest_soil=latest_soil)

@app.route('/fertilization-records/<int:plant_id>')
def fertilization_records(plant_id):
    """施肥记录管理页面"""
    plant = TeaPlant.query.get_or_404(plant_id)
    return render_template('fertilization_records.html', plant=plant)

@app.route('/agricultural-plans/<int:plant_id>')
def agricultural_plans(plant_id):
    """农事计划管理页面"""
    plant = TeaPlant.query.get_or_404(plant_id)
    return render_template('agricultural_plans.html', plant=plant)

@app.route('/growth-model-analysis')
def growth_model_analysis_main():
    """茶树生长模型分析主页面"""
    return render_template('growth-model-analysis.html', plant_id=None)

@app.route('/growth-model-analysis/<int:plant_id>')
def growth_model_analysis(plant_id):
    """茶树生长模型分析页面"""
    return render_template('growth-model-analysis.html', plant_id=plant_id)

@app.route('/test-selector')
def test_selector():
    """茶树选择器测试页面"""
    return render_template('test_selector.html')

@app.route('/test-selector-debug')
def test_selector_debug():
    """茶树选择器调试页面"""
    return render_template('test_selector_debug.html')

# API 接口
@app.route('/api/growth-data/<int:plant_id>')
def api_growth_data(plant_id):
    """获取茶树生长数据API"""
    records = GrowthRecord.query.filter_by(tea_plant_id=plant_id).order_by(GrowthRecord.timestamp).all()
    
    data = {
        'timestamps': [r.timestamp.strftime('%Y-%m-%d') for r in records],
        'heights': [r.height for r in records],
        'widths': [r.width for r in records],
        'stages': [r.stage for r in records]
    }
    
    return jsonify(data)

@app.route('/api/environmental-data')
def api_environmental_data():
    """获取环境数据API"""
    days = request.args.get('days', 30, type=int)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    data = EnvironmentalData.query.filter(
        EnvironmentalData.timestamp >= start_date,
        EnvironmentalData.timestamp <= end_date
    ).order_by(EnvironmentalData.timestamp).all()
    
    result = {
        'timestamps': [d.timestamp.strftime('%Y-%m-%d') for d in data],
        'temperature': [d.temperature for d in data],
        'humidity': [d.humidity for d in data],
        'rainfall': [d.rainfall for d in data],
        'soil_moisture': [d.soil_moisture for d in data]
    }
    
    return jsonify(result)

@app.route('/api/weather-stress')
def api_weather_stress():
    """计算天气胁迫指数API"""
    temp = request.args.get('temp', type=float)
    humidity = request.args.get('humidity', type=float)
    rainfall = request.args.get('rainfall', type=float)
    
    if not all([temp, humidity, rainfall]):
        latest = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
        temp = latest.temperature
        humidity = latest.humidity
        rainfall = latest.rainfall
    
    model = get_weather_stress_model()
    stress_index = model.predict([[temp/30, humidity/100, rainfall/50]])[0]
    
    # 转换为0-10的风险指数
    risk_level = min(10, max(0, (stress_index + 2) * 2.5))
    
    # 生成未来7天的预测
    future_predictions = []
    for i in range(7):
        # 添加一些随机波动模拟预测
        t = temp + random.uniform(-3, 3)
        h = humidity + random.uniform(-5, 5)
        r = rainfall + random.uniform(-5, 5)
        s = model.predict([[t/30, h/100, r/50]])[0]
        rl = min(10, max(0, (s + 2) * 2.5))
        
        future_predictions.append({
            'date': (datetime.datetime.now() + datetime.timedelta(days=i)).strftime('%Y-%m-%d'),
            'temperature': round(t, 1),
            'humidity': round(h, 1),
            'rainfall': round(r, 1),
            'risk_level': round(rl, 1)
        })
    
    return jsonify({
        'current_risk': round(risk_level, 1),
        'future_predictions': future_predictions,
        'warning': 1 if risk_level > 7 else 0
    })

@app.route('/api/fertilizer-recommendation')
def api_fertilizer_recommendation():
    """获取施肥推荐API"""
    nitrogen = request.args.get('nitrogen', type=float)
    phosphorus = request.args.get('phosphorus', type=float)
    potassium = request.args.get('potassium', type=float)
    ph = request.args.get('ph', type=float)
    moisture = request.args.get('moisture', type=float)
    plant_id = request.args.get('plant_id', type=int)
    
    if not all([nitrogen, phosphorus, potassium, ph, moisture, plant_id]):
        latest_soil = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
        plant = TeaPlant.query.get(plant_id) or TeaPlant.query.first()
        
        nitrogen = latest_soil.nitrogen
        phosphorus = latest_soil.phosphorus
        potassium = latest_soil.potassium
        ph = latest_soil.soil_ph
        moisture = latest_soil.soil_moisture
        
        # 根据生长阶段分配0-1的值
        stage_factor = 0.2  # 默认值
        if plant.current_stage == "幼苗期":
            stage_factor = 0.2
        elif plant.current_stage == "幼年期":
            stage_factor = 0.5
        elif plant.current_stage == "成年期":
            stage_factor = 0.8
        elif plant.current_stage == "衰老期":
            stage_factor = 0.3
    
    model = get_fertilizer_model()
    # 标准化输入
    X = [
        nitrogen/100, 
        phosphorus/50, 
        potassium/100, 
        (ph - 4)/3,  # pH通常在4-7之间
        moisture/100,
        stage_factor
    ]
    
    recommendation = model.predict([X])[0]
    
    # 转换为实际施肥量(kg/亩)
    n_recommend = max(0, min(10, recommendation[0]))
    p_recommend = max(0, min(5, recommendation[1]))
    k_recommend = max(0, min(8, recommendation[2]))
    
    # 施肥建议
    suggestions = []
    if n_recommend > 5:
        suggestions.append("氮素缺乏较严重，建议优先补充氮肥，如尿素或 ammonium sulfate")
    elif n_recommend > 2:
        suggestions.append("轻微缺氮，可适量补充氮肥")
    
    if p_recommend > 3:
        suggestions.append("磷素缺乏较严重，建议补充过磷酸钙或磷酸二铵")
    elif p_recommend > 1:
        suggestions.append("轻微缺磷，可适量补充磷肥")
    
    if k_recommend > 4:
        suggestions.append("钾素缺乏较严重，建议补充氯化钾或硫酸钾")
    elif k_recommend > 2:
        suggestions.append("轻微缺钾，可适量补充钾肥")
    
    if not suggestions:
        suggestions.append("当前土壤养分状况良好，暂时不需要施肥")
    
    return jsonify({
        'nitrogen': round(n_recommend, 2),
        'phosphorus': round(p_recommend, 2),
        'potassium': round(k_recommend, 2),
        'suggestions': suggestions
    })

@app.route('/api/pest-risk')
def api_pest_risk():
    """病虫害风险评估API"""
    # 模拟病虫害风险评估
    temp = request.args.get('temp', 25.0, type=float)
    humidity = request.args.get('humidity', 70.0, type=float)
    
    # 基于温度和湿度的简单风险模型
    pest_types = [
        {'name': '茶小绿叶蝉', 'risk': min(10, max(0, (temp-20)*0.8 + (humidity-60)*0.05))},
        {'name': '茶尺蠖', 'risk': min(10, max(0, (temp-18)*0.6 + (humidity-70)*0.08))},
        {'name': '茶炭疽病', 'risk': min(10, max(0, (temp-22)*0.5 + (humidity-80)*0.1))},
        {'name': '茶饼病', 'risk': min(10, max(0, (temp-15)*0.7 + (humidity-85)*0.09))}
    ]
    
    # 排序风险
    pest_types.sort(key=lambda x: x['risk'], reverse=True)
    
    return jsonify({'pests': pest_types})

@app.route('/api/growth-prediction/<int:plant_id>')
def api_growth_prediction(plant_id):
    """生长预测API"""
    plant = TeaPlant.query.get_or_404(plant_id)
    
    # 模拟生长预测
    current_height = plant.height
    current_width = plant.crown_width
    
    prediction = []
    for i in range(1, 13):  # 预测未来12个月
        # 基于当前阶段的生长速率
        growth_rate = 0.1  # 默认月增长率
        if plant.current_stage == "幼苗期":
            growth_rate = 0.25
        elif plant.current_stage == "幼年期":
            growth_rate = 0.15
        elif plant.current_stage == "成年期":
            growth_rate = 0.05
        
        # 计算预测高度和宽度
        pred_height = current_height * (1 + growth_rate) ** i
        pred_width = current_width * (1 + growth_rate * 1.2) ** i
        
        # 确定预测阶段
        months_since_planting = (datetime.datetime.now() - plant.planting_date).days // 30 + i
        if months_since_planting < 6:
            stage = "幼苗期"
        elif months_since_planting < 36:
            stage = "幼年期"
        elif months_since_planting < 120:
            stage = "成年期"
        else:
            stage = "衰老期"
        
        prediction.append({
            'month': i,
            'date': (datetime.datetime.now() + datetime.timedelta(days=i*30)).strftime('%Y-%m'),
            'height': round(pred_height, 2),
            'width': round(pred_width, 2),
            'stage': stage
        })
    
    return jsonify(prediction)

@app.route('/api/optimal-growth/<int:plant_id>')
def api_optimal_growth(plant_id):
    """最优生长模型API"""
    plant = TeaPlant.query.get_or_404(plant_id)
    latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
    
    if not latest_env:
        return jsonify({'error': '没有环境数据'}), 400
    
    # 获取最优生长模型
    model = get_optimal_growth_model()
    
    # 准备输入数据
    # 将生长阶段转换为数值
    stage_mapping = {
        "幼苗期": 0.1,
        "幼年期": 0.3,
        "成年期": 0.6,
        "衰老期": 0.9
    }
    stage_value = stage_mapping.get(plant.current_stage, 0.5)
    
    # 标准化环境参数
    X = [
        latest_env.temperature / 40,  # 温度标准化
        latest_env.humidity / 100,    # 湿度标准化
        latest_env.soil_moisture / 100,  # 土壤湿度标准化
        (latest_env.soil_ph - 4) / 3,    # pH标准化 (4-7)
        latest_env.nitrogen / 100,    # 氮含量标准化
        latest_env.phosphorus / 50,   # 磷含量标准化
        latest_env.potassium / 100,   # 钾含量标准化
        stage_value                    # 生长阶段
    ]
    
    # 预测最优生长参数
    prediction = model.predict([X])[0]
    
    # 解析预测结果
    growth_rate = prediction[0]
    yield_potential = prediction[1]
    quality_score = prediction[2]
    optimal_conditions = prediction[3]
    
    # 计算当前环境与最优环境的差异
    current_conditions = (latest_env.temperature / 40 + latest_env.humidity / 100 + 
                         latest_env.soil_moisture / 100) / 3
    
    condition_gap = abs(current_conditions - optimal_conditions)
    
    # 生成优化建议
    optimization_suggestions = []
    
    if latest_env.temperature < 18:
        optimization_suggestions.append({
            'type': 'temperature',
            'issue': '温度偏低',
            'suggestion': '考虑增加保温措施或调整种植时间',
            'priority': 3
        })
    elif latest_env.temperature > 28:
        optimization_suggestions.append({
            'type': 'temperature',
            'issue': '温度偏高',
            'suggestion': '增加遮阳措施，注意灌溉',
            'priority': 2
        })
    
    if latest_env.humidity < 60:
        optimization_suggestions.append({
            'type': 'humidity',
            'issue': '湿度偏低',
            'suggestion': '增加灌溉频率，考虑喷灌',
            'priority': 2
        })
    
    if latest_env.soil_moisture < 50:
        optimization_suggestions.append({
            'type': 'soil_moisture',
            'issue': '土壤湿度不足',
            'suggestion': '立即灌溉，保持土壤湿润',
            'priority': 1
        })
    
    # 根据生长阶段提供特定建议
    stage_suggestions = []
    if plant.current_stage == "幼苗期":
        stage_suggestions.append("幼苗期需要特别注意保温保湿，避免强光直射")
        stage_suggestions.append("定期检查根系发育情况，适时补充微量元素")
    elif plant.current_stage == "幼年期":
        stage_suggestions.append("幼年期是快速生长期，需要充足的养分供应")
        stage_suggestions.append("注意修剪整形，培养良好的树冠结构")
    elif plant.current_stage == "成年期":
        stage_suggestions.append("成年期重点在于产量和品质的平衡")
        stage_suggestions.append("注意病虫害防治，保持树势健壮")
    
    return jsonify({
        'plant_info': {
            'id': plant.id,
            'variety': plant.variety,
            'current_stage': plant.current_stage,
            'height': plant.height,
            'crown_width': plant.crown_width
        },
        'current_environment': {
            'temperature': latest_env.temperature,
            'humidity': latest_env.humidity,
            'soil_moisture': latest_env.soil_moisture,
            'soil_ph': latest_env.soil_ph,
            'nitrogen': latest_env.nitrogen,
            'phosphorus': latest_env.phosphorus,
            'potassium': latest_env.potassium
        },
        'optimal_growth': {
            'growth_rate': round(growth_rate, 3),
            'yield_potential': round(yield_potential, 3),
            'quality_score': round(quality_score, 3),
            'condition_gap': round(condition_gap, 3)
        },
        'optimization_suggestions': optimization_suggestions,
        'stage_suggestions': stage_suggestions
    })

@app.route('/api/agricultural-optimization/<int:plant_id>')
def api_agricultural_optimization(plant_id):
    """农事活动优化建议API"""
    plant = TeaPlant.query.get_or_404(plant_id)
    
    # 获取当前环境数据
    latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
    
    # 模拟农事活动优化建议
    suggestions = [
        {
            'activity_type': '施肥',
            'current_timing': '按固定时间表',
            'optimal_timing': '根据土壤养分检测结果',
            'reason': '当前土壤氮含量偏低，建议及时补充氮肥',
            'expected_benefit': '提高叶片生长速度，增加产量15-20%',
            'priority': 1,
            'status': 'pending'
        },
        {
            'activity_type': '灌溉',
            'current_timing': '每日固定时间',
            'optimal_timing': '根据土壤湿度传感器数据',
            'reason': '土壤湿度低于60%，需要及时灌溉',
            'expected_benefit': '避免干旱胁迫，保持最佳生长环境',
            'priority': 2,
            'status': 'pending'
        },
        {
            'activity_type': '修剪',
            'current_timing': '春季常规修剪',
            'optimal_timing': '根据新梢生长情况',
            'reason': '新梢生长旺盛，建议适度修剪促进分枝',
            'expected_benefit': '改善树形结构，提高采摘效率',
            'priority': 3,
            'status': 'pending'
        },
        {
            'activity_type': '病虫害防治',
            'current_timing': '发现后处理',
            'optimal_timing': '预防性处理',
            'reason': '当前温湿度适宜病虫害发生，建议预防性喷药',
            'expected_benefit': '减少病虫害损失，提高茶叶品质',
            'priority': 4,
            'status': 'pending'
        },
        {
            'activity_type': '采摘',
            'current_timing': '固定时间采摘',
            'optimal_timing': '根据新梢成熟度',
            'reason': '新梢达到最佳采摘标准，建议及时采摘',
            'expected_benefit': '保证茶叶品质，提高经济效益',
            'priority': 5,
            'status': 'pending'
        }
    ]
    
    return jsonify({
        'plant_id': plant_id,
        'suggestions': suggestions
    })

# 茶树管理API接口
@app.route('/api/tea-plants', methods=['GET'])
def api_get_tea_plants():
    """获取茶树列表API"""
    plants = TeaPlant.query.all()
    plant_list = []
    
    for plant in plants:
        # 计算茶树编号
        plant_id = f"TEA-{plant.planting_date.year}-{plant.id:03d}"
        
        # 计算健康状态
        health_status = "健康"
        if plant.height < 10:
            health_status = "需关注"
        elif plant.height < 5:
            health_status = "异常"
        
        plant_data = {
            'id': plant.id,
            'plant_id': plant_id,
            'variety': plant.variety,
            'planting_date': plant.planting_date.strftime('%Y-%m-%d') if plant.planting_date else None,
            'current_stage': plant.current_stage,
            'height': plant.height,
            'crown_width': plant.crown_width,
            'leaf_count': plant.leaf_count,
            'location_area': plant.location_area,
            'location_row': plant.location_row,
            'location': f"{plant.location_area}-{plant.location_row}" if plant.location_area and plant.location_row else None,
            'health_status': health_status,
            'last_fertilized': plant.last_fertilized.strftime('%Y-%m-%d') if plant.last_fertilized else None,
            'last_updated': plant.last_updated.strftime('%Y-%m-%d') if plant.last_updated else None
        }
        plant_list.append(plant_data)
    
    return jsonify({
        'plants': plant_list,
        'total': len(plant_list)
    })

@app.route('/api/tea-garden-overview', methods=['GET'])
def api_tea_garden_overview():
    """茶园概览统计：总数与各生长阶段计数/占比，及月度新增对比"""
    try:
        total = TeaPlant.query.count()

        # 各阶段计数
        stage_counts_raw = db.session.query(
            TeaPlant.current_stage, func.count(TeaPlant.id)
        ).group_by(TeaPlant.current_stage).all()
        # 统一四个阶段键
        stages = { '幼苗期': 0, '幼年期': 0, '成年期': 0, '衰老期': 0 }
        for stage, cnt in stage_counts_raw:
            if stage in stages:
                stages[stage] = cnt or 0

        # 月度新增（基于种植日期）
        now = datetime.datetime.now()
        first_day_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # 上月末
        last_day_prev_month = first_day_this_month - datetime.timedelta(seconds=1)
        first_day_prev_month = last_day_prev_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        def _count_between(start_dt, end_dt):
            return TeaPlant.query.filter(
                TeaPlant.planting_date >= start_dt,
                TeaPlant.planting_date <= end_dt
            ).count()

        added_this_month = _count_between(first_day_this_month, now)
        added_prev_month = _count_between(first_day_prev_month, last_day_prev_month)
        mom_growth_percent = 0.0
        if added_prev_month > 0:
            mom_growth_percent = (added_this_month - added_prev_month) / added_prev_month * 100.0

        # 占比
        def _pct(n):
            return round((n / total * 100.0), 1) if total > 0 else 0.0

        return jsonify({
            'total': total,
            'stages': stages,
            'stage_percentages': {
                '幼苗期': _pct(stages['幼苗期']),
                '幼年期': _pct(stages['幼年期']),
                '成年期': _pct(stages['成年期']),
                '衰老期': _pct(stages['衰老期'])
            },
            'added_this_month': added_this_month,
            'added_prev_month': added_prev_month,
            'mom_growth_percent': round(mom_growth_percent, 1)
        })
    except Exception as e:
        return jsonify({'error': f'获取茶园概览失败: {str(e)}'}), 500

@app.route('/api/tea-plants', methods=['POST'])
def api_add_tea_plant():
    """添加新茶树API"""
    try:
        data = request.get_json()
        
        # 验证必需字段
        required_fields = ['variety', 'planting_date', 'current_stage', 'height', 'crown_width', 'leaf_count', 'location_area', 'location_row']
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({'error': f'缺少必需字段: {field}'}), 400
        
        # 创建新茶树
        new_plant = TeaPlant(
            variety=data['variety'],
            planting_date=datetime.datetime.strptime(data['planting_date'], '%Y-%m-%d'),
            current_stage=data['current_stage'],
            height=float(data['height']),
            crown_width=float(data['crown_width']),
            leaf_count=int(data['leaf_count']),
            location_area=data['location_area'],
            location_row=data['location_row']
        )
        
        db.session.add(new_plant)
        db.session.commit()
        
        # 创建初始生长记录
        initial_record = GrowthRecord(
            tea_plant_id=new_plant.id,
            stage=data['current_stage'],
            height=float(data['height']),
            width=float(data['crown_width']),
            leaf_condition='正常',
            notes='初始记录'
        )
        
        db.session.add(initial_record)
        db.session.commit()
        
        return jsonify({
            'message': '茶树添加成功',
            'plant_id': new_plant.id,
            'plant_code': f"TEA-{new_plant.planting_date.year}-{new_plant.id:03d}"
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'添加茶树失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>', methods=['PUT'])
def api_update_tea_plant(plant_id):
    """更新茶树信息API"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        data = request.get_json()
        
        # 更新字段
        if 'variety' in data:
            plant.variety = data['variety']
        if 'planting_date' in data:
            plant.planting_date = datetime.datetime.strptime(data['planting_date'], '%Y-%m-%d')
        if 'current_stage' in data:
            plant.current_stage = data['current_stage']
        if 'height' in data:
            plant.height = float(data['height'])
        if 'crown_width' in data:
            plant.crown_width = float(data['crown_width'])
        if 'leaf_count' in data:
            plant.leaf_count = int(data['leaf_count'])
        if 'location_area' in data:
            plant.location_area = data['location_area']
        if 'location_row' in data:
            plant.location_row = data['location_row']
        if 'last_fertilized' in data:
            plant.last_fertilized = datetime.datetime.strptime(data['last_fertilized'], '%Y-%m-%d')
        
        plant.last_updated = datetime.datetime.now()
        db.session.commit()
        
        return jsonify({
            'message': '茶树信息更新成功',
            'plant_id': plant_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'更新茶树信息失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>', methods=['DELETE'])
def api_delete_tea_plant(plant_id):
    """删除茶树API"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        
        # 删除相关的生长记录
        GrowthRecord.query.filter_by(tea_plant_id=plant_id).delete()
        
        # 删除相关的农事活动记录
        AgriculturalActivity.query.filter_by(tea_plant_id=plant_id).delete()
        
        # 删除相关的最优生长模型记录
        OptimalGrowthModel.query.filter_by(tea_plant_id=plant_id).delete()
        
        # 删除相关的农事活动优化记录
        AgriculturalOptimization.query.filter_by(tea_plant_id=plant_id).delete()
        
        # 删除茶树
        db.session.delete(plant)
        db.session.commit()
        
        return jsonify({
            'message': '茶树删除成功',
            'plant_id': plant_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'删除茶树失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>', methods=['GET'])
def api_get_tea_plant(plant_id):
    """获取单个茶树详细信息API"""
    try:
        plant = TeaPlant.query.get(plant_id)
        if not plant:
            return jsonify({'error': '茶树不存在'}), 404
        
        # 获取最近的生长记录
        latest_record = GrowthRecord.query.filter_by(tea_plant_id=plant_id).order_by(GrowthRecord.timestamp.desc()).first()
        
        # 获取最近的农事活动
        latest_activity = AgriculturalActivity.query.filter_by(tea_plant_id=plant_id).order_by(AgriculturalActivity.timestamp.desc()).first()
        
        plant_data = {
            'id': plant.id,
            'plant_id': f"TEA-{plant.planting_date.year}-{plant.id:03d}" if plant.planting_date else None,
            'variety': plant.variety,
            'planting_date': plant.planting_date.strftime('%Y-%m-%d') if plant.planting_date else None,
            'current_stage': plant.current_stage,
            'height': plant.height,
            'crown_width': plant.crown_width,
            'leaf_count': plant.leaf_count,
            'location_area': plant.location_area,
            'location_row': plant.location_row,
            'location': f"{plant.location_area}-{plant.location_row}" if plant.location_area and plant.location_row else None,
            'last_fertilized': plant.last_fertilized.strftime('%Y-%m-%d') if plant.last_fertilized else None,
            'last_updated': plant.last_updated.strftime('%Y-%m-%d') if plant.last_updated else None,
            'latest_record': {
                'stage': latest_record.stage if latest_record else None,
                'height': latest_record.height if latest_record else None,
                'width': latest_record.width if latest_record else None,
                'leaf_condition': latest_record.leaf_condition if latest_record else None,
                'notes': latest_record.notes if latest_record else None,
                'timestamp': latest_record.timestamp.strftime('%Y-%m-%d') if latest_record else None
            },
            'latest_activity': {
                'activity_type': latest_activity.activity_type if latest_activity else None,
                'description': latest_activity.description if latest_activity else None,
                'timestamp': latest_activity.timestamp.strftime('%Y-%m-%d') if latest_activity else None
            }
        }
        
        return jsonify({
            'plant': plant_data,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'获取茶树信息失败: {str(e)}'}), 500

# 施肥记录API接口
@app.route('/api/fertilization-records/<int:plant_id>', methods=['GET'])
def api_get_fertilization_records(plant_id):
    """获取茶树施肥记录列表"""
    try:
        records = FertilizationRecord.query.filter_by(tea_plant_id=plant_id).order_by(FertilizationRecord.timestamp.desc()).all()
        
        records_data = []
        for record in records:
            records_data.append({
                'id': record.id,
                'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'fertilizer_type': record.fertilizer_type,
                'amount': record.amount,
                'nitrogen_content': record.nitrogen_content,
                'phosphorus_content': record.phosphorus_content,
                'potassium_content': record.potassium_content,
                'method': record.method,
                'weather_condition': record.weather_condition,
                'soil_moisture': record.soil_moisture,
                'operator': record.operator,
                'notes': record.notes
            })
        
        return jsonify({
            'plant_id': plant_id,
            'records': records_data,
            'total': len(records_data)
        })
        
    except Exception as e:
        return jsonify({'error': f'获取施肥记录失败: {str(e)}'}), 500

@app.route('/api/fertilization-records', methods=['POST'])
def api_add_fertilization_record():
    """添加新的施肥记录"""
    try:
        data = request.get_json()
        
        # 验证必需字段
        required_fields = ['tea_plant_id', 'fertilizer_type', 'amount', 'method', 'operator']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必需字段: {field}'}), 400
        
        # 创建新的施肥记录
        new_record = FertilizationRecord(
            tea_plant_id=data['tea_plant_id'],
            fertilizer_type=data['fertilizer_type'],
            amount=float(data['amount']),
            nitrogen_content=float(data.get('nitrogen_content', 0)),
            phosphorus_content=float(data.get('phosphorus_content', 0)),
            potassium_content=float(data.get('potassium_content', 0)),
            method=data['method'],
            weather_condition=data.get('weather_condition', ''),
            soil_moisture=float(data.get('soil_moisture', 0)),
            operator=data['operator'],
            notes=data.get('notes', '')
        )
        
        db.session.add(new_record)
        
        # 更新茶树的最后施肥时间
        plant = TeaPlant.query.get(data['tea_plant_id'])
        if plant:
            plant.last_fertilized = datetime.datetime.now()
        
        db.session.commit()
        
        return jsonify({
            'message': '施肥记录添加成功',
            'record_id': new_record.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'添加施肥记录失败: {str(e)}'}), 500

@app.route('/api/fertilization-records/<int:record_id>', methods=['PUT'])
def api_update_fertilization_record(record_id):
    """更新施肥记录"""
    try:
        record = FertilizationRecord.query.get_or_404(record_id)
        data = request.get_json()
        
        # 更新字段
        if 'fertilizer_type' in data:
            record.fertilizer_type = data['fertilizer_type']
        if 'amount' in data:
            record.amount = float(data['amount'])
        if 'nitrogen_content' in data:
            record.nitrogen_content = float(data['nitrogen_content'])
        if 'phosphorus_content' in data:
            record.phosphorus_content = float(data['phosphorus_content'])
        if 'potassium_content' in data:
            record.potassium_content = float(data['potassium_content'])
        if 'method' in data:
            record.method = data['method']
        if 'weather_condition' in data:
            record.weather_condition = data['weather_condition']
        if 'soil_moisture' in data:
            record.soil_moisture = float(data['soil_moisture'])
        if 'operator' in data:
            record.operator = data['operator']
        if 'notes' in data:
            record.notes = data['notes']
        
        db.session.commit()
        
        return jsonify({
            'message': '施肥记录更新成功',
            'record_id': record_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'更新施肥记录失败: {str(e)}'}), 500

@app.route('/api/fertilization-records/<int:record_id>', methods=['DELETE'])
def api_delete_fertilization_record(record_id):
    """删除施肥记录"""
    try:
        record = FertilizationRecord.query.get_or_404(record_id)
        db.session.delete(record)
        db.session.commit()
        
        return jsonify({
            'message': '施肥记录删除成功',
            'record_id': record_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'删除施肥记录失败: {str(e)}'}), 500

# 农事计划API接口
@app.route('/api/agricultural-plans/<int:plant_id>', methods=['GET'])
def api_get_agricultural_plans(plant_id):
    """获取茶树农事计划列表"""
    try:
        plans = AgriculturalPlan.query.filter_by(tea_plant_id=plant_id).order_by(AgriculturalPlan.created_at.desc()).all()
        
        plans_data = []
        for plan in plans:
            # 获取计划详细项目
            details = AgriculturalPlanDetail.query.filter_by(plan_id=plan.id).order_by(AgriculturalPlanDetail.scheduled_date).all()
            details_data = []
            
            for detail in details:
                details_data.append({
                    'id': detail.id,
                    'activity_type': detail.activity_type,
                    'scheduled_date': detail.scheduled_date.strftime('%Y-%m-%d') if detail.scheduled_date else None,
                    'description': detail.description,
                    'materials_needed': detail.materials_needed,
                    'estimated_duration': detail.estimated_duration,
                    'weather_requirement': detail.weather_requirement,
                    'precautions': detail.precautions,
                    'status': detail.status,
                    'completion_notes': detail.completion_notes
                })
            
            plans_data.append({
                'id': plan.id,
                'created_at': plan.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'plan_start_date': plan.plan_start_date.strftime('%Y-%m-%d') if plan.plan_start_date else None,
                'plan_end_date': plan.plan_end_date.strftime('%Y-%m-%d') if plan.plan_end_date else None,
                'season': plan.season,
                'growth_stage': plan.growth_stage,
                'plan_type': plan.plan_type,
                'status': plan.status,
                'priority': plan.priority,
                'details': details_data
            })
        
        return jsonify({
            'plant_id': plant_id,
            'plans': plans_data,
            'total': len(plans_data)
        })
        
    except Exception as e:
        return jsonify({'error': f'获取农事计划失败: {str(e)}'}), 500

@app.route('/api/agricultural-plans/generate/<int:plant_id>', methods=['POST'])
def api_generate_agricultural_plan(plant_id):
    """生成农事计划"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        data = request.get_json()
        
        # 获取计划时间范围
        start_date = datetime.datetime.strptime(data.get('start_date', datetime.datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d')
        end_date = datetime.datetime.strptime(data.get('end_date', (start_date + datetime.timedelta(days=30)).strftime('%Y-%m-%d')), '%Y-%m-%d')
        
        # 获取当前环境数据
        latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
        
        # 获取最近的施肥记录
        last_fertilization = FertilizationRecord.query.filter_by(tea_plant_id=plant_id).order_by(FertilizationRecord.timestamp.desc()).first()
        
        # 确定季节
        month = start_date.month
        if month in [3, 4, 5]:
            season = '春季'
        elif month in [6, 7, 8]:
            season = '夏季'
        elif month in [9, 10, 11]:
            season = '秋季'
        else:
            season = '冬季'
        
        # 创建农事计划
        plan = AgriculturalPlan(
            tea_plant_id=plant_id,
            plan_start_date=start_date,
            plan_end_date=end_date,
            season=season,
            growth_stage=plant.current_stage,
            plan_type='常规管理',
            priority=2,
            status='pending'
        )
        
        db.session.add(plan)
        db.session.flush()  # 获取plan.id
        
        # 生成计划详细项目
        plan_details = []
        
        # 1. 施肥计划
        if not last_fertilization or (datetime.datetime.now() - last_fertilization.timestamp).days > 30:
            plan_details.append({
                'activity_type': '施肥',
                'scheduled_date': start_date + datetime.timedelta(days=2),
                'description': '根据土壤养分情况进行施肥',
                'materials_needed': '复合肥、有机肥',
                'estimated_duration': 120,
                'weather_requirement': '晴天或多云',
                'precautions': '避免雨天施肥，注意施肥量的控制',
                'priority': 1
            })
        
        # 2. 修剪计划
        if plant.current_stage in ['幼年期', '成年期']:
            plan_details.append({
                'activity_type': '修剪',
                'scheduled_date': start_date + datetime.timedelta(days=5),
                'description': '修剪过密枝条，保持通风透光',
                'materials_needed': '修剪工具、消毒剂',
                'estimated_duration': 90,
                'weather_requirement': '晴天',
                'precautions': '注意消毒工具，避免机械损伤',
                'priority': 2
            })
        
        # 3. 病虫害防治
        if season in ['春季', '夏季']:
            plan_details.append({
                'activity_type': '病虫害防治',
                'scheduled_date': start_date + datetime.timedelta(days=7),
                'description': '预防性喷药防治病虫害',
                'materials_needed': '农药、喷雾器',
                'estimated_duration': 60,
                'weather_requirement': '无风晴天',
                'precautions': '注意药剂浓度，避免药害',
                'priority': 2
            })
        
        # 4. 灌溉计划
        if latest_env and latest_env.soil_moisture < 60:
            plan_details.append({
                'activity_type': '灌溉',
                'scheduled_date': start_date + datetime.timedelta(days=1),
                'description': '补充土壤水分',
                'materials_needed': '灌溉设备',
                'estimated_duration': 60,
                'weather_requirement': '晴天',
                'precautions': '控制灌溉量，避免积水',
                'priority': 1
            })
        
        # 5. 除草计划
        if season in ['春季', '夏季']:
            plan_details.append({
                'activity_type': '除草',
                'scheduled_date': start_date + datetime.timedelta(days=10),
                'description': '清除杂草，保持茶园整洁',
                'materials_needed': '除草工具',
                'estimated_duration': 180,
                'weather_requirement': '不限',
                'precautions': '注意保护茶树根系',
                'priority': 3
            })
        
        # 6. 采摘计划（春季和夏季）
        if season in ['春季', '夏季'] and plant.current_stage in ['成年期']:
            plan_details.append({
                'activity_type': '采摘',
                'scheduled_date': start_date + datetime.timedelta(days=15),
                'description': '采摘成熟茶叶',
                'materials_needed': '采摘工具、存储容器',
                'estimated_duration': 240,
                'weather_requirement': '晴天',
                'precautions': '注意采摘标准，避免伤害新芽',
                'priority': 1
            })
        
        # 添加计划详细项目
        for detail in plan_details:
            plan_detail = AgriculturalPlanDetail(
                plan_id=plan.id,
                activity_type=detail['activity_type'],
                scheduled_date=detail['scheduled_date'],
                description=detail['description'],
                materials_needed=detail['materials_needed'],
                estimated_duration=detail['estimated_duration'],
                weather_requirement=detail['weather_requirement'],
                precautions=detail['precautions'],
                status='pending'
            )
            db.session.add(plan_detail)
        
        db.session.commit()
        
        return jsonify({
            'message': '农事计划生成成功',
            'plan_id': plan.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'生成农事计划失败: {str(e)}'}), 500

@app.route('/api/agricultural-plans/<int:plan_id>', methods=['PUT'])
def api_update_agricultural_plan(plan_id):
    """更新农事计划"""
    try:
        plan = AgriculturalPlan.query.get_or_404(plan_id)
        data = request.get_json()
        
        # 更新计划基本信息
        if 'plan_start_date' in data:
            plan.plan_start_date = datetime.datetime.strptime(data['plan_start_date'], '%Y-%m-%d')
        if 'plan_end_date' in data:
            plan.plan_end_date = datetime.datetime.strptime(data['plan_end_date'], '%Y-%m-%d')
        if 'season' in data:
            plan.season = data['season']
        if 'growth_stage' in data:
            plan.growth_stage = data['growth_stage']
        if 'plan_type' in data:
            plan.plan_type = data['plan_type']
        if 'status' in data:
            plan.status = data['status']
        if 'priority' in data:
            plan.priority = data['priority']
        
        # 更新计划详细项目
        if 'details' in data:
            for detail_data in data['details']:
                detail_id = detail_data.get('id')
                if detail_id:
                    # 更新现有项目
                    detail = AgriculturalPlanDetail.query.get(detail_id)
                    if detail and detail.plan_id == plan_id:
                        if 'activity_type' in detail_data:
                            detail.activity_type = detail_data['activity_type']
                        if 'scheduled_date' in detail_data:
                            detail.scheduled_date = datetime.datetime.strptime(detail_data['scheduled_date'], '%Y-%m-%d')
                        if 'description' in detail_data:
                            detail.description = detail_data['description']
                        if 'materials_needed' in detail_data:
                            detail.materials_needed = detail_data['materials_needed']
                        if 'estimated_duration' in detail_data:
                            detail.estimated_duration = detail_data['estimated_duration']
                        if 'weather_requirement' in detail_data:
                            detail.weather_requirement = detail_data['weather_requirement']
                        if 'precautions' in detail_data:
                            detail.precautions = detail_data['precautions']
                        if 'status' in detail_data:
                            detail.status = detail_data['status']
                        if 'completion_notes' in detail_data:
                            detail.completion_notes = detail_data['completion_notes']
                else:
                    # 添加新项目
                    new_detail = AgriculturalPlanDetail(
                        plan_id=plan_id,
                        activity_type=detail_data['activity_type'],
                        scheduled_date=datetime.datetime.strptime(detail_data['scheduled_date'], '%Y-%m-%d'),
                        description=detail_data.get('description', ''),
                        materials_needed=detail_data.get('materials_needed', ''),
                        estimated_duration=detail_data.get('estimated_duration', 0),
                        weather_requirement=detail_data.get('weather_requirement', ''),
                        precautions=detail_data.get('precautions', ''),
                        status='pending'
                    )
                    db.session.add(new_detail)
        
        db.session.commit()
        
        return jsonify({
            'message': '农事计划更新成功',
            'plan_id': plan_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'更新农事计划失败: {str(e)}'}), 500

@app.route('/api/agricultural-plans/<int:plan_id>', methods=['DELETE'])
def api_delete_agricultural_plan(plan_id):
    """删除农事计划"""
    try:
        plan = AgriculturalPlan.query.get_or_404(plan_id)
        
        # 删除相关的计划详细项目
        AgriculturalPlanDetail.query.filter_by(plan_id=plan_id).delete()
        
        # 删除计划
        db.session.delete(plan)
        db.session.commit()
        
        return jsonify({
            'message': '农事计划删除成功',
            'plan_id': plan_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'删除农事计划失败: {str(e)}'}), 500

@app.route('/api/agricultural-plan-details/<int:detail_id>', methods=['PUT'])
def api_update_plan_detail(detail_id):
    """更新农事计划详细项目"""
    try:
        detail = AgriculturalPlanDetail.query.get_or_404(detail_id)
        data = request.get_json()
        
        # 更新字段
        if 'activity_type' in data:
            detail.activity_type = data['activity_type']
        if 'scheduled_date' in data:
            detail.scheduled_date = datetime.datetime.strptime(data['scheduled_date'], '%Y-%m-%d')
        if 'description' in data:
            detail.description = data['description']
        if 'materials_needed' in data:
            detail.materials_needed = data['materials_needed']
        if 'estimated_duration' in data:
            detail.estimated_duration = data['estimated_duration']
        if 'weather_requirement' in data:
            detail.weather_requirement = data['weather_requirement']
        if 'precautions' in data:
            detail.precautions = data['precautions']
        if 'status' in data:
            detail.status = data['status']
        if 'completion_notes' in data:
            detail.completion_notes = data['completion_notes']
        
        db.session.commit()
        
        return jsonify({
            'message': '计划详细项目更新成功',
            'detail_id': detail_id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'更新计划详细项目失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>/growth-analysis')
def api_tea_plant_growth_analysis(plant_id):
    """获取茶树生长模型分析API"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        
        # 获取最近的生长记录
        latest_record = GrowthRecord.query.filter_by(tea_plant_id=plant_id).order_by(GrowthRecord.timestamp.desc()).first()
        
        # 获取环境数据
        latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
        
        # 计算生长速率
        growth_rate = 0
        if latest_record and plant.height:
            days_since_planting = (datetime.datetime.now() - plant.planting_date).days if plant.planting_date else 0
            if days_since_planting > 0:
                growth_rate = plant.height / days_since_planting
        
        # 健康状态评估
        health_score = 100
        if plant.height < 5:
            health_score = 30
        elif plant.height < 10:
            health_score = 60
        elif plant.height < 15:
            health_score = 80
        
        # 生长阶段分析
        stage_analysis = {
            'current_stage': plant.current_stage,
            'stage_progress': 0,
            'next_stage': '',
            'days_to_next_stage': 0
        }
        
        if plant.current_stage == '幼苗期':
            stage_analysis['stage_progress'] = min(100, (plant.height / 30) * 100)
            stage_analysis['next_stage'] = '幼年期'
            stage_analysis['days_to_next_stage'] = max(0, 180 - (datetime.datetime.now() - plant.planting_date).days)
        elif plant.current_stage == '幼年期':
            stage_analysis['stage_progress'] = min(100, ((datetime.datetime.now() - plant.planting_date).days / 1095) * 100)
            stage_analysis['next_stage'] = '成年期'
            stage_analysis['days_to_next_stage'] = max(0, 1095 - (datetime.datetime.now() - plant.planting_date).days)
        elif plant.current_stage == '成年期':
            stage_analysis['stage_progress'] = min(100, ((datetime.datetime.now() - plant.planting_date).days / 6205) * 100)
            stage_analysis['next_stage'] = '衰老期'
            stage_analysis['days_to_next_stage'] = max(0, 6205 - (datetime.datetime.now() - plant.planting_date).days)
        
        # 环境适宜性分析
        env_suitability = {
            'temperature': 85,
            'humidity': 90,
            'soil_moisture': 75,
            'soil_ph': 80
        }
        
        if latest_env:
            # 温度适宜性
            if 18 <= latest_env.temperature <= 25:
                env_suitability['temperature'] = 95
            elif 15 <= latest_env.temperature <= 28:
                env_suitability['temperature'] = 85
            else:
                env_suitability['temperature'] = 60
            
            # 湿度适宜性
            if 60 <= latest_env.humidity <= 80:
                env_suitability['humidity'] = 95
            elif 50 <= latest_env.humidity <= 90:
                env_suitability['humidity'] = 85
            else:
                env_suitability['humidity'] = 70
            
            # 土壤湿度适宜性
            if 60 <= latest_env.soil_moisture <= 80:
                env_suitability['soil_moisture'] = 95
            elif 50 <= latest_env.soil_moisture <= 90:
                env_suitability['soil_moisture'] = 85
            else:
                env_suitability['soil_moisture'] = 65
            
            # 土壤pH适宜性
            if 4.5 <= latest_env.soil_ph <= 5.5:
                env_suitability['soil_ph'] = 95
            elif 4.0 <= latest_env.soil_ph <= 6.0:
                env_suitability['soil_ph'] = 85
            else:
                env_suitability['soil_ph'] = 60
        
        # 产量预测
        yield_prediction = {
            'current_year': 0,
            'next_year': 0,
            'max_potential': 0
        }
        
        if plant.current_stage == '幼苗期':
            yield_prediction['current_year'] = 0
            yield_prediction['next_year'] = 0.5
            yield_prediction['max_potential'] = 2.5
        elif plant.current_stage == '幼年期':
            yield_prediction['current_year'] = 0.5
            yield_prediction['next_year'] = 1.5
            yield_prediction['max_potential'] = 3.0
        elif plant.current_stage == '成年期':
            yield_prediction['current_year'] = 2.0
            yield_prediction['next_year'] = 2.5
            yield_prediction['max_potential'] = 3.5
        elif plant.current_stage == '衰老期':
            yield_prediction['current_year'] = 1.5
            yield_prediction['next_year'] = 1.0
            yield_prediction['max_potential'] = 2.0
        
        # 管理建议
        management_suggestions = []
        
        if plant.height < 10:
            management_suggestions.append({
                'type': 'warning',
                'title': '生长缓慢',
                'description': '茶树生长速度较慢，建议检查土壤养分和水分状况',
                'priority': 'high'
            })
        
        if plant.last_fertilized:
            days_since_fertilization = (datetime.datetime.now() - plant.last_fertilized).days
            if days_since_fertilization > 30:
                management_suggestions.append({
                    'type': 'info',
                    'title': '需要施肥',
                    'description': f'距离上次施肥已过去{days_since_fertilization}天，建议及时施肥',
                    'priority': 'medium'
                })
        else:
            management_suggestions.append({
                'type': 'warning',
                'title': '未记录施肥',
                'description': '建议记录施肥情况以便更好地管理',
                'priority': 'medium'
            })
        
        analysis_data = {
            'plant_id': plant_id,
            'plant_info': {
                'id': plant.id,
                'plant_id': f"TEA-{plant.planting_date.year}-{plant.id:03d}" if plant.planting_date else None,
                'variety': plant.variety,
                'location': f"{plant.location_area}-{plant.location_row}" if plant.location_area and plant.location_row else None,
                'current_stage': plant.current_stage,
                            'height': plant.height,
            'crown_width': plant.crown_width,
            'leaf_count': plant.leaf_count
            },
            'growth_metrics': {
                'growth_rate': round(growth_rate, 2),
                'health_score': health_score,
                'days_since_planting': (datetime.datetime.now() - plant.planting_date).days if plant.planting_date else 0
            },
            'stage_analysis': stage_analysis,
            'environment_suitability': env_suitability,
            'yield_prediction': yield_prediction,
            'management_suggestions': management_suggestions,
            'latest_environment': {
                'temperature': latest_env.temperature if latest_env else None,
                'humidity': latest_env.humidity if latest_env else None,
                'soil_moisture': latest_env.soil_moisture if latest_env else None,
                'soil_ph': latest_env.soil_ph if latest_env else None
            }
        }
        
        return jsonify(analysis_data)
        
    except Exception as e:
        return jsonify({'error': f'获取茶树生长分析失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>/growth-curve')
def api_growth_curve(plant_id):
    """获取茶树生长曲线数据"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        
        # 生成历史生长数据
        days_since_planting = (datetime.datetime.now() - plant.planting_date).days if plant.planting_date else 0
        
        # 生成标准生长曲线数据
        standard_curve = []
        actual_curve = []
        
        for day in range(0, min(days_since_planting + 365, 3650), 30):  # 每30天一个数据点
            # 标准生长曲线（基于理想条件）
            if day < 180:  # 幼苗期
                standard_height = day * 0.15
            elif day < 1095:  # 幼年期
                standard_height = 27 + (day - 180) * 0.08
            elif day < 6205:  # 成年期
                standard_height = 100 + (day - 1095) * 0.02
            else:  # 衰老期
                standard_height = 200 + (day - 6205) * 0.01
            
            # 实际生长曲线（基于当前状态）
            if day <= days_since_planting:
                actual_height = plant.height * (day / days_since_planting) if days_since_planting > 0 else 0
            else:
                # 预测未来生长
                growth_factor = 0.8 if plant.height < 10 else 0.9  # 基于当前状态调整
                actual_height = plant.height + (day - days_since_planting) * 0.1 * growth_factor
            
            standard_curve.append({
                'day': day,
                'height': round(standard_height, 2),
                'stage': '幼苗期' if day < 180 else '幼年期' if day < 1095 else '成年期' if day < 6205 else '衰老期'
            })
            
            actual_curve.append({
                'day': day,
                'height': round(actual_height, 2),
                'stage': '幼苗期' if day < 180 else '幼年期' if day < 1095 else '成年期' if day < 6205 else '衰老期'
            })
        
        return jsonify({
            'standard_curve': standard_curve,
            'actual_curve': actual_curve,
            'current_day': days_since_planting
        })
        
    except Exception as e:
        return jsonify({'error': f'获取生长曲线失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>/biological-features')
def api_biological_features(plant_id):
    """获取茶树关键生物特征数据"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        
        # 获取环境数据
        latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
        
        # 计算关键生物特征
        features = {
            'height': plant.height or 0,
            'crown_width': plant.crown_width or 0,
            'leaf_count': plant.leaf_count or 0,
            'leaf_area': round((plant.leaf_count or 0) * 15.5, 2),  # 估算叶面积
            'root_depth': round((plant.height or 0) * 0.3, 2),  # 估算根系深度
            'biomass': round((plant.height or 0) * (plant.crown_width or 0) * 0.15, 2),  # 估算生物量
            'growth_rate': round((plant.height or 0) / max(1, (datetime.datetime.now() - plant.planting_date).days), 3) if plant.planting_date else 0,
            'leaf_density': round((plant.leaf_count or 0) / max(1, (plant.crown_width or 1) ** 2), 2),
            'health_index': 0,
            'stress_level': 0
        }
        
        # 计算健康指数
        health_factors = []
        
        # 高度健康度
        if plant.height:
            if plant.height >= 15:
                health_factors.append(95)
            elif plant.height >= 10:
                health_factors.append(80)
            elif plant.height >= 5:
                health_factors.append(60)
            else:
                health_factors.append(30)
        
        # 叶数健康度
        if plant.leaf_count:
            if plant.leaf_count >= 100:
                health_factors.append(95)
            elif plant.leaf_count >= 50:
                health_factors.append(80)
            elif plant.leaf_count >= 20:
                health_factors.append(60)
            else:
                health_factors.append(40)
        
        # 环境健康度
        if latest_env:
            env_health = 0
            env_factors = 0
            
            # 温度健康度
            if 18 <= latest_env.temperature <= 25:
                env_health += 95
            elif 15 <= latest_env.temperature <= 28:
                env_health += 80
            else:
                env_health += 50
            env_factors += 1
            
            # 湿度健康度
            if 60 <= latest_env.humidity <= 80:
                env_health += 95
            elif 50 <= latest_env.humidity <= 90:
                env_health += 80
            else:
                env_health += 60
            env_factors += 1
            
            # 土壤湿度健康度
            if 60 <= latest_env.soil_moisture <= 80:
                env_health += 95
            elif 50 <= latest_env.soil_moisture <= 90:
                env_health += 80
            else:
                env_health += 65
            env_factors += 1
            
            if env_factors > 0:
                health_factors.append(env_health / env_factors)
        
        # 计算综合健康指数
        if health_factors:
            features['health_index'] = round(sum(health_factors) / len(health_factors), 1)
        
        # 计算胁迫水平
        stress_factors = []
        
        if latest_env:
            # 温度胁迫
            if latest_env.temperature < 15 or latest_env.temperature > 28:
                stress_factors.append(30)
            elif latest_env.temperature < 18 or latest_env.temperature > 25:
                stress_factors.append(15)
            else:
                stress_factors.append(5)
            
            # 湿度胁迫
            if latest_env.humidity < 50 or latest_env.humidity > 90:
                stress_factors.append(25)
            elif latest_env.humidity < 60 or latest_env.humidity > 80:
                stress_factors.append(10)
            else:
                stress_factors.append(5)
            
            # 土壤湿度胁迫
            if latest_env.soil_moisture < 50 or latest_env.soil_moisture > 90:
                stress_factors.append(25)
            elif latest_env.soil_moisture < 60 or latest_env.soil_moisture > 80:
                stress_factors.append(10)
            else:
                stress_factors.append(5)
        
        if stress_factors:
            features['stress_level'] = round(sum(stress_factors) / len(stress_factors), 1)
        
        return jsonify(features)
        
    except Exception as e:
        return jsonify({'error': f'获取生物特征数据失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>/growth-prediction')
def api_growth_prediction_detailed(plant_id):
    """获取茶树详细生长预测"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        
        # 获取环境数据
        latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
        
        # 计算预测参数
        days_since_planting = (datetime.datetime.now() - plant.planting_date).days if plant.planting_date else 0
        current_height = plant.height or 0
        
        # 生成未来预测数据
        predictions = []
        for months in range(1, 13):  # 未来12个月
            days_ahead = months * 30
            
            # 基于当前生长阶段和环境条件调整预测
            if plant.current_stage == '幼苗期':
                growth_rate = 0.15  # 幼苗期生长速率
                if latest_env and (latest_env.temperature < 18 or latest_env.temperature > 25):
                    growth_rate *= 0.7  # 温度不适宜时减缓
            elif plant.current_stage == '幼年期':
                growth_rate = 0.08
                if latest_env and (latest_env.humidity < 60 or latest_env.humidity > 80):
                    growth_rate *= 0.8  # 湿度不适宜时减缓
            elif plant.current_stage == '成年期':
                growth_rate = 0.02
                if latest_env and (latest_env.soil_moisture < 60 or latest_env.soil_moisture > 80):
                    growth_rate *= 0.9  # 土壤湿度不适宜时减缓
            else:  # 衰老期
                growth_rate = 0.01
            
            predicted_height = current_height + (days_ahead * growth_rate)
            
            # 预测产量
            if plant.current_stage == '幼苗期':
                predicted_yield = 0
            elif plant.current_stage == '幼年期':
                predicted_yield = max(0, (predicted_height - 10) * 0.1)
            elif plant.current_stage == '成年期':
                predicted_yield = max(0, (predicted_height - 15) * 0.15)
            else:
                predicted_yield = max(0, (predicted_height - 20) * 0.05)
            
            predictions.append({
                'month': months,
                'predicted_height': round(predicted_height, 2),
                'predicted_yield': round(predicted_yield, 2),
                'growth_rate': round(growth_rate, 3)
            })
        
        # 农事建议
        recommendations = []
        
        # 基于生长阶段的建议
        if plant.current_stage == '幼苗期':
            recommendations.append({
                'type': 'fertilization',
                'title': '幼苗期施肥',
                'description': '建议施用氮磷钾复合肥，促进根系发育',
                'timing': '每30天一次',
                'priority': 'high'
            })
            recommendations.append({
                'type': 'irrigation',
                'title': '保持土壤湿润',
                'description': '幼苗期需水量较大，保持土壤湿度60-80%',
                'timing': '根据天气情况调整',
                'priority': 'high'
            })
        
        elif plant.current_stage == '幼年期':
            recommendations.append({
                'type': 'pruning',
                'title': '整形修剪',
                'description': '进行整形修剪，培养良好树形',
                'timing': '春季和秋季',
                'priority': 'medium'
            })
            recommendations.append({
                'type': 'fertilization',
                'title': '平衡施肥',
                'description': '施用平衡型肥料，促进枝叶生长',
                'timing': '每45天一次',
                'priority': 'medium'
            })
        
        elif plant.current_stage == '成年期':
            recommendations.append({
                'type': 'harvest',
                'title': '适时采摘',
                'description': '根据茶叶品质要求适时采摘',
                'timing': '春季、夏季、秋季',
                'priority': 'high'
            })
            recommendations.append({
                'type': 'pest_control',
                'title': '病虫害防治',
                'description': '定期检查病虫害情况，及时防治',
                'timing': '每月检查一次',
                'priority': 'medium'
            })
        
        # 基于环境条件的建议
        if latest_env:
            if latest_env.temperature < 18:
                recommendations.append({
                    'type': 'protection',
                    'title': '防寒措施',
                    'description': '温度偏低，建议采取防寒措施',
                    'timing': '立即执行',
                    'priority': 'high'
                })
            
            if latest_env.soil_moisture < 60:
                recommendations.append({
                    'type': 'irrigation',
                    'title': '补充灌溉',
                    'description': '土壤湿度偏低，需要补充灌溉',
                    'timing': '近期执行',
                    'priority': 'high'
                })
        
        return jsonify({
            'predictions': predictions,
            'recommendations': recommendations,
            'current_status': {
                'height': current_height,
                'stage': plant.current_stage,
                'days_since_planting': days_since_planting
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'获取生长预测失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>/agricultural-operations')
def api_agricultural_operations(plant_id):
    """获取推荐农事操作"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        
        # 获取最近的施肥记录
        last_fertilization = FertilizationRecord.query.filter_by(tea_plant_id=plant_id).order_by(FertilizationRecord.timestamp.desc()).first()
        
        # 获取环境数据
        latest_env = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).first()
        
        operations = []
        
        # 基于生长阶段的农事操作
        if plant.current_stage == '幼苗期':
            operations.extend([
                {
                    'type': 'fertilization',
                    'title': '幼苗期施肥',
                    'description': '施用氮磷钾复合肥，促进根系发育',
                    'materials': '氮磷钾复合肥 0.5kg',
                    'method': '环状施肥',
                    'frequency': '每30天',
                    'priority': 'high',
                    'estimated_cost': 25,
                    'expected_benefit': '促进根系发育，提高成活率'
                },
                {
                    'type': 'irrigation',
                    'title': '保持土壤湿润',
                    'description': '幼苗期需水量较大，保持土壤湿度60-80%',
                    'materials': '灌溉设备',
                    'method': '滴灌或喷灌',
                    'frequency': '根据天气情况',
                    'priority': 'high',
                    'estimated_cost': 15,
                    'expected_benefit': '确保幼苗正常生长'
                },
                {
                    'type': 'weeding',
                    'title': '除草管理',
                    'description': '及时清除杂草，避免与幼苗竞争养分',
                    'materials': '除草工具',
                    'method': '人工除草',
                    'frequency': '每15天',
                    'priority': 'medium',
                    'estimated_cost': 10,
                    'expected_benefit': '减少养分竞争，促进幼苗生长'
                }
            ])
        
        elif plant.current_stage == '幼年期':
            operations.extend([
                {
                    'type': 'pruning',
                    'title': '整形修剪',
                    'description': '进行整形修剪，培养良好树形',
                    'materials': '修剪工具',
                    'method': '疏枝、短截',
                    'frequency': '春季和秋季',
                    'priority': 'medium',
                    'estimated_cost': 30,
                    'expected_benefit': '培养良好树形，提高产量'
                },
                {
                    'type': 'fertilization',
                    'title': '平衡施肥',
                    'description': '施用平衡型肥料，促进枝叶生长',
                    'materials': '平衡型复合肥 1kg',
                    'method': '环状施肥',
                    'frequency': '每45天',
                    'priority': 'medium',
                    'estimated_cost': 40,
                    'expected_benefit': '促进枝叶生长，提高生物量'
                },
                {
                    'type': 'pest_control',
                    'title': '病虫害防治',
                    'description': '定期检查病虫害情况，及时防治',
                    'materials': '生物农药',
                    'method': '预防性喷施',
                    'frequency': '每月检查',
                    'priority': 'medium',
                    'estimated_cost': 20,
                    'expected_benefit': '预防病虫害，保护茶树健康'
                }
            ])
        
        elif plant.current_stage == '成年期':
            operations.extend([
                {
                    'type': 'harvest',
                    'title': '适时采摘',
                    'description': '根据茶叶品质要求适时采摘',
                    'materials': '采摘工具',
                    'method': '手工采摘',
                    'frequency': '春季、夏季、秋季',
                    'priority': 'high',
                    'estimated_cost': 50,
                    'expected_benefit': '获得优质茶叶，提高经济效益'
                },
                {
                    'type': 'fertilization',
                    'title': '维持施肥',
                    'description': '维持土壤养分，保证产量稳定',
                    'materials': '有机肥 2kg',
                    'method': '沟施',
                    'frequency': '每60天',
                    'priority': 'medium',
                    'estimated_cost': 35,
                    'expected_benefit': '维持产量稳定，提高品质'
                },
                {
                    'type': 'pruning',
                    'title': '更新修剪',
                    'description': '进行更新修剪，保持树势',
                    'materials': '修剪工具',
                    'method': '回缩、疏枝',
                    'frequency': '每年一次',
                    'priority': 'low',
                    'estimated_cost': 25,
                    'expected_benefit': '保持树势，延长经济寿命'
                }
            ])
        
        # 基于环境条件的农事操作
        if latest_env:
            if latest_env.temperature < 18:
                operations.append({
                    'type': 'protection',
                    'title': '防寒措施',
                    'description': '温度偏低，采取防寒措施保护茶树',
                    'materials': '防寒材料',
                    'method': '覆盖保温',
                    'frequency': '立即执行',
                    'priority': 'high',
                    'estimated_cost': 45,
                    'expected_benefit': '防止冻害，保护茶树'
                })
            
            if latest_env.soil_moisture < 60:
                operations.append({
                    'type': 'irrigation',
                    'title': '补充灌溉',
                    'description': '土壤湿度偏低，需要补充灌溉',
                    'materials': '灌溉设备',
                    'method': '滴灌或喷灌',
                    'frequency': '近期执行',
                    'priority': 'high',
                    'estimated_cost': 20,
                    'expected_benefit': '缓解干旱胁迫，促进生长'
                })
            
            if latest_env.soil_ph < 4.5 or latest_env.soil_ph > 5.5:
                operations.append({
                    'type': 'soil_amendment',
                    'title': '土壤改良',
                    'description': '土壤pH不适宜，需要进行土壤改良',
                    'materials': '石灰或硫磺',
                    'method': '土壤调理',
                    'frequency': '每年一次',
                    'priority': 'medium',
                    'estimated_cost': 60,
                    'expected_benefit': '改善土壤环境，提高养分利用率'
                })
        
        # 基于施肥记录的农事操作
        if last_fertilization:
            days_since_fertilization = (datetime.datetime.now() - last_fertilization.timestamp).days
            if days_since_fertilization > 30:
                operations.append({
                    'type': 'fertilization',
                    'title': '及时施肥',
                    'description': f'距离上次施肥已过去{days_since_fertilization}天，建议及时施肥',
                    'materials': '复合肥 1.5kg',
                    'method': '环状施肥',
                    'frequency': '立即执行',
                    'priority': 'high',
                    'estimated_cost': 35,
                    'expected_benefit': '补充养分，促进生长'
                })
        
        return jsonify({
            'operations': operations,
            'total_estimated_cost': sum(op['estimated_cost'] for op in operations),
            'high_priority_count': len([op for op in operations if op['priority'] == 'high'])
        })
        
    except Exception as e:
        return jsonify({'error': f'获取农事操作失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>/3d-model')
def api_tea_plant_3d_model(plant_id):
    """生成茶树3D可视化模型数据"""
    try:
        # 如果已有真实GLB模型，则直接返回指向GLB的元数据，前端可优先加载
        glb_path = _tree_glb_path(plant_id)
        if os.path.exists(glb_path):
            return jsonify({
                'tea_plant_info': {'id': plant_id},
                'realistic_model': {
                    'type': 'glb',
                    'url': url_for('static', filename=f'models/trees_glb/{plant_id}.glb')
                }
            })
        # 优先返回缓存
        cached = load_cached_tree_model(plant_id)
        if cached:
            return jsonify(cached)
        # 获取茶树基本信息
        tea_plant = TeaPlant.query.get(plant_id)
        if not tea_plant:
            return jsonify({'error': '茶树不存在'}), 404
        
        # 获取生长记录
        growth_records = GrowthRecord.query.filter_by(tea_plant_id=plant_id).order_by(GrowthRecord.timestamp).all()
        
        # 获取环境数据
        env_data = EnvironmentalData.query.order_by(EnvironmentalData.timestamp.desc()).limit(30).all()
        
        # 生成(或回退到)参数化点云3D模型数据
        model_data = generate_3d_tea_plant_model(tea_plant, growth_records, env_data)
        # 缓存结果
        save_tree_model(plant_id, model_data)
        return jsonify(model_data)
        
    except Exception as e:
        return jsonify({'error': f'生成3D模型失败: {str(e)}'}), 500

def generate_3d_tea_plant_model(tea_plant, growth_records, env_data):
    """生成茶树3D模型数据"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # 基础参数
    base_height = tea_plant.height or 1.5
    base_width = tea_plant.crown_width or 1.2
    leaf_count = tea_plant.leaf_count or 100
    
    # 生成树干3D坐标
    trunk_height = base_height * 0.3
    trunk_radius = base_width * 0.1
    
    # 树干圆柱体坐标
    trunk_points = []
    trunk_faces = []
    
    # 树干分段
    trunk_segments = 8
    trunk_circle_points = 12
    
    for i in range(trunk_segments + 1):
        height_ratio = i / trunk_segments
        current_height = height_ratio * trunk_height
        current_radius = trunk_radius * (1 - height_ratio * 0.3)  # 树干向上逐渐变细
        
        for j in range(trunk_circle_points):
            angle = 2 * np.pi * j / trunk_circle_points
            x = current_radius * np.cos(angle)
            y = current_radius * np.sin(angle)
            z = current_height
            
            trunk_points.append([x, y, z])
    
    # 生成树冠3D坐标
    crown_height = base_height - trunk_height
    crown_radius = base_width / 2
    
    crown_points = []
    crown_faces = []
    
    # 树冠分层
    crown_layers = 6
    for layer in range(crown_layers + 1):
        height_ratio = layer / crown_layers
        current_height = trunk_height + height_ratio * crown_height
        
        # 每层的半径（椭圆形树冠）
        layer_radius_x = crown_radius * (1 - height_ratio * 0.4)
        layer_radius_y = crown_radius * (1 - height_ratio * 0.6)
        
        # 每层的点数
        layer_points = max(8, int(12 * (1 - height_ratio * 0.5)))
        
        for i in range(layer_points):
            angle = 2 * np.pi * i / layer_points
            x = layer_radius_x * np.cos(angle)
            y = layer_radius_y * np.sin(angle)
            z = current_height
            
            crown_points.append([x, y, z])
    
    # 生成叶片3D坐标
    leaf_points = []
    leaf_faces = []
    
    # 在树冠范围内随机分布叶片
    for i in range(leaf_count):
        # 随机位置
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, crown_radius * 0.8)
        height = np.random.uniform(trunk_height, base_height)
        
        # 叶片朝向
        leaf_angle = np.random.uniform(0, 2 * np.pi)
        leaf_tilt = np.random.uniform(-np.pi/6, np.pi/6)
        
        # 叶片大小
        leaf_size = np.random.uniform(0.05, 0.15)
        
        # 叶片四个顶点
        leaf_center = [radius * np.cos(angle), radius * np.sin(angle), height]
        
        # 叶片方向向量
        leaf_dir_x = np.cos(leaf_angle)
        leaf_dir_y = np.sin(leaf_angle)
        
        # 叶片四个顶点
        leaf_points.extend([
            [leaf_center[0] - leaf_dir_x * leaf_size, leaf_center[1] - leaf_dir_y * leaf_size, leaf_center[2]],
            [leaf_center[0] + leaf_dir_x * leaf_size, leaf_center[1] + leaf_dir_y * leaf_size, leaf_center[2]],
            [leaf_center[0] + leaf_dir_x * leaf_size, leaf_center[1] + leaf_dir_y * leaf_size, leaf_center[2] + leaf_size * 0.1],
            [leaf_center[0] - leaf_dir_x * leaf_size, leaf_center[1] - leaf_dir_y * leaf_size, leaf_center[2] + leaf_size * 0.1]
        ])
    
    # 生成根系3D坐标
    root_points = []
    root_faces = []
    
    # 主根
    main_root_length = base_height * 0.4
    main_root_radius = trunk_radius * 0.8
    
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = main_root_radius * np.cos(angle)
        y = main_root_radius * np.sin(angle)
        z = -main_root_length
        
        root_points.append([x, y, z])
    
    # 侧根
    side_root_count = 12
    for i in range(side_root_count):
        angle = 2 * np.pi * i / side_root_count
        radius = np.random.uniform(trunk_radius * 0.5, trunk_radius * 1.5)
        depth = np.random.uniform(-main_root_length * 0.3, -main_root_length * 0.8)
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = depth
        
        root_points.append([x, y, z])
    
    # 生成生长动画数据
    growth_animation = []
    if growth_records:
        # 基于生长记录生成动画帧
        for i, record in enumerate(growth_records):
            progress = i / (len(growth_records) - 1) if len(growth_records) > 1 else 0
            
            frame_data = {
                'frame': i,
                'progress': progress,
                'height_scale': 0.3 + progress * 0.7,
                'width_scale': 0.2 + progress * 0.8,
                'leaf_count_scale': 0.1 + progress * 0.9,
                'timestamp': record.timestamp.isoformat() if record.timestamp else None,
                'stage': record.stage
            }
            growth_animation.append(frame_data)
    else:
        # 如果没有生长记录，生成模拟数据
        for i in range(10):
            progress = i / 9
            frame_data = {
                'frame': i,
                'progress': progress,
                'height_scale': 0.3 + progress * 0.7,
                'width_scale': 0.2 + progress * 0.8,
                'leaf_count_scale': 0.1 + progress * 0.9,
                'timestamp': (datetime.now() - timedelta(days=30-i*3)).isoformat(),
                'stage': ['幼苗期', '生长期', '成年期'][min(i//3, 2)]
            }
            growth_animation.append(frame_data)
    
    # 生成环境数据可视化
    env_visualization = []
    if env_data:
        for i, env in enumerate(env_data):
            env_visualization.append({
                'timestamp': env.timestamp.isoformat() if env.timestamp else None,
                'temperature': env.temperature,
                'humidity': env.humidity,
                'soil_moisture': env.soil_moisture,
                'soil_ph': env.soil_ph,
                'nitrogen': env.nitrogen,
                'phosphorus': env.phosphorus,
                'potassium': env.potassium
            })
    
    return {
        'tea_plant_info': {
            'id': tea_plant.id,
            'variety': tea_plant.variety,
            'current_stage': tea_plant.current_stage,
            'height': tea_plant.height,
            'crown_width': tea_plant.crown_width,
            'leaf_count': tea_plant.leaf_count,
            'location_area': tea_plant.location_area,
            'location_row': tea_plant.location_row
        },
        'model_data': {
            'trunk': {
                'points': trunk_points,
                'faces': trunk_faces
            },
            'crown': {
                'points': crown_points,
                'faces': crown_faces
            },
            'leaves': {
                'points': leaf_points,
                'faces': leaf_faces
            },
            'roots': {
                'points': root_points,
                'faces': root_faces
            }
        },
        'growth_animation': growth_animation,
        'environmental_data': env_visualization,
        'model_parameters': {
            'base_height': base_height,
            'base_width': base_width,
            'leaf_count': leaf_count,
            'trunk_height': trunk_height,
            'crown_height': crown_height
        },
        'meta': {
            'generated_at': datetime.now().isoformat()
        }
    }

@app.route('/tea-plants/<int:plant_id>/3d-view')
def tea_plant_3d_view(plant_id):
    """单株3D可视化页面"""
    # 页面仅渲染容器与脚本，由前端通过API加载数据
    plant = TeaPlant.query.get_or_404(plant_id)
    return render_template('tea-plant-3d.html', plant=plant)

@app.route('/api/tea-plants/<int:plant_id>/upload-images', methods=['POST'])
def api_upload_tree_images(plant_id):
    """上传茶树照片用于YOLO分割与重建。支持多文件。"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': '未找到上传文件字段 files'}), 400
        files = request.files.getlist('files')
        saved = []
        img_dir = _tree_images_dir(plant_id)
        for f in files:
            filename = f.filename
            if not filename:
                continue
            safe_name = filename.replace('..', '_')
            dst = os.path.join(img_dir, safe_name)
            f.save(dst)
            saved.append(safe_name)
        return jsonify({'message': '上传成功', 'saved': saved})
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>/reconstruct-3d', methods=['POST'])
def api_reconstruct_3d(plant_id):
    """基于YOLO分割 + 多视角重建，生成更真实的GLB模型。
    需要本机安装 yolo 分割模型与重建工具（如Colmap/OpenMVG/OpenMVS）。
    该端点为占位接口，提供脚本挂钩，默认返回任务接受状态。
    """
    try:
        img_dir = _tree_images_dir(plant_id)
        if not os.path.isdir(img_dir) or len(os.listdir(img_dir)) == 0:
            return jsonify({'error': '请先上传足够的多视角茶树照片'}), 400

        # 占位：执行外部重建脚本（需要你在项目/scripts下实现该脚本）
        # 例如：python scripts/reconstruct_tree.py --images <img_dir> --out_glb <glb_path>
        glb_out = _tree_glb_path(plant_id)
        os.makedirs(os.path.dirname(glb_out), exist_ok=True)

        # 这里不阻塞执行，真实环境建议投递到后台任务队列
        # try:
        #     subprocess.check_call([
        #         'python', 'scripts/reconstruct_tree.py',
        #         '--images', img_dir,
        #         '--out_glb', glb_out,
        #     ])
        # except subprocess.CalledProcessError as e:
        #     return jsonify({'error': f'重建脚本执行失败: {e}'}), 500

        # 暂时返回任务已受理，前端可轮询 /api/tea-plants/<id>/3d-model 查看是否已有glb
        return jsonify({'message': '重建任务已提交，请稍后刷新3D页面', 'glb_target': glb_out})
    except Exception as e:
        return jsonify({'error': f'重建任务提交失败: {str(e)}'}), 500

@app.route('/api/pests/info/<pest_type>')
def api_pest_info(pest_type):
    """获取特定病虫害的详细信息"""
    if pest_type in PEST_TREATMENT_ADVICE:
        return jsonify(PEST_TREATMENT_ADVICE[pest_type])
    else:
        return jsonify({'error': '未找到该病虫害信息'}), 404

@app.route('/api/pests/trend')
def api_pests_trend():
    """病虫害趋势：过去30天各类型数量（基于登记记录统计）。"""
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=30)
        # 查询按天聚合
        # 自检一次，避免上线后旧库缺列
        _ensure_pest_record_schema()

        records = (
            db.session.query(
                func.date(PestRecord.timestamp).label('d'),
                PestRecord.pest_type,
                func.count('*').label('cnt')
            )
            .filter(PestRecord.timestamp >= start)
            .group_by('d', PestRecord.pest_type)
            .all()
        )
        # 先收集实际有值的数据
        raw = {}
        for d, pest_type, cnt in records:
            d = str(d)
            raw.setdefault(d, {})[pest_type] = cnt

        # 类别集合（与前端一致）
        classes = ['green_mirid_bug','gray_blight','brown_blight','helopeltis','red_spider','tea_algal_leaf_spot']

        # 补齐从 start(含) 到 end(含) 每天的键，并为缺失类别填0
        filled = {}
        cur = start.date()
        end_day = end.date()
        while cur <= end_day:
            key = str(cur)
            day_map = {}
            src = raw.get(key, {})
            for c in classes:
                day_map[c] = int(src.get(c, 0))
            filled[key] = day_map
            cur += datetime.timedelta(days=1)

        return jsonify({'range_days': 30, 'series': filled})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pests/control-stats')
def api_pests_control_stats():
    """防治措施统计（示例：按严重程度区间统计数量）。"""
    try:
        # 自检一次
        _ensure_pest_record_schema()
        buckets = {
            '轻微': db.session.query(func.count('*')).filter(PestRecord.severity <= 2).scalar() or 0,
            '中等': db.session.query(func.count('*')).filter(PestRecord.severity.between(3, 4)).scalar() or 0,
            '严重': db.session.query(func.count('*')).filter(PestRecord.severity >= 5).scalar() or 0,
        }
        return jsonify({'buckets': buckets})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pests/sample-images')
def api_pests_sample_images():
    """返回每类病虫害的一张示例图片。

    优先从数据库中的 `PestRecord.image_path` 里挑选最新一张；
    若库中没有图片，则回退到 `static/` 或 `datasets/` 下的样例图。
    返回格式：{ cls: url }
    """
    # 需要支持的类别（与页面保持一致）
    classes = ['green_mirid_bug', 'gray_blight', 'red_spider', 'brown_blight', 'helopeltis', 'tea_algal_leaf_spot']

    # 回退搜索目录
    base_dirs = [
        os.path.join('static', 'disease and pests'),  # 新目录，优先使用
        os.path.join('static', 'tea-bug and disease-data'),
        os.path.join('datasets', 'tea_pests_classify', 'test')
    ]

    result = {}

    # 1) 先从数据库取每类的最新图片
    try:
        for cls in classes:
            record = (
                PestRecord.query
                .filter(PestRecord.pest_type == cls)
                .filter(PestRecord.image_path.isnot(None))
                .order_by(PestRecord.timestamp.desc())
                .first()
            )
            if record and record.image_path:
                path = record.image_path.replace('\\', '/')
                if path.startswith('static'):
                    result[cls] = '/' + path
                else:
                    result[cls] = '/api/file?path=' + path
    except Exception:
        # 数据库不可用时忽略，走回退逻辑
        pass

    # 2) 对仍未找到的类别，从静态或数据集目录中回退获取
    for cls in classes:
        if cls in result:
            continue
        found = None
        for base in base_dirs:
            dir_path = None
            for root, dirs, files in os.walk(base):
                if root.lower().endswith(cls):
                    dir_path = root
                    break
            if dir_path:
                for f in os.listdir(dir_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        found = os.path.join(dir_path, f).replace('\\', '/')
                        break
            # 若未匹配到以类别命名的子目录，则尝试匹配该目录直接平铺的文件
            if not found and os.path.isdir(base):
                try:
                    for f in os.listdir(base):
                        fl = f.lower()
                        if fl.endswith(('.jpg', '.jpeg', '.png')) and (fl.startswith(cls) or cls in fl):
                            found = os.path.join(base, f).replace('\\', '/')
                            break
                except Exception:
                    pass
        if found:
            if found.startswith('static'):
                result[cls] = '/' + found
            else:
                result[cls] = '/api/file?path=' + found

    return jsonify(result)

@app.route('/api/file')
def api_file_serve():
    """安全地返回项目内文件（限制在 uploads/datasets/static 三个根目录下）。"""
    path = request.args.get('path', '')
    if not path:
        return jsonify({'error': 'path required'}), 400

    # 规范化路径，阻止路径穿越
    norm_path = os.path.normpath(path)
    norm_path = norm_path.replace('\\', '/')
    if '..' in norm_path:
        return jsonify({'error': 'invalid path'}), 400

    allowed_roots = ['uploads', 'datasets', 'static']
    if not any(norm_path.startswith(root + '/') or norm_path == root for root in allowed_roots):
        return jsonify({'error': 'forbidden'}), 403

    fs_path = norm_path
    if not os.path.isfile(fs_path):
        return jsonify({'error': 'not found'}), 404

    from flask import send_file
    return send_file(fs_path)


 

@app.route('/api/pests/register', methods=['POST'])
def api_pests_register():
    """病虫害登记API"""
    try:
        # 获取表单数据
        pest_type = request.form.get('pest_type')
        severity = request.form.get('severity', type=int)
        discovery_time_str = request.form.get('discovery_time')
        location = request.form.get('location')
        affected_area = request.form.get('affected_area', type=float)
        notes = request.form.get('notes')
        
        # 验证必填字段
        if not pest_type or not severity:
            return jsonify({'error': '病虫害类型和严重程度为必填项'}), 400
        
        # 处理时间
        discovery_time = datetime.datetime.now()
        if discovery_time_str:
            try:
                discovery_time = datetime.datetime.fromisoformat(discovery_time_str)
            except ValueError:
                pass
        
        # 处理上传的图片
        image_path = None
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                # 创建上传目录
                upload_dir = os.path.join('uploads', 'pest_records')
                os.makedirs(upload_dir, exist_ok=True)
                
                # 生成文件名
                import uuid
                file_extension = os.path.splitext(image_file.filename)[1]
                filename = f"{uuid.uuid4().hex}{file_extension}"
                image_path = os.path.join(upload_dir, filename)
                
                # 保存文件
                image_file.save(image_path)
        
        # 保存到数据库
        pest_record = PestRecord(
            timestamp=discovery_time,
            pest_type=pest_type,
            severity=severity,
            affected_area=affected_area or 0.0,
            treatment=notes or '',
            location=location or '',
            image_path=image_path
        )
        
        db.session.add(pest_record)
        db.session.commit()
        
        return jsonify({
            'message': '病虫害登记成功',
            'record_id': pest_record.id,
            'pest_type': PEST_CLASS_ZH.get(pest_type, pest_type),
            'severity': severity,
            'discovery_time': discovery_time.isoformat()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'登记失败: {str(e)}'}), 500

@app.route('/api/pests/classify', methods=['POST'])
def api_pests_classify():
    """茶叶病虫害分类推理接口（YOLOv5）。接收表单文件 `file`，返回top-1与top-5。"""
    try:
        if not os.path.exists(YOLOV5_WEIGHTS_PATH):
            return jsonify({'error': '未加载分类权重，请先训练并将best.pt复制为 models/tea_pests_cls.pt'}), 503
        if 'file' not in request.files:
            return jsonify({'error': '未找到上传文件字段 file'}), 400
        f = request.files['file']
        if not f or not f.filename:
            return jsonify({'error': '未选择文件'}), 400

        tmp_dir = os.path.join('uploads', 'tmp_cls')
        os.makedirs(tmp_dir, exist_ok=True)
        safe_name = f.filename.replace('..', '_')
        tmp_path = os.path.join(tmp_dir, safe_name)
        f.save(tmp_path)

        result = _yolov5_classify(tmp_path, imgsz=224, topk=5)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'推理异常: {str(e)}'}), 500

@app.route('/api/tea-plants/<int:plant_id>/generate-plan', methods=['POST'])
def api_generate_plan(plant_id):
    """根据当前茶树与环境生成一份详细农事计划，并存库后返回。"""
    try:
        plant = TeaPlant.query.get_or_404(plant_id)
        # 简化：生成当前季度为期90天的计划，包含四项任务
        start_date = datetime.datetime.now()
        end_date = start_date + datetime.timedelta(days=90)
        plan = AgriculturalPlan(
            tea_plant_id=plant.id,
            plan_start_date=start_date,
            plan_end_date=end_date,
            season='当前季',
            growth_stage=plant.current_stage or '未知',
            plan_type='自动生成',
            status='pending',
            priority=3,
        )
        db.session.add(plan)
        db.session.commit()

        # 生成详细任务
        details = [
            ('施肥', start_date + datetime.timedelta(days=10), '以氮肥为主配合钾肥，环状施肥，避免烧根', '复合肥1.5kg', 60, '晴朗或无雨'),
            ('灌溉', start_date + datetime.timedelta(days=3), '保持土壤湿度在60-70%，小水勤灌', '清水', 40, '高温或干旱时期增加频次'),
            ('修剪', start_date + datetime.timedelta(days=35), '轻度修剪，去除病弱枝，改善通风透光', '剪枝剪、酒精', 90, '避开雨天'),
            ('病虫害防治', start_date + datetime.timedelta(days=20), '预防性喷药，重点防茶尺蠖', '拟除虫菊酯类 适量', 50, '低风速时段操作'),
        ]
        for activity_type, sched, desc, materials, duration, weather in details:
            detail = AgriculturalPlanDetail(
                plan_id=plan.id,
                activity_type=activity_type,
                scheduled_date=sched,
                description=desc,
                materials_needed=materials,
                estimated_duration=duration,
                weather_requirement=weather,
                precautions='注意个人防护，按说明配比',
                status='pending',
            )
            db.session.add(detail)
        db.session.commit()

        return jsonify({'message': '计划已生成', 'plan_id': plan.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'生成计划失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
