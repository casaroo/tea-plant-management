# 最优生长模型功能说明

## 功能概述

最优生长模型是茶叶生长模型系统的核心功能之一，它基于茶树累计的生长环境数据及作物模型，对各个节点的农事活动进行修正，不再简单以时间进行农事活动的依据。

## 主要功能

### 1. 最优生长参数分析
- **生长速率**: 基于当前环境条件预测的最优生长速率
- **产量潜力**: 在当前条件下能达到的最大产量潜力
- **品质评分**: 茶叶品质的综合评分
- **环境匹配度**: 当前环境与最优环境的匹配程度

### 2. 环境优化建议
系统会根据当前环境数据与最优生长条件进行对比，提供针对性的优化建议：

- **温度优化**: 当温度偏离最适范围时，提供保温或降温建议
- **湿度优化**: 当湿度不足时，建议增加灌溉频率
- **土壤湿度优化**: 当土壤湿度不足时，建议立即灌溉
- **土壤养分优化**: 根据氮、磷、钾含量提供施肥建议

### 3. 农事活动优化
基于当前环境和茶树生长阶段，提供农事活动的优化建议：

- **施肥优化**: 根据土壤养分状况调整施肥时机和用量
- **灌溉优化**: 根据土壤湿度状况调整灌溉计划
- **修剪优化**: 根据茶树生长阶段和高度提供修剪建议
- **病虫害防治优化**: 根据环境条件调整防治策略
- **采摘优化**: 根据新梢发育情况确定最佳采摘时机

### 4. 生长阶段专项建议
针对不同生长阶段提供专门的管理建议：

- **幼苗期**: 保温保湿，避免强光直射，检查根系发育
- **幼年期**: 充足养分供应，修剪整形，培养良好树冠
- **成年期**: 平衡产量和品质，病虫害防治，保持树势
- **衰老期**: 重修剪或更新换代

## API接口

### 1. 最优生长模型API
```
GET /api/optimal-growth/<plant_id>
```

**响应示例:**
```json
{
  "plant_info": {
    "id": 1,
    "variety": "龙井",
    "current_stage": "幼苗期",
    "height": 15.2,
    "width": 20.5
  },
  "current_environment": {
    "temperature": 24.9,
    "humidity": 72.1,
    "soil_moisture": 46.6,
    "soil_ph": 5.58,
    "nitrogen": 28.0,
    "phosphorus": 17.4,
    "potassium": 17.1
  },
  "optimal_growth": {
    "growth_rate": 0.284,
    "yield_potential": 1.246,
    "quality_score": 0.556,
    "condition_gap": 0.056
  },
  "optimization_suggestions": [
    {
      "type": "soil_moisture",
      "issue": "土壤湿度不足",
      "suggestion": "立即灌溉，保持土壤湿润",
      "priority": 1
    }
  ],
  "stage_suggestions": [
    "幼苗期需要特别注意保温保湿，避免强光直射",
    "定期检查根系发育情况，适时补充微量元素"
  ]
}
```

### 2. 农事活动优化API
```
GET /api/agricultural-optimization/<plant_id>
```

**响应示例:**
```json
{
  "plant_id": 1,
  "optimizations": [
    {
      "activity_type": "灌溉",
      "current_timing": "按计划时间",
      "optimal_timing": "立即执行",
      "reason": "土壤湿度不足，影响根系发育",
      "expected_benefit": "改善水分利用效率",
      "priority": 1
    }
  ],
  "total_count": 1
}
```

## 数据模型

### 1. OptimalGrowthModel (最优生长模型)
```python
class OptimalGrowthModel(db.Model):
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
```

### 2. AgriculturalOptimization (农事活动优化)
```python
class AgriculturalOptimization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tea_plant_id = db.Column(db.Integer, db.ForeignKey('tea_plant.id'))
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    activity_type = db.Column(db.String(100))  # 活动类型
    current_timing = db.Column(db.String(100))  # 当前时机
    optimal_timing = db.Column(db.String(100))  # 最优时机
    reason = db.Column(db.Text)  # 优化原因
    expected_benefit = db.Column(db.String(200))  # 预期收益
    priority = db.Column(db.Integer)  # 优先级 1-5
    status = db.Column(db.String(20), default='pending')  # 状态
```

## 算法说明

### 1. 最优生长模型训练
系统使用随机森林算法训练最优生长模型：

- **输入特征**: 温度、湿度、土壤湿度、pH、氮、磷、钾、生长阶段
- **输出目标**: 生长速率、产量潜力、品质评分、最适环境参数
- **训练数据**: 基于茶树生长规律生成的模拟数据

### 2. 优化建议生成
系统根据以下规则生成优化建议：

- **优先级1 (高)**: 影响茶树生存的关键问题
- **优先级2 (中)**: 影响生长和产量的重要问题  
- **优先级3 (低)**: 影响品质的优化问题

### 3. 农事活动优化
基于以下因素进行农事活动优化：

- **环境条件**: 温度、湿度、土壤条件
- **生长阶段**: 幼苗期、幼年期、成年期、衰老期
- **历史数据**: 历年农事活动效果
- **预测模型**: 未来环境变化预测

## 使用说明

### 1. 访问最优生长模型页面
1. 启动应用: `python app.py`
2. 访问: `http://localhost:5000/growth-model`
3. 查看最优生长模型分析结果

### 2. 查看优化建议
- 在页面中查看环境优化建议
- 查看农事活动优化建议
- 查看生长阶段专项建议

### 3. API调用
```python
import requests

# 获取最优生长模型数据
response = requests.get('http://localhost:5000/api/optimal-growth/1')
data = response.json()

# 获取农事活动优化建议
response = requests.get('http://localhost:5000/api/agricultural-optimization/1')
data = response.json()
```

## 技术特点

1. **数据驱动**: 基于真实环境数据和历史农事活动数据
2. **智能分析**: 使用机器学习算法进行预测和优化
3. **实时更新**: 根据最新环境数据实时调整建议
4. **可视化展示**: 直观的图表和界面展示分析结果
5. **优先级管理**: 根据影响程度对建议进行优先级排序

## 未来扩展

1. **深度学习模型**: 引入更复杂的深度学习模型提高预测精度
2. **多品种支持**: 支持不同茶树品种的最优生长模型
3. **地理信息集成**: 结合地理信息提供更精准的建议
4. **移动端支持**: 开发移动端应用方便田间使用
5. **专家系统**: 集成茶叶种植专家知识库
