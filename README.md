# 航空货运动态定价系统

## 系统简介
航空货运动态定价系统是一个基于多种算法模型的价格优化系统，可以帮助航空货运公司确定最优的货物价格策略。系统包含三种核心定价模型：

1. **需求弹性模型** - 基于价格-需求关系的简单定价模型
2. **博弈论模型** - 考虑竞争对手行为的策略性定价模型
3. **动态规划模型** - 考虑多种约束和长期影响的复杂优化模型

## 文件结构
```
essential_files/
├── api.py                     # 后端API服务器
├── index.html                 # 前端界面主文件
├── README.md                  # 系统文档
├── requirements.txt           # Python依赖库列表
├── aircraft_config.py         # 机型配置模块
├── route_config.py            # 航线配置模块
├── air_freight_pricing_model.py  # 需求弹性定价模型实现
├── gametheory.py              # 博弈论定价模型实现
├── Dynamic_Programming.py     # 动态规划定价模型实现
└── static/                    # 静态资源目录
    ├── api-client.js          # API客户端，处理前端与后端通信
    └── styles.css             # 界面样式表
```

## 系统要求
- Python 3.8+
- Flask 2.0.1
- Flask-CORS 3.0.10
- NumPy 1.21.0
- SciPy 1.7.0
- Pandas 1.3.0
- Matplotlib 3.4.2
- Seaborn 0.11.1
- Joblib 1.4.2

## 启动方法
1. 安装依赖库：
   ```
   pip install -r requirements.txt
   ```

2. 启动后端API服务器：
   ```
   python api.py
   ```

3. 在浏览器中打开index.html文件。也可以使用任何静态文件服务器托管前端文件。

系统默认API服务器地址为 http://localhost:8000，前端将自动连接此地址。

## 注意事项
- 确保所有文件的相对路径保持不变，特别是static目录中的资源文件
- 如需修改API服务器地址，请编辑static/api-client.js文件中的API_BASE_URL变量
