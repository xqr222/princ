import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 检查导入错误并详细报告
try:
    from flask import Flask, request, jsonify, make_response, send_from_directory, render_template
    print("成功导入 Flask")
except ImportError as e:
    print(f"错误: 无法导入 Flask - {e}")
    print(f"Python路径: {sys.path}")
    sys.exit(1)

try:
    from flask_cors import CORS, cross_origin
    print("成功导入 Flask-CORS")
except ImportError as e:
    print(f"错误: 无法导入 Flask-CORS - {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"成功导入 NumPy {np.__version__}")
except ImportError as e:
    print(f"错误: 无法导入 NumPy - {e}")
    sys.exit(1)

try:
    import scipy.optimize as optimize
    print("成功导入 SciPy")
except ImportError as e:
    print(f"错误: 无法导入 SciPy - {e}")
    sys.exit(1)

import json
import time
import traceback
import sqlite3 # 添加 sqlite3 导入

# 尝试导入模型相关模块
try:
    from aircraft_config import AircraftConfig
    print("成功导入 AircraftConfig")
except ImportError as e:
    print(f"错误: 无法导入 AircraftConfig - {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from route_config import RouteConfig
    print("成功导入 RouteConfig")
except ImportError as e:
    print(f"错误: 无法导入 RouteConfig - {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from air_freight_pricing_model import ElasticityBasedPricingModel
    print("成功导入 ElasticityBasedPricingModel")
except ImportError as e:
    print(f"错误: 无法导入 ElasticityBasedPricingModel - {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from gametheory import AirCargoCompetitiveModel, NashEquilibriumSolver
    print("成功导入 AirCargoCompetitiveModel")
except ImportError as e:
    print(f"错误: 无法导入 AirCargoCompetitiveModel - {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from Dynamic_Programming import EnhancedAirCargoDP
    print("成功导入 EnhancedAirCargoDP")
except ImportError as e:
    print(f"错误: 无法导入 EnhancedAirCargoDP - {e}")
    traceback.print_exc()
    sys.exit(1)

# CabinOptimizerDP已移除，不再需要导入

print("所有导入成功，准备创建 Flask 应用")

app = Flask(__name__, static_folder='static', template_folder='.')
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": "*"}}, supports_credentials=True)  # 完全开放CORS
print("Flask 应用创建成功，已启用完全CORS支持")

# 初始化全局配置存储
route_configs = {}
aircraft_configs = {} # 确保 aircraft_configs 也被初始化

# 添加默认航线配置：大连-广州
def init_default_configs():
    # 创建默认A320机型配置
    if "A320" not in aircraft_configs:
        aircraft_configs["A320"] = AircraftConfig("A320", config={
            'max_payload': 6964,
            'max_volume': 500000,
            'base_operating_cost': 3500
        })

    # 创建大连-广州航线配置
    dalian_guangzhou_id = "大连-广州"
    if dalian_guangzhou_id not in route_configs:
        dalian_guangzhou_config = RouteConfig({
            'origin': '大连',
            'destination': '广州',
            'distance': 2500,
            'flight_type': '干线',
            'competition_level': 'high',
            'popularity': 'high',
            'flight_frequency': 14,
            'season_factor': 1.0
        })
        # 设置A320为默认机型
        dalian_guangzhou_config.set_aircraft(aircraft_configs["A320"])
        route_configs[dalian_guangzhou_id] = dalian_guangzhou_config
        print(f"已添加默认航线配置: {dalian_guangzhou_id}")

# 日志配置
import logging
import os
from datetime import datetime

# 创建日志目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 定义日志文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"api_{timestamp}.log")
results_file = os.path.join(log_dir, f"results_{timestamp}.log")

# 设置基本日志
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('api')

# 添加文件处理器，记录到日志文件
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 添加控制台处理器以确保所有日志都显示在终端
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 创建一个专门用于记录结果的日志器
results_logger = logging.getLogger('results')
results_logger.setLevel(logging.INFO)
results_handler = logging.FileHandler(results_file)
results_formatter = logging.Formatter('%(message)s')
results_handler.setFormatter(results_formatter)
results_logger.addHandler(results_handler)

# 确保结果日志器不传播到根日志器
results_logger.propagate = False

logger.info(f"API服务启动，日志文件位置: {log_file}, 结果文件位置: {results_file}")
print(f"API服务启动，日志文件位置: {log_file}, 结果文件位置: {results_file}")

# 初始化默认配置
init_default_configs()

# 为所有响应添加字符集设置
@app.after_request
def add_charset_and_cors(response):
    """确保所有响应都包含正确的字符集设置和CORS头部"""
    # 为JSON响应设置正确的字符集
    if response.mimetype == 'application/json':
        response.mimetype = 'application/json; charset=utf-8'
    
    # 添加CORS头部
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    
    return response

@app.route('/')
def index():
    logger.info("尝试加载index页面")
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"渲染index.html时出错: {e}")
        return f"加载页面出错: {str(e)}", 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    logger.debug("Favicon.ico requested")
    return jsonify({"status": "ok", "message": "API服务正常运行"})

@app.route('/favicon.ico')
def favicon():
    logger.debug("Favicon.ico requested")
    return '', 204 # No Content

@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def chrome_devtools_json():
    logger.debug("Chrome DevTools JSON requested")
    return '', 204 # No Content

@app.route('/api/freight_requests', methods=['GET'])
def get_freight_requests():
    """从数据库获取所有货运请求"""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pricing_data.db')
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row # 允许按列名访问数据
        cursor = conn.cursor()
        cursor.execute("SELECT id, request_timestamp, route_id, cargo_type, weight_kg, volume_cm3 FROM freight_requests ORDER BY request_timestamp DESC")
        requests_data = [dict(row) for row in cursor.fetchall()]
        return jsonify(requests_data)
    except sqlite3.Error as e:
        logger.error(f"数据库错误 (get_freight_requests): {e}")
        return jsonify({"error": "数据库错误", "details": str(e)}), 500
    except Exception as e:
        logger.error(f"获取货运请求时发生意外错误: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "获取货运请求时发生意外错误", "details": str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/routes', methods=['GET'])
def get_routes():
    # Return available route identifiers and basic info
    return jsonify([rc.get_route_info() for rc in route_configs.values()])

@app.route('/api/aircrafts', methods=['GET'])
def get_aircrafts():
    # Return available aircraft types and basic info
    app.logger.debug(f"AIRCRAFT_CONFIGS content: {aircraft_configs}")
    # 关键修改:
    # 1. 将循环变量 type, config 重命名为 ac_type, ac_config 以避免覆盖内置函数
    # 2. 对 ac_config 对象使用 .to_dict() 方法将其转换为字典
    options = [{"type": ac_type, **ac_config.to_dict()} for ac_type, ac_config in aircraft_configs.items()]
    app.logger.debug(f"Final aircraft options to be returned: {options}")
    return jsonify(options)
    
@app.route('/api/config/route/<route_id>', methods=['GET'])
def get_route_config(route_id):
    if route_id in route_configs:
        return jsonify(route_configs[route_id].get_route_info())
    else:
        return jsonify({"error": "Route configuration not found"}), 404

@app.route('/api/config/aircraft/<aircraft_type>', methods=['GET'])
def get_aircraft_config(aircraft_type):
    if aircraft_type in aircraft_configs:
        return jsonify(aircraft_configs[aircraft_type].to_dict())
    else:
        # Try creating a default if known type
        if aircraft_type == "A320":
            if "A320" not in aircraft_configs:
                aircraft_configs["A320"] = AircraftConfig("A320", config={
                    'max_payload': 6964,
                    'max_volume': 500000,
                    'base_operating_cost': 3500
                })
            return jsonify(aircraft_configs["A320"].to_dict())
        return jsonify({"error": "Aircraft configuration not found"}), 404

# Ensure database functions are available if api.py needs to fetch RouteConfig details by ID
# For example, if the ElasticityModel needed more from RouteConfig than just params loaded via route_id
try:
    from database import get_route_by_id as get_route_by_id_from_db # Alias to avoid conflict if other get_route_by_id exists
    DB_ROUTES_AVAILABLE = True
except ImportError:
    DB_ROUTES_AVAILABLE = False
    def get_route_by_id_from_db(route_id):
        return None # Dummy

@app.route('/api/elasticity/calculate', methods=['POST'])
def calculate_elasticity_pricing():
    """
    计算基于需求弹性的定价结果。
    优先从数据库加载与 route_id 关联的参数，允许通过请求体覆盖。
    
    Request Example (with route_id):
    {
        "route_id": 1, // <<< NEW/PREFERRED: ID of the route in the database
        "price_elasticity": { "快件": -1.8 }, // Optional: overrides DB/defaults
        "initial_prices": { "鲜活": 8.0 },   // Optional: overrides DB/defaults
        "base_demands": { "普货": 70 },     // Optional: overrides DB/defaults
        "booking_period": 14,
        "num_cargo_types": 3
        // Old parameters like origin, destination, aircraft_type etc. are still accepted 
        // for creating a RouteConfig if needed, but model params primarily come via route_id.
    }
    """
    request_data = request.get_json()
    logging.info(f"收到弹性定价计算请求: {request_data}")
    if not request_data:
        return jsonify({"status": "error", "message": "未提供请求数据"}), 400

    try:
        route_id_from_request = request_data.get('route_id')
        final_route_id = None
        if route_id_from_request is not None:
            try:
                final_route_id = int(route_id_from_request)
            except ValueError:
                return jsonify({"error": "无效的 route_id 格式，必须是整数"}), 400

        # ... rest of the function ...

    except Exception as e:
        logger.error(f"计算弹性定价时发生意外错误: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "计算弹性定价时发生意外错误", "details": str(e)}), 500 