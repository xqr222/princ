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
        
        # 英文到中文键名映射 (修正了'鲜活'的空格)
        key_map_en_to_cn = {"express": "快件", "fresh": "鲜活", "regular": "普货"}

        def translate_keys_robust(data_dict, en_to_cn_map):
            if not isinstance(data_dict, dict):
                return data_dict
            translated_dict = {}
            if data_dict: # 确保 data_dict 不是 None
                for key, value in data_dict.items():
                    if key in en_to_cn_map: # 如果是已知的英文键，则翻译
                        translated_dict[en_to_cn_map[key]] = value
                    else: # 否则，保持原样 (可能是已翻译的中文键或其他键)
                        translated_dict[key] = value
            return translated_dict
        
        # --- 结束：RouteConfig 创建逻辑 ---

        # 从请求中提取 elasticity_params (如果存在)
        elasticity_params_from_request = request_data.get('elasticity_params', {}) # 如果没有则为空字典

        # 获取原始覆盖参数
        # 1. initial_prices
        initial_prices_override_raw = elasticity_params_from_request.get('initial_prices')
        if initial_prices_override_raw is None:
            initial_prices_override_raw = request_data.get('initial_prices')

        # 2. price_elasticity (处理复数和单数形式，以及顶层和嵌套)
        price_elasticity_override_raw = elasticity_params_from_request.get('price_elasticities') # 优先子对象复数
        if price_elasticity_override_raw is None:
            price_elasticity_override_raw = elasticity_params_from_request.get('price_elasticity') # 子对象单数
        if price_elasticity_override_raw is None:
            price_elasticity_override_raw = request_data.get('price_elasticity') # 顶层单数
        if price_elasticity_override_raw is None:
            price_elasticity_override_raw = request_data.get('price_elasticities') # 顶层复数

        # 3. base_demands
        base_demands_override_raw = elasticity_params_from_request.get('base_demands')
        if base_demands_override_raw is None:
            base_demands_override_raw = request_data.get('base_demands')
        
        # 翻译键名 (这部分应已存在，确保它在上述提取逻辑之后)
        initial_prices_override = translate_keys_robust(initial_prices_override_raw, key_map_en_to_cn)
        price_elasticity_override = translate_keys_robust(price_elasticity_override_raw, key_map_en_to_cn)
        base_demands_override = translate_keys_robust(base_demands_override_raw, key_map_en_to_cn)
        
        # 日志打印 (确保这些在翻译之后)
        logging.info(f"翻译后的 initial_prices_override: {initial_prices_override}")
        logging.info(f"翻译后的 price_elasticity_override: {price_elasticity_override}")
        logging.info(f"翻译后的 base_demands_override: {base_demands_override}")
        
        # 获取其他直接参数
        booking_period = int(request_data.get('booking_period', 14))
        num_cargo_types = int(request_data.get('num_cargo_types', 3))
        num_airlines = int(request_data.get('num_airlines', 1))
        num_segments = int(request_data.get('num_segments', 1))

        route_config_for_model = None # 初始化为 None
        # 优先根据请求中的 route_id 从数据库加载配置 (如果提供了 route_id 且 DB_ROUTES_AVAILABLE)
        if final_route_id and DB_ROUTES_AVAILABLE:
            route_db_data = get_route_by_id_from_db(final_route_id)
            if route_db_data:
                # route_db_data 应该包含创建 RouteConfig 所需的航线信息
                # 可能还需要从中提取或关联 aircraft_type 来加载 AircraftConfig
                route_config_for_model = RouteConfig(route_info=route_db_data)
                aircraft_type_from_db = route_db_data.get('aircraft_type', 'A320') # 假设数据库返回机型
                aircraft_config_instance = aircraft_configs.get(aircraft_type_from_db)
                if aircraft_config_instance:
                    route_config_for_model.set_aircraft(aircraft_config_instance)
                else:
                    logger.warning(f"未找到机型 {aircraft_type_from_db} 的配置，RouteConfig 将不包含机型信息或使用默认。")
            else:
                logger.warning(f"提供了 route_id {final_route_id} 但未从数据库找到对应航线。")

        # 如果没有通过 route_id 加载，或者需要根据请求中的详细参数创建/覆盖
        if route_config_for_model is None and 'origin' in request_data and 'destination' in request_data:
            origin = request_data.get('origin')
            destination = request_data.get('destination')
            # ... (获取其他航线参数如 distance, flight_type 等) ...
            route_info_dict = {
                'origin': origin, 'destination': destination, 
                # ... (其他参数) ...
            }
            if final_route_id: # 如果之前有 final_route_id 但没加载成功，也尝试用它
                 route_info_dict['id'] = final_route_id
            
            route_config_for_model = RouteConfig(route_info=route_info_dict)
            
            aircraft_type_req = request_data.get('aircraft_type')
            if aircraft_type_req:
                aircraft_config_instance = aircraft_configs.get(aircraft_type_req)
                if aircraft_config_instance:
                    route_config_for_model.set_aircraft(aircraft_config_instance)
                else:
                    # 可以选择创建一个临时的，或记录警告
                    logger.warning(f"请求中指定的机型 {aircraft_type_req} 未在全局配置中找到。")
                    # temp_aircraft_config = AircraftConfig(aircraft_type_req, config=request_data.get('aircraft_config', {}))
                    # route_config_for_model.set_aircraft(temp_aircraft_config)
        
        # 如果到这里 route_config_for_model 仍然是 None，可能需要一个默认处理或错误返回
        if route_config_for_model is None:
            logger.error("无法确定或创建 route_config_for_model。弹性模型可能无法正确初始化。")
            # 可以选择返回错误，或者尝试使用一个非常通用的默认配置
            # return jsonify({"status": "error", "message": "无法确定航线配置"}), 400
            # 或者，如果模型允许 route_config=None 并有内部处理，则可以继续，但目前看模型需要它。
            # 暂时创建一个空的或者基于默认值的，以避免直接的NameError，但这治标不治本
            logger.warning("Fallback: Creating a default RouteConfig for elasticity model as it was None.")
            default_route_info = {'origin': 'Unknown', 'destination': 'Unknown', 'distance': 0}
            route_config_for_model = RouteConfig(route_info=default_route_info)
            default_aircraft = aircraft_configs.get("A320") # 尝试获取默认A320
            if default_aircraft:
                route_config_for_model.set_aircraft(default_aircraft)

        # ... 实例化 ElasticityBasedPricingModel ...
        logger.info(f"Instantiating ElasticityBasedPricingModel with route_id: {final_route_id}")
        model = ElasticityBasedPricingModel(
            route_id=final_route_id,
            route_config=route_config_for_model,
            booking_period=booking_period,
            initial_prices=initial_prices_override,
            price_elasticity=price_elasticity_override,
            base_demands=base_demands_override,
            num_cargo_types=num_cargo_types,
            num_airlines=num_airlines,
            num_segments=num_segments
        )
        
        # The model's optimize_prices now returns: 
        # optimal_prices, max_revenue, initial_total_revenue, demand_details_list, daily_prices_list
        optimal_prices_array, optimal_total_revenue, initial_total_revenue, demand_details_list, daily_prices_list = model.optimize_prices()

        # Prepare price_table for API response
        price_table_for_api = []
        if demand_details_list: # demand_details_list contains segment info, typically we only have 1 segment (n=0) for elasticity model on UI
            # We need to iterate through unique cargo types present in demand_details_list or based on num_cargo_types
            # Assuming demand_details_list has items for segment 0 if n=1
            processed_cargo_types_for_table = set()
            for detail in demand_details_list:
                cargo_type_name_raw = detail.get("cargo_type")
                if not cargo_type_name_raw: # 最好加上这个检查
                    continue
                cargo_type_name = cargo_type_name_raw.strip() # <--- 添加 .strip()
                if cargo_type_name in processed_cargo_types_for_table:
                    continue # Already added to table for this cargo type
                
                initial_price = detail.get("initial_price_unit", 0)
                optimal_price = detail.get("optimal_price_unit", 0)
                change_percentage = ((optimal_price - initial_price) / initial_price) * 100 if initial_price else 0
                
                price_table_for_api.append({
                    "cargoType": cargo_type_name, # This will be Chinese name, e.g., "快件"
                    "initialPrice": initial_price,
                    "optimalPrice": optimal_price,
                    "change": round(change_percentage, 1)
                })
                processed_cargo_types_for_table.add(cargo_type_name)
        
        revenue_increase_percentage = 0
        if initial_total_revenue > 0:
            revenue_increase_percentage = ((optimal_total_revenue - initial_total_revenue) / initial_total_revenue) * 100
        else: # Avoid division by zero; if initial revenue is 0, any positive optimal revenue is infinite increase, or 0 if optimal is also 0
            revenue_increase_percentage = float('inf') if optimal_total_revenue > 0 else 0

        response_data = {
                "status": "success",
            "message": "需求弹性模型计算完成 (参数已尝试从数据库加载)",
            "route_id_used": final_route_id,
            "price_table": price_table_for_api, # NEW: This is what front-end expects
            "initial_revenue": round(initial_total_revenue, 2),
            "optimal_revenue": round(optimal_total_revenue, 2),
            "revenue_increase": round(revenue_increase_percentage, 1),
            "optimal_prices_per_unit_array": optimal_prices_array.tolist() if isinstance(optimal_prices_array, np.ndarray) else optimal_prices_array,
            # "demand_details": demand_details_list, # Still useful for detailed debugging if needed
            "daily_prices_list": daily_prices_list # Renamed from daily_prices_table_json, now it's a list of dicts
        }
        logger.info(f"弹性模型响应数据: {response_data}")
        logging.info(f"最终返回给前端的 daily_prices_list: {daily_prices_list}") 

        # 在返回之前，清理 daily_prices_list 中每个项目的 cargoType
        if isinstance(response_data.get("daily_prices_list"), list):
            for item in response_data["daily_prices_list"]:
                if isinstance(item, dict) and "cargoType" in item and isinstance(item["cargoType"], str):
                    item["cargoType"] = item["cargoType"].strip() # <--- 清理
        
        logger.info(f"弹性模型响应数据 (已清理 cargoType): {response_data}") # 更新日志消息
        # logging.info(f"最终返回给前端的 daily_prices_list (已清理 cargoType): {response_data.get('daily_prices_list')}") # 这行日志可以与上一行合并或按需保留
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"弹性模型计算错误: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": "模型计算失败", "details": str(e)}), 500

@app.route('/api/gametheory/calculate', methods=['POST'])
def calculate_gametheory_pricing():
    """
    计算基于博弈论的动态定价结果
    接收航线信息、公司信息（运力、初始价格）和博弈参数
    模拟多个销售阶段，找到纳什均衡价格和各公司收益
    新增：支持通过 route_id 从数据库加载部分博弈论参数
    """
    try:
        data = request.get_json()
        logger.info(f"收到博弈论定价计算请求(精确模拟): {json.dumps(data, ensure_ascii=False, indent=2)}")

        # --- T_periods Handling ---
        T_periods_arg = data.get('time_periods') 
        final_T_periods_for_solver = None 
        if T_periods_arg is not None:
            try:
                parsed_T = int(T_periods_arg)
                if parsed_T > 0:
                    final_T_periods_for_solver = parsed_T
                    logger.info(f"博弈论API: 从请求中获取 T_periods = {final_T_periods_for_solver}")
                else:
                     logger.warning(f"博弈论API: 请求的 time_periods ({T_periods_arg}) 无效，将依赖后端默认值。")
            except (ValueError, TypeError):
                 logger.warning(f"博弈论API: 请求的 time_periods ('{T_periods_arg}') 不是有效整数/数字，将依赖后端默认值。")
        else:
             logger.info("博弈论API: 请求中未提供 time_periods，模型将自行决定 (DB > 默认)。")
        
        # --- k_value Handling (from API request, model will prioritize DB if route_id is used) ---
        k_value_from_api = data.get('k_value') # Could be None
        if k_value_from_api is not None:
            try:
                k_value_from_api = float(k_value_from_api)
                logger.info(f"博弈论API: 从请求中获取 k_value = {k_value_from_api}")
            except (ValueError, TypeError):
                logger.warning(f"博弈论API: 请求的 k_value ('{data.get('k_value')}') 不是有效数字，将设为 None 让模型决定。")
                k_value_from_api = None
        else:
            logger.info("博弈论API: 请求中未提供 k_value，模型将自行决定 (DB > 默认)。")


        # --- route_id Handling (for DB parameter loading by the model) ---
        route_id_str_gt = data.get('route_id') 
        route_id_for_gt_model = None
        if route_id_str_gt is not None:
            try:
                route_id_for_gt_model = int(route_id_str_gt)
                logger.info(f"博弈论API: 从请求中获取到 route_id: {route_id_for_gt_model}，将传递给博弈论模型。")
            except ValueError:
                logger.warning(f"博弈论API: 请求中提供的 route_id ('{route_id_str_gt}') 不是有效的整数，将设为 None。")
                route_id_for_gt_model = None


        # --- RouteConfig Handling (primarily for non-DB sourced route info if route_id is not used/found) ---
        route_details_from_req = {
            'origin': data.get('origin', '大连'),
            'destination': data.get('destination', '广州'),
            'distance': data.get('distance', 2500),
            'flight_type': data.get('flight_type', '干线'),
            'competition_level': data.get('competition_level', 'high'),
            'popularity': data.get('popularity', 'high'),
            'season_factor': data.get('season_factor', 1.0),
        }
        aircraft_type_from_req = data.get('aircraft_type', 'A320')
        
        current_aircraft_config = aircraft_configs.get(aircraft_type_from_req)
        if not current_aircraft_config:
            logger.warning(f"博弈论API: 未在全局配置中找到机型 {aircraft_type_from_req}，尝试使用默认A320。")
            current_aircraft_config = aircraft_configs.get("A320")
            if not current_aircraft_config: # Should not happen if init_default_configs worked
                 logger.error(f"博弈论API: 连默认A320机型配置都未找到。将创建一个临时的。")
                 current_aircraft_config = AircraftConfig(aircraft_type_from_req)


        flight_route_config = RouteConfig(route_info=route_details_from_req)
        flight_route_config.set_aircraft(current_aircraft_config)
        logger.info(f"博弈论API: 创建的 flight_route_config 基于: {route_details_from_req} 和机型 {aircraft_type_from_req}")


        # --- Company Data Handling ---
        companies_data = data.get('companies', [])
        if len(companies_data) != 2:
            return jsonify({"error": "需要两家公司的数据"}), 400

        comp1_data = companies_data[0]
        comp2_data = companies_data[1]

        company1_init_args = {
             'company_id': comp1_data.get('id', comp1_data.get('name')),
             'W': float(comp1_data.get('initial_capacity')), 
             'route_config': flight_route_config,
             'k_value': k_value_from_api,
             'initial_price': float(comp1_data.get('initialPrice')),
             'route_id': route_id_for_gt_model,
             'T_periods': final_T_periods_for_solver
        }
        company2_init_args = {
             'company_id': comp2_data.get('id', comp2_data.get('name')),
             'W': float(comp2_data.get('initial_capacity')),
             'route_config': flight_route_config,
             'k_value': k_value_from_api,
             'initial_price': float(comp2_data.get('initialPrice')),
             'route_id': route_id_for_gt_model,
             'T_periods': final_T_periods_for_solver
        }
        
        logger.info(f"博弈论API: Company 1 Init Args: {company1_init_args}")
        logger.info(f"博弈论API: Company 2 Init Args: {company2_init_args}")

        company1 = AirCargoCompetitiveModel(**company1_init_args)
        company2 = AirCargoCompetitiveModel(**company2_init_args)

        # --- Solver Initialization ---
        solver_init_args = {
             'company1': company1,
             'company2': company2,
        }
        if final_T_periods_for_solver is not None:
             solver_init_args['T_periods'] = final_T_periods_for_solver
             logger.info(f"博弈论API: NashEquilibriumSolver 将使用来自API的 T_periods: {final_T_periods_for_solver}")
        else:
            logger.info(f"博弈论API: NashEquilibriumSolver 将使用 company1.T_periods: {company1.T_periods}")


        solver_opt = NashEquilibriumSolver(**solver_init_args)
        actual_T_periods = solver_opt.T_periods 
        logger.info(f"博弈论API: 开始博弈均衡和最优收入模拟 (共 {actual_T_periods} 阶段)...")

        # --- Nash Equilibrium Calculation ---
            p1_nash_prices_all_periods, p2_nash_prices_all_periods = solver_opt.find_nash_equilibrium()
        logger.info(f"博弈论API: Nash均衡价格计算完成: P1长度={len(p1_nash_prices_all_periods) if p1_nash_prices_all_periods is not None else 'None'}, P2长度={len(p2_nash_prices_all_periods) if p2_nash_prices_all_periods is not None else 'None'}")

        if (p1_nash_prices_all_periods is None or (hasattr(p1_nash_prices_all_periods, '__len__') and len(p1_nash_prices_all_periods) == 0)) or \
           (p2_nash_prices_all_periods is None or (hasattr(p2_nash_prices_all_periods, '__len__') and len(p2_nash_prices_all_periods) == 0)):
                raise ValueError("Nash均衡价格计算返回空列表或None")
        
            if len(p1_nash_prices_all_periods) != actual_T_periods or len(p2_nash_prices_all_periods) != actual_T_periods:
             raise ValueError(f"Nash均衡价格列表长度 ({len(p1_nash_prices_all_periods)}) 与实际模拟周期 ({actual_T_periods})不匹配。")

        # --- Simulation with Nash Prices ---
        results = {
            "company1": {"name": company1.company_id, "prices": [], "demands": [], "sales": [], "revenues": [], "total_revenue": 0, "capacity_left": float(company1.initial_W)},
            "company2": {"name": company2.company_id, "prices": [], "demands": [], "sales": [], "revenues": [], "total_revenue": 0, "capacity_left": float(company2.initial_W)}
        }
        period_details_log = []
        
        company1.W = company1.initial_W
        company1.price_history, company1.sales_history, company1.revenue_history, company1.demand_history = [], [], [], []
        
        company2.W = company2.initial_W
        company2.price_history, company2.sales_history, company2.revenue_history, company2.demand_history = [], [], [], []

        for t_idx in range(actual_T_periods):
            period_num = t_idx + 1 
                eq_price1 = p1_nash_prices_all_periods[t_idx]
                eq_price2 = p2_nash_prices_all_periods[t_idx]
            
            demand1 = company1.calculate_demand(own_price=eq_price1, competitor_price=eq_price2, t=period_num)
            demand2 = company2.calculate_demand(own_price=eq_price2, competitor_price=eq_price1, t=period_num)
            
            sale1 = company1.calculate_sales(demand=demand1, remaining_capacity=company1.W)
            sale2 = company2.calculate_sales(demand=demand2, remaining_capacity=company2.W)

            revenue1 = sale1 * eq_price1
            revenue2 = sale2 * eq_price2

            company1.update_inventory(sale1)
            company2.update_inventory(sale2)
            
            results["company1"]["prices"].append(round(eq_price1, 2))
            results["company1"]["demands"].append(round(demand1, 2))
            results["company1"]["sales"].append(round(sale1, 2))
            results["company1"]["revenues"].append(round(revenue1, 2))
            
            results["company2"]["prices"].append(round(eq_price2, 2))
            results["company2"]["demands"].append(round(demand2, 2))
            results["company2"]["sales"].append(round(sale2, 2))
            results["company2"]["revenues"].append(round(revenue2, 2))

            period_details_log.append(f"阶段 {period_num}: C1_P={eq_price1:.2f}, C2_P={eq_price2:.2f}, C1_S={sale1:.1f}, C2_S={sale2:.1f}")

        results["company1"]["total_revenue"] = round(sum(results["company1"]["revenues"]), 2)
        results["company1"]["capacity_left"] = round(company1.W, 2)
        results["company2"]["total_revenue"] = round(sum(results["company2"]["revenues"]), 2)
        results["company2"]["capacity_left"] = round(company2.W, 2)
        
        total_simulation_revenue = results["company1"]["total_revenue"] + results["company2"]["total_revenue"]
        results["total_simulation_revenue"] = total_simulation_revenue
        
        logger.info(f"博弈论API: 模拟完成 ({actual_T_periods} 阶段)。C1收益:{results['company1']['total_revenue']:.2f}, C2收益:{results['company2']['total_revenue']:.2f}")
        
        final_response = {
            "status": "success",
            "message": "博弈论定价计算完成",
            "config": {
                "route_id_passed_to_model": route_id_for_gt_model,
                "time_periods_requested_in_api": T_periods_arg, 
                "time_periods_simulated_by_solver": actual_T_periods,
                "k_value_from_api_request": k_value_from_api,
                "k_value_used_by_c1_model": company1.k_value, 
                "company1_initial_capacity": float(comp1_data.get('initial_capacity')),
                "company2_initial_capacity": float(comp2_data.get('initial_capacity')),
                "company1_initial_price": float(comp1_data.get('initialPrice')),
                "company2_initial_price": float(comp2_data.get('initialPrice')),
                "flight_route_config_used_origin": flight_route_config.route_info.get('origin')
            },
            "results": results,
            "log": period_details_log
        }
        return jsonify(final_response)

    except TypeError as te:
        logger.error(f"博弈论API: /api/gametheory/calculate 类型错误: {te} - 数据: {data}")
        logger.error(traceback.format_exc())
        # Ensure the data dict is JSON serializable for the error response
        serializable_data_str = "Data not serializable"
        try:
            serializable_data_str = json.dumps(data)
        except:
            pass
        return jsonify({"error": f"服务器内部类型错误: {str(te)}", "request_data_preview": serializable_data_str[:500]}), 500 # Include partial data
    except ValueError as ve:
        logger.error(f"博弈论API: /api/gametheory/calculate 值错误: {ve} - 数据: {data}")
        logger.error(traceback.format_exc())
        serializable_data_str = "Data not serializable"
        try:
            serializable_data_str = json.dumps(data)
        except:
            pass
        return jsonify({"error": f"服务器内部值错误: {str(ve)}", "request_data_preview": serializable_data_str[:500]}), 500
    except KeyError as ke:
        logger.error(f"博弈论API: /api/gametheory/calculate 键错误: {ke} - 数据: {data}")
        logger.error(traceback.format_exc())
        serializable_data_str = "Data not serializable"
        try:
            serializable_data_str = json.dumps(data)
        except:
            pass
        return jsonify({"error": f"请求中缺少必要键: {str(ke)}", "request_data_preview": serializable_data_str[:500]}), 400
    except Exception as e:
        logger.error(f"博弈论API: /api/gametheory/calculate 发生意外错误: {e}")
        logger.error(traceback.format_exc()) 
        return jsonify({"error": "计算博弈论模型时发生服务器内部错误", "details": str(e)}), 500

@app.route('/api/dynamicdp/calculate', methods=['POST', 'OPTIONS'])
@cross_origin(supports_credentials=True)
def calculate_dynamicdp_pricing():
    """计算基于动态规划的定价API端点"""
    try:
        data = request.json
        logging.info(f"收到动态规划定价计算请求: {data}")
        
        # --- 新增：获取 route_id ---
        route_id_str = data.get('route_id')
        route_id_for_model = None
        if route_id_str is not None:
            try:
                route_id_for_model = int(route_id_str)
                logging.info(f"从请求中获取到 route_id: {route_id_for_model}，将传递给DP模型。")
            except ValueError:
                logging.warning(f"请求中提供的 route_id ('{route_id_str}') 不是有效的整数，将忽略。DP模型将依赖直接参数或硬编码默认值。")
                # Consider returning 400 if route_id format is strictly required and invalid
                # return jsonify({"error": "无效的 route_id 格式，必须是整数。"}), 400
        # --- 结束新增 ---

        # 获取其他参数 (这些将作为直接覆盖值传递给模型)
        # DP模型的核心参数会通过下面的覆盖或DB加载 (via route_id)
        capacity_weight_override = data.get('max_payload') 
        capacity_volume_override = data.get('max_volume')  
        booking_period_override = data.get('booking_period')
        gamma_override = data.get('gamma')
        c_d_ratio_override = data.get('C_D_ratio')
        
        # 其他非DB加载的参数
        booking_start_date = data.get('booking_start_date')
        booking_end_date = data.get('booking_end_date')
        penalty_factor_override = data.get('penalty_factor') # Example of another overrideable param
        cv_override = data.get('cv') # Example

        # 确保 booking_period (如果作为覆盖值提供) 是整数
        if booking_period_override is not None:
            try:
                booking_period_override = int(booking_period_override)
        except (ValueError, TypeError):
                logging.warning(f"请求中提供的 booking_period ('{data.get('booking_period')}') 无效，将设为None让模型决定。")
                booking_period_override = None
        
        # 动态规划模型实例化参数准备
        model_init_params = {
            'route_id': route_id_for_model,
            'capacity_weight': capacity_weight_override,
            'capacity_volume': capacity_volume_override,
            'time_periods': booking_period_override,
            'gamma': gamma_override,
            'C_D_ratio': c_d_ratio_override,
            'booking_start_date': booking_start_date,
            'booking_end_date': booking_end_date,
            'penalty_factor': penalty_factor_override,
            'cv': cv_override
        }
        # 移除值为 None 的参数，以便模型内部的默认/DB加载逻辑能正确触发
        active_model_init_params = {k: v for k, v in model_init_params.items() if v is not None}
        logging.info(f"准备传递给 EnhancedAirCargoDP 的最终参数: {active_model_init_params}")
        
        # 创建动态规划模型实例并执行计算
        calculation_succeeded = False 
        results_from_model = None 
        booking_days_price_table = [] 

        model = EnhancedAirCargoDP(**active_model_init_params)
        
        # 获取模型实际使用的参数用于日志和返回
        final_capacity_weight = model.capacity_weight
        final_capacity_volume = model.capacity_volume
        final_booking_period = model.time_periods 
        final_gamma = model.gamma
        final_cd_ratio = model.C_D_ratio
        
        logging.info(f"DP模型内部实际使用的参数: route_id={route_id_for_model}, CapW={final_capacity_weight}, CapV={final_capacity_volume}, T={final_booking_period}, Gamma={final_gamma}, CD_Ratio={final_cd_ratio}")

        # ---- 原有的 route_info 构建逻辑 (主要用于返回给前端，不直接驱动模型参数) ----
        origin = data.get('origin', 'PVG')
        destination = data.get('destination', 'LAX')
        distance = data.get('distance', 10000)
        competition_level = data.get('competition_level', 'medium')
        popularity = data.get('popularity', 'medium')
        season_factor = data.get('season_factor', 1.0)
        flight_type = data.get('flight_type', '干线')
        market_share = data.get('market_share', 0.5)
        flight_frequency = data.get('flight_frequency', 7)
        aircraft_type_req = data.get('aircraft_type', 'A320') # Aircraft type from request for info display

        route_info_for_response = {
            'origin': origin, 'destination': destination, 'distance': distance,
            'competition_level': competition_level, 'popularity': popularity,
            'season_factor': season_factor, 'flight_type': flight_type,
            'market_share': market_share, 'flight_frequency': flight_frequency
        }
        # ---- 结束 route_info 构建 ----

        try:
            logging.info(f"创建动态规划模型实例成功 (route_id: {route_id_for_model}, active_params: {active_model_init_params})...")
            logging.info("开始执行动态规划模型核心求解 (solve_wvs)...")
            model.solve_wvs(parallel=False, verbose=True) 
            logging.info("动态规划模型核心求解 (solve_wvs) 完成。")

            if hasattr(model, 'optimal_prices') and model.optimal_prices:
                logging.info("optimal_prices 已由 solve_wvs 填充，继续进行收益模拟...")
                results_from_model = model.fast_simulate_revenue(num_simulations=100, verbose=False)
                logging.info(f"收益模拟结果: {results_from_model}")

                if results_from_model and 'daily_prices' in results_from_model:
                    daily_prices_from_model = results_from_model['daily_prices']
                    logging.info(f"获取每日价格数据成功，类型: {type(daily_prices_from_model)}")
                    calculation_succeeded = True
                    booking_type_names = ['小型快件', '中型鲜活', '大型普货'] 
                    for type_name, prices_list in daily_prices_from_model.items(): 
                    day_price_data = {"bookingType": type_name, "dailyPrices": []}
                        for day_index, price in enumerate(prices_list): 
                        day_price_data["dailyPrices"].append({
                            "day": day_index + 1,
                            "price": round(float(price), 2)
                        })
                    booking_days_price_table.append(day_price_data)
            else:
                    logging.warning("收益模拟结果有效，但缺少 'daily_prices'。")
            else:
                logging.error("solve_wvs 未能成功填充 optimal_prices，无法进行收益模拟或价格提取。")
        except Exception as model_error:
            logging.error(f"动态规划模型计算过程中发生错误: {str(model_error)}")
            logging.error(traceback.format_exc())
            # calculation_succeeded remains False

        if not calculation_succeeded:
            logging.warning("由于计算未完全成功或daily_prices缺失，将使用模拟价格表（如果表为空）。")
            if not booking_days_price_table:
                booking_types_fallback = ['小型快件', '中型鲜活', '大型普货']
                for booking_type in booking_types_fallback:
                day_price_data = {"bookingType": booking_type, "dailyPrices": []}
                    base_price_fallback = 12 if booking_type == '小型快件' else (9 if booking_type == '中型鲜活' else 7)
                    # Use final_booking_period (actual from model) for fallback table generation
                    num_days_fallback = int(final_booking_period) 
                    for day in range(1, num_days_fallback + 1):
                        day_ratio_fallback = (day - 1) / max(1, num_days_fallback - 1)
                        price_factor_fallback = 0.85 + 0.4 * day_ratio_fallback
                        daily_price_fallback = round(base_price_fallback * price_factor_fallback, 2)
                        day_price_data["dailyPrices"].append({"day": day, "price": daily_price_fallback})
                booking_days_price_table.append(day_price_data)

        base_revenue_val, wvs_revenue_val, improvement_val, weight_util_val, volume_util_val = 0,0,0,0,0
        if calculation_succeeded and results_from_model and isinstance(results_from_model, dict):
            base_revenue_val = int(results_from_model.get('base_revenue', 0))
            wvs_revenue_val = int(results_from_model.get('wvs_revenue', 0))
            improvement_val = round((wvs_revenue_val / max(1,base_revenue_val) - 1) * 100, 1) if base_revenue_val > 0 else 0.0
            weight_util_val = int(results_from_model.get('weight_utilization', 0))
            volume_util_val = int(results_from_model.get('volume_utilization', 0))
        else:
            # Fallback/simulated metrics if calculation failed or results incomplete
            base_revenue_val = 950000; wvs_revenue_val = 1140000; improvement_val = 20.0
            weight_util_val = 91; volume_util_val = 85
            logging.info("使用模拟/默认的收益和利用率指标。")

        priceTable = []
        if booking_days_price_table:
            num_days = len(booking_days_price_table[0]['dailyPrices']) if booking_days_price_table else 0
            if num_days >= 3:
                stage1_end = num_days // 3
                stage2_end = 2 * num_days // 3
            else: 
                stage1_end = 1
                stage2_end = 1 if num_days < 2 else 2
            for item in booking_days_price_table:
                daily_prices_list = [p['price'] for p in item['dailyPrices']]
                early_price = round(np.mean(daily_prices_list[:stage1_end]), 2) if daily_prices_list[:stage1_end] else 0.0
                mid_price = round(np.mean(daily_prices_list[stage1_end:stage2_end]), 2) if daily_prices_list[stage1_end:stage2_end] else 0.0
                late_price = round(np.mean(daily_prices_list[stage2_end:]), 2) if daily_prices_list[stage2_end:] else 0.0
                priceTable.append({
                    "bookingType": item["bookingType"],
                    "earlyPrice": early_price, "midPrice": mid_price, "latePrice": late_price
                })

        result_payload = {
            'status': 'success' if calculation_succeeded else 'warning_simulated_data',
            'priceTable': priceTable, 
            'baseRevenue': base_revenue_val,
            'wvsRevenue': wvs_revenue_val,
            'improvement': improvement_val,
            'weightUtilization': weight_util_val,
            'volumeUtilization': volume_util_val,
            'route_info': route_info_for_response, # Display info from request
            'aircraft_info': {
                'type': aircraft_type_req, # Display aircraft type from request
                'capacity': { # Display actual capacity used by model
                    'max_payload': int(final_capacity_weight),
                    'max_volume': int(final_capacity_volume)
                }
            },
            'bookingDaysPriceTable': booking_days_price_table,
            'bookingPeriod': final_booking_period, 
            'effective_params_used_by_model': { 
                 'route_id_passed_to_model': route_id_for_model,
                 'capacity_weight': final_capacity_weight,
                 'capacity_volume': final_capacity_volume,
                 'time_periods': final_booking_period,
                 'gamma': final_gamma,
                 'c_d_ratio': final_cd_ratio,
                 'penalty_factor': model.penalty_factor, # Get actual from model instance
                 'cv': model.cv # Get actual from model instance
            }
        }
        
        log_message_prefix = "动态规划定价计算成功完成" if calculation_succeeded else "动态规划定价计算部分失败或使用模拟数据"
        logging.info(f"{log_message_prefix}. 返回结果: {result_payload}")
        # ... (rest of the detailed logging and printing to console/results_logger remains same) ...
        return jsonify(result_payload)
    
    except Exception as e:
        error_message = f"动态规划定价计算出错: {str(e)}"
        logging.error(error_message)
        traceback.print_exc()
        return jsonify({"status": "error", "message": error_message, "traceback": traceback.format_exc()}), 500

def extract_category_prices(optimal_prices, num_categories):
    """从动态规划模型的最优价格中提取各类别的价格"""
    try:
        if isinstance(optimal_prices, dict):
            # 如果是字典格式，尝试按照类别提取
            if 'category_prices' in optimal_prices:
                return optimal_prices['category_prices']
            else:
                # 尝试找到价格数据
                for key, value in optimal_prices.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) == num_categories:
                        return value
                # 如果找不到合适的数据，返回第一个预订期的价格
                first_period = list(optimal_prices.values())[0]
                if isinstance(first_period, (list, np.ndarray)):
                    return first_period
        
        elif isinstance(optimal_prices, (list, np.ndarray)):
            # 如果是列表或数组
            if len(optimal_prices) == num_categories:
                # 直接返回，因为长度正好匹配类别数
                return optimal_prices
            elif len(optimal_prices) > num_categories:
                # 返回前num_categories个元素
                return optimal_prices[:num_categories]
        
        # 默认情况：返回模拟数据
        return [8.5, 13.0, 7.0][:num_categories]
    
    except Exception as e:
        logging.error(f"提取类别价格失败: {str(e)}")
        # 返回默认值
        return [8.5, 13.0, 7.0][:num_categories]

@app.errorhandler(Exception)
def handle_exception(e):
    """全局异常处理"""
    logger.error(f"API错误: {str(e)}", exc_info=True)
    # 为所有响应启用CORS
    response = make_response(jsonify({
        'status': 'error',
        'message': str(e),
        'traceback': traceback.format_exc()
    }), 500)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/api/options', methods=['OPTIONS'])
def options():
    """处理预检请求"""
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

@app.route('/api/test', methods=['GET', 'POST'])
def api_test():
    """测试API，返回静态数据"""
    logger.info("测试API被调用")
    if request.method == 'POST':
        try:
            data = request.json
            logger.info(f"接收到POST数据: {data}")
        except Exception as e:
            logger.error(f"解析POST数据出错: {e}")
    
    # 返回静态测试数据
    return jsonify({
        'status': 'success',
        'message': 'API服务器正常工作',
        'test_data': {
            'timestamp': time.time(),
            'server_info': 'Flask API Server',
            'sample_prices': [10.5, 8.2, 6.7]
        }
    })

@app.route('/api/price_request_models', methods=['POST', 'OPTIONS'])
@cross_origin(supports_credentials=True)
def price_request_models():
    logger.info("API: Received request for /api/price_request_models")
    if request.method == 'OPTIONS':
        logger.debug("Handling OPTIONS request for /api/price_request_models")
        response = make_response()
        # 在 CORS(app) 中已经设置了全局的CORS头部，这里可以按需覆盖或添加
        # response.headers.add('Access-Control-Allow-Origin', '*') # 已全局设置
        # response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization') # 已全局设置
        # response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS') # 已全局设置
        return response

    try:
        request_data = request.json
        logger.debug(f"Request data: {request_data}")

        required_fields = ['route_id', 'cargo_type', 'weight_kg', 'volume_cm3']
        if not all(field in request_data for field in required_fields):
            logger.warning(f"Missing required fields in request: {request_data}")
            return jsonify({"error": "Missing required fields", "details": f"Required: {required_fields}"}), 400

        # --------------------------------------------------------------------
        # 1. 准备通用配置 (航线和飞机)
        # --------------------------------------------------------------------
        route_id_from_request = request_data.get('route_id') # 这是来自 freight_requests 的数字ID
        # TODO: 实现一个函数，根据数字 route_id 从数据库的 routes 表查找航线名（如 "大连-广州"）
        #       和关联的飞机类型（如 "A320"），然后加载或创建 RouteConfig 和 AircraftConfig。
        #       SELECT r.origin_city, r.destination_city, a.aircraft_type 
        #       FROM routes r JOIN aircrafts a ON r.aircraft_id = a.id 
        #       WHERE r.id = ?
        #       目前暂时硬编码使用默认的 "大连-广州" 航线和 "A320" 机型。
        
        # 模拟从数据库查询航线和机型名称
        # 在实际应用中，这里应该有一个数据库查询
        # route_db_id_to_name = {1: "大连-广州", 2: "上海-北京", ...} 
        # aircraft_db_id_to_name = {1: "A320", 2: "B737", ...}
        # 假设请求中的 route_id = 1 对应 "大连-广州", 默认用 "A320"
        
        route_id_str = "大连-广州" 
        aircraft_type_str = "A320"

        current_route_config = route_configs.get(route_id_str)
        current_aircraft_config = aircraft_configs.get(aircraft_type_str)

        if not current_route_config or not current_aircraft_config:
            logger.error(f"Default route/aircraft config not found. Route: {route_id_str}, Aircraft: {aircraft_type_str}")
            init_default_configs() 
            current_route_config = route_configs.get(route_id_str)
            current_aircraft_config = aircraft_configs.get(aircraft_type_str)
            if not current_route_config or not current_aircraft_config:
                 return jsonify({"error": "Default route/aircraft configuration not found after re-init."}), 500
        
        logger.debug(f"Using RouteConfig: {current_route_config.get_route_info()}")
        logger.debug(f"Using AircraftConfig: {current_aircraft_config.to_dict()}")

        results = {
            "request_details": request_data,
            "elasticity_model": {"price": None, "unit_price_kg": None},
            "gametheory_model": {"price": None, "unit_price_kg": None},
            "dynamicdp_model": {"price": None, "unit_price_kg": None},
            "errors": {}
        }
        weight_kg = float(request_data['weight_kg'])
        cargo_type = request_data['cargo_type']

        # --------------------------------------------------------------------
        # 2. 需求弹性模型 (ElasticityBasedPricingModel)
        # --------------------------------------------------------------------
        try:
            logger.info("Running Elasticity Model...")
            
            base_prices = {'普货': 10, '快件': 20, '鲜活': 15, '特种货物': 25, '危险品': 30, '其他': 8}
            elasticity_coeffs = {'普货': -1.5, '快件': -1.2, '鲜活': -1.8, '特种货物': -1.1, '危险品': -1.0, '其他': -2.0}
            
            base_op_cost = current_aircraft_config.config.get('base_operating_cost', 3500)
            fixed_cost_elasticity = base_op_cost * 0.1 
            max_payload_kg = current_aircraft_config.config.get('max_payload', 6000)
            if max_payload_kg == 0: max_payload_kg = 6000 # Avoid division by zero
            estimated_variable_cost_per_kg = (base_op_cost * 0.9) / max_payload_kg
            
            cargo_type_for_model = cargo_type if cargo_type in base_prices else '其他'

            elasticity_model_instance = ElasticityBasedPricingModel(
                route_config=current_route_config,
                aircraft_config=current_aircraft_config,
                base_price=base_prices.get(cargo_type_for_model, base_prices['其他']),
                elasticity_coefficient=elasticity_coeffs.get(cargo_type_for_model, elasticity_coeffs['其他']),
                fixed_cost=fixed_cost_elasticity, 
                variable_cost_per_unit=estimated_variable_cost_per_kg 
            )
            
            # optimize_price() 返回 P_optimal (最优单位价格), R_optimal, C_optimal, Q_optimal
            unit_price_kg, _, _, _ = elasticity_model_instance.optimize_price()
            
            if unit_price_kg is not None:
                total_price = round(unit_price_kg * weight_kg, 2)
                results["elasticity_model"] = {"price": total_price, "unit_price_kg": round(unit_price_kg, 2)}
                logger.info(f"Elasticity Model: Unit Price/kg = {unit_price_kg}, Total for request = {total_price}")
            else:
                results["errors"]["elasticity"] = "Elasticity model could not determine a price."

        except Exception as e:
            logger.error(f"Error in Elasticity Model: {e}")
            logger.error(traceback.format_exc())
            results["errors"]["elasticity"] = str(e)

        # --------------------------------------------------------------------
        # 3. 博弈论模型 (Game Theory Model)
        # --------------------------------------------------------------------
        try:
            logger.info("Running Game Theory Model...")
            # 简化参数设定 - 这些参数在实际应用中需要更精细的校准
            # 公司1 (我们) 的成本函数参数: C1 = a1 + b1*q1
            a1_cost = estimated_variable_cost_per_kg * weight_kg * 0.1 # 假设一个小的固定成本部分
            b1_cost = estimated_variable_cost_per_kg # 单位可变成本
            
            # 公司2 (竞争对手) 的成本函数参数: C2 = a2 + b2*q2
            a2_cost = a1_cost * 1.1 # 假设竞争对手固定成本略高
            b2_cost = b1_cost * 1.05 # 假设竞争对手单位成本略高

            # 需求函数参数: q1 = alpha1 - beta1*p1 + gamma1*p2
            # 对于单个请求，其本身就是需求。这里的需求函数参数更多是为了求解均衡价格。
            # 我们假设一个市场环境，其中价格敏感度（beta）和交叉价格敏感度（gamma）存在。
            # alpha 代表基础市场需求，这里也需要设定一个参考值。
            # 这些值非常粗略，仅为模型运行。
            base_market_demand_estimation = 1000 # 假设一个市场基础需求量 (kg)
            price_sensitivity_beta = base_market_demand_estimation / base_prices.get(cargo_type_for_model, base_prices['其他']) * 0.5 # 价格每变动1单位，需求变动百分比
            cross_price_sensitivity_gamma = price_sensitivity_beta * 0.3 # 交叉弹性通常小于自身弹性

            alpha1_demand = base_market_demand_estimation 
            beta1_demand = price_sensitivity_beta
            gamma1_demand = cross_price_sensitivity_gamma
            
            alpha2_demand = base_market_demand_estimation # 对称市场
            beta2_demand = price_sensitivity_beta
            gamma2_demand = cross_price_sensitivity_gamma

            game_model = AirCargoCompetitiveModel(
                cost_func1_params={'a': a1_cost, 'b': b1_cost},
                cost_func2_params={'a': a2_cost, 'b': b2_cost},
                demand_func1_params={'alpha': alpha1_demand, 'beta': beta1_demand, 'gamma': gamma1_demand},
                demand_func2_params={'alpha': alpha2_demand, 'beta': beta2_demand, 'gamma': gamma2_demand}
            )
            solver = NashEquilibriumSolver(game_model)

            # 定义价格搜索范围 - 基于成本和基准价
            min_price_search = b1_cost # 最低不应低于单位可变成本
            max_price_search = base_prices.get(cargo_type_for_model, base_prices['其他']) * 3 # 最高为基准价的3倍
            
            # solve_nash_equilibrium() 可能不接受 price_range 作为直接参数
            # NashEquilibriumSolver.solve_best_response_dynamics 需要价格范围
            # 查看 gametheory.py, solve_nash_equilibrium() 使用 scipy.optimize.fsolve
            # 它需要一个初始猜测。
            initial_guess = [base_prices.get(cargo_type_for_model, base_prices['其他']), 
                             base_prices.get(cargo_type_for_model, base_prices['其他']) * 1.1]

            # p1_star, p2_star, q1_star, q2_star = solver.solve_nash_equilibrium(initial_guess=initial_guess)
            # solve_nash_equilibrium 返回 p1, p2, q1, q2, rev1, rev2, profit1, profit2
            # 我们需要的是单位价格 p1_star

            equilibrium = solver.solve_nash_equilibrium(initial_guess=initial_guess)
            if equilibrium and len(equilibrium) >= 2 : # 确保返回了价格
                p1_star_unit_price = equilibrium[0] # 公司1的纳什均衡单位价格
                if p1_star_unit_price is not None and p1_star_unit_price > 0:
                    total_price_gt = round(p1_star_unit_price * weight_kg, 2)
                    results["gametheory_model"] = {"price": total_price_gt, "unit_price_kg": round(p1_star_unit_price, 2)}
                    logger.info(f"Game Theory Model: Nash Equilibrium Price/kg (p1*) = {p1_star_unit_price}, Total for request = {total_price_gt}")
                else:
                    results["errors"]["gametheory"] = "Game theory model resulted in an invalid price."
                    logger.warning(f"Game theory model resulted in an invalid price: {p1_star_unit_price}")
            else:
                results["errors"]["gametheory"] = "Game theory model could not determine Nash equilibrium."
                logger.warning("Game theory model could not determine Nash equilibrium.")

        except Exception as e:
            logger.error(f"Error in Game Theory Model: {e}")
            logger.error(traceback.format_exc())
            results["errors"]["gametheory"] = str(e)

        # TODO: 集成动态规划模型

        logger.info(f"API: Returning results for /api/price_request_models: {results}")
        return jsonify(results)

    except Exception as e:
        logger.error(f"General error in /api/price_request_models: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

# /api/optimize_cabin_fill端点已移除，系统将只专注于定价功能

if __name__ == '__main__':
    print("\n" + "*"*80)
    print("*"*20 + " 航空货运定价API服务器启动 " + "*"*20)
    print("*"*80)
    print("所有API调用的结果将在此控制台窗口中显示")
    print("*"*80 + "\n")
    print(f"启动API服务器，监听端口 8000")
    app.run(debug=True, host='0.0.0.0', port=8000) 