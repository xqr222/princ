import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
import os
import platform
import logging

# Attempt to import database utility for Game Theory parameters
try:
    from database import get_route_gametheory_params
    GT_DATABASE_ACCESS_AVAILABLE = True
    print("成功导入 get_route_gametheory_params from database.py (for GameTheory)")
except ImportError as e:
    print(f"警告: 无法从 database.py 导入 get_route_gametheory_params - {e} (for GameTheory)")
    GT_DATABASE_ACCESS_AVAILABLE = False
    def get_route_gametheory_params(route_id): # Dummy function if import fails
        print(f"警告: 模拟的 get_route_gametheory_params 调用 route_id: {route_id} - 数据库访问不可用 (for GameTheory)")
        return {
            "gt_k_value": None, "gt_demand_base_factor": None,
            "gt_price_sensitivity_factor": None, "gt_cross_price_sensitivity_factor": None,
            "gt_cap_util_sensitivity_factor": None, "default_booking_period_days": None
        }

# Setup logger
logger = logging.getLogger('gametheory')
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if run multiple times
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# 添加导入AircraftConfig
try:
    from aircraft_config import AircraftConfig
    AIRCRAFT_CONFIG_AVAILABLE = True
except ImportError:
    print("警告: aircraft_config模块未找到，将使用默认机型参数。")
    AIRCRAFT_CONFIG_AVAILABLE = False

# 设置中文字体
def set_chinese_font():
    system = platform.system()
    if system == 'Windows':
        # Windows系统
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/msyh.ttc'     # 微软雅黑
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
                return True
    elif system == 'Linux':
        # Linux系统
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc'
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
                plt.rcParams['axes.unicode_minus'] = False
                return True
    elif system == 'Darwin':
        # macOS系统
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf'
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                plt.rcParams['font.sans-serif'] = ['PingFang SC']
                plt.rcParams['axes.unicode_minus'] = False
                return True
    
    # 如果找不到中文字体，使用matplotlib内置的字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return False

# 调用设置中文字体函数
set_chinese_font()

class AirCargoCompetitiveModel:
    """航空货运竞争博弈定价模型"""
    
    def __init__(self, company_id, W=None, route_info=None, route_config=None, 
                 k_value=None, initial_price=None, T_periods=None, flight_info=None, 
                 route_id=None): # Added route_id
        """
        初始化博弈论定价模型
        
        参数:
        company_id: 公司标识
        W: 总运力约束
        route_info: 航线信息 (将被 route_config 或 DB 信息覆盖)
        route_config: 航线配置对象 (将被 DB 信息部分覆盖)
        k_value: 需求参数 k (DB > direct_param > default)
        initial_price: 初始价格
        T_periods: 博弈的总阶段数 (DB > direct_param > default)
        flight_info: 航班信息
        route_id: 数据库中的航线ID (用于加载特定参数)
        """
        self.company_id = company_id
        self.route_id = route_id
        self.flight_info = flight_info

        # --- Parameter loading with priority: DB (via route_id) > Direct > Default/RouteConfig-derived ---
        db_params = None
        db_gt_k_value = None
        db_gt_demand_base_factor = None
        db_gt_price_sensitivity_factor = None
        db_gt_cross_price_sensitivity_factor = None
        db_gt_cap_util_sensitivity_factor = None
        db_default_booking_period_days = None

        if route_id is not None and GT_DATABASE_ACCESS_AVAILABLE:
            logger.info(f"Company {self.company_id}: Attempting to load GameTheory params from DB for route_id: {route_id}")
            db_params = get_route_gametheory_params(route_id)
            if db_params:
                db_gt_k_value = db_params.get("gt_k_value")
                db_gt_demand_base_factor = db_params.get("gt_demand_base_factor")
                db_gt_price_sensitivity_factor = db_params.get("gt_price_sensitivity_factor")
                db_gt_cross_price_sensitivity_factor = db_params.get("gt_cross_price_sensitivity_factor")
                db_gt_cap_util_sensitivity_factor = db_params.get("gt_cap_util_sensitivity_factor")
                db_default_booking_period_days = db_params.get("default_booking_period_days")
                logger.info(f"Company {self.company_id}: Loaded GT params from DB: k={db_gt_k_value}, T_periods={db_default_booking_period_days}, Factors=[{db_gt_demand_base_factor}, {db_gt_price_sensitivity_factor}, {db_gt_cross_price_sensitivity_factor}, {db_gt_cap_util_sensitivity_factor}]")
            else:
                logger.warning(f"Company {self.company_id}: Failed to load GT params from DB for route_id: {route_id}.")
        elif route_id is not None and not GT_DATABASE_ACCESS_AVAILABLE:
            logger.warning(f"Company {self.company_id}: route_id provided but DB access for GT params is not available.")

        # Finalize T_periods (Booking Horizon)
        # Priority: Direct argument > DB value > Default
        if T_periods is not None: # Direct argument from API/constructor
            self.T_periods = int(T_periods)
            logger.info(f"Company {self.company_id}: Using T_periods from direct argument: {self.T_periods}")
        elif db_default_booking_period_days is not None: # DB value (if direct arg was None)
            self.T_periods = int(db_default_booking_period_days)
            logger.info(f"Company {self.company_id}: Using T_periods from DB (as direct arg was None): {self.T_periods}")
        else: # Default (if both direct arg and DB value were None)
            self.T_periods = 14 
            logger.info(f"Company {self.company_id}: Using default T_periods: {self.T_periods}")

        # Finalize k_value
        # Priority: Direct argument > DB value > Default
        if k_value is not None: # Direct argument from API/constructor
            self.k_value = float(k_value)
            logger.info(f"Company {self.company_id}: Using k_value from direct argument: {self.k_value}")
        elif db_gt_k_value is not None: # DB value (if direct arg was None)
            self.k_value = float(db_gt_k_value)
            logger.info(f"Company {self.company_id}: Using k_value from DB (as direct arg was None): {self.k_value}")
        else: # Default (if both direct arg and DB value were None)
            self.k_value = 2.0 
            logger.info(f"Company {self.company_id}: Using default k_value: {self.k_value}")
        
        # Store DB-loaded factors for use in _initialize_model_specific_params
        self.db_gt_demand_base_factor = db_gt_demand_base_factor if db_gt_demand_base_factor is not None else 1.0
        self.db_gt_price_sensitivity_factor = db_gt_price_sensitivity_factor if db_gt_price_sensitivity_factor is not None else 1.0
        self.db_gt_cross_price_sensitivity_factor = db_gt_cross_price_sensitivity_factor if db_gt_cross_price_sensitivity_factor is not None else 1.0
        self.db_gt_cap_util_sensitivity_factor = db_gt_cap_util_sensitivity_factor if db_gt_cap_util_sensitivity_factor is not None else 1.0
        # --- End of new parameter loading logic --- 