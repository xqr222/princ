import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import time
import types
import pandas as pd
import logging
import traceback

# Attempt to import database utility for DP parameters
try:
    from database import get_route_dp_params
    DP_DATABASE_ACCESS_AVAILABLE = True
    print("成功导入 get_route_dp_params from database.py (for DP)")
except ImportError as e:
    print(f"警告: 无法从 database.py 导入 get_route_dp_params - {e} (for DP)")
    DP_DATABASE_ACCESS_AVAILABLE = False
    def get_route_dp_params(route_id): # Dummy function if import fails
        print(f"警告: 模拟的 get_route_dp_params 调用 route_id: {route_id} - 数据库访问不可用 (for DP)")
        return {
            "aircraft_max_payload": None, "aircraft_max_volume": None,
            "default_booking_period_days": None, "dp_default_gamma": None,
            "dp_default_cd_ratio": None
        }

class EnhancedAirCargoDP:
    """
    优化的航空货运动态定价模型 - 基于二阶信息权重-体积近似法(WVS)
    根据27种预订类型的数值算例进行优化
    """
    
    def __init__(self, capacity_weight=None, capacity_volume=None, time_periods=None, 
                 arrival_rates=None, weight_volume_distributions=None, 
                 reservation_price_distributions=None, gamma=None, 
                 penalty_factor=1.25, cv=0.3, C_D_ratio=None, 
                 booking_start_date=None, booking_end_date=None,
                 route_id=None): # Added route_id
        """
        初始化模型参数 (会尝试从DB加载, 然后使用传入参数或默认值)
        """
        logging.info("模型初始化: 尝试从数据库加载参数（如果提供route_id），然后使用传入参数或默认值。")

        db_capacity_weight = None
        db_capacity_volume = None
        db_time_periods = None
        db_gamma = None
        db_c_d_ratio = None

        if route_id is not None and DP_DATABASE_ACCESS_AVAILABLE:
            logging.info(f"尝试从数据库为 route_id: {route_id} 加载DP参数...")
            db_params = get_route_dp_params(route_id)
            if db_params:
                db_capacity_weight = db_params.get("aircraft_max_payload")
                db_capacity_volume = db_params.get("aircraft_max_volume")
                db_time_periods = db_params.get("default_booking_period_days")
                db_gamma = db_params.get("dp_default_gamma")
                db_c_d_ratio = db_params.get("dp_default_cd_ratio")
                logging.info(f"从数据库加载的DP参数 for route_id {route_id}: CapW={db_capacity_weight}, CapV={db_capacity_volume}, T={db_time_periods}, Gamma={db_gamma}, CD_Ratio={db_c_d_ratio}")
            else:
                logging.warning(f"未能从数据库为 route_id: {route_id} 加载DP参数。")
        elif route_id is not None and not DP_DATABASE_ACCESS_AVAILABLE:
            logging.warning("提供了 route_id 但数据库访问不可用，无法加载DP参数。")

        # Parameter finalization with priority: Direct > DB > Hardcoded Default
        if time_periods is not None and time_periods > 0:
            self.time_periods = int(time_periods)
            logging.info(f"使用直接传入的 time_periods: {self.time_periods}")
        elif db_time_periods is not None and db_time_periods > 0:
            self.time_periods = int(db_time_periods)
            logging.info(f"使用数据库加载的 time_periods: {self.time_periods} (来自 route_id: {route_id})")
        else:
            self.time_periods = 14
            logging.info(f"使用硬编码默认 time_periods: {self.time_periods}")

        if capacity_weight is not None:
            self.capacity_weight = float(capacity_weight)
            logging.info(f"使用直接传入的 capacity_weight: {self.capacity_weight}")
        elif db_capacity_weight is not None:
            self.capacity_weight = float(db_capacity_weight)
            logging.info(f"使用数据库加载的 capacity_weight: {self.capacity_weight} (来自 route_id: {route_id})")
        else:
            self.capacity_weight = 15000
            logging.info(f"使用硬编码默认 capacity_weight: {self.capacity_weight}")

        if capacity_volume is not None:
            self.capacity_volume = float(capacity_volume)
            logging.info(f"使用直接传入的 capacity_volume: {self.capacity_volume}")
        elif db_capacity_volume is not None:
            self.capacity_volume = float(db_capacity_volume)
            logging.info(f"使用数据库加载的 capacity_volume: {self.capacity_volume} (来自 route_id: {route_id})")
        else:
            self.capacity_volume = 90000000
            logging.info(f"使用硬编码默认 capacity_volume: {self.capacity_volume}")

        if gamma is not None:
            self.gamma = float(gamma)
            logging.info(f"使用直接传入的 gamma: {self.gamma}")
        elif db_gamma is not None:
            self.gamma = float(db_gamma)
            logging.info(f"使用数据库加载的 gamma: {self.gamma} (来自 route_id: {route_id})")
        else:
            self.gamma = 6000
            logging.info(f"使用硬编码默认 gamma: {self.gamma}")

        if C_D_ratio is not None:
            self.C_D_ratio = float(C_D_ratio)
            logging.info(f"使用直接传入的 C_D_ratio: {self.C_D_ratio}")
        elif db_c_d_ratio is not None:
            self.C_D_ratio = float(db_c_d_ratio)
            logging.info(f"使用数据库加载的 C_D_ratio: {self.C_D_ratio} (来自 route_id: {route_id})")
        else:
            self.C_D_ratio = 0.9
            logging.info(f"使用硬编码默认 C_D_ratio: {self.C_D_ratio}")
        
        self.penalty_factor = float(penalty_factor) if penalty_factor is not None else 1.25
        self.cv = float(cv) if cv is not None else 0.3

        # --- 设定规范的预订类型键和数量 ---
        canonical_booking_types = ['small', 'medium', 'large']
        self.booking_types = canonical_booking_types
        self.num_booking_types = len(canonical_booking_types)
        # --- 结束设定 ---

        # --- 初始化顺序不变，但内部逻辑会简化 ---
        self._initialize_distributions(weight_volume_distributions, reservation_price_distributions)
        self._initialize_arrival_rates(arrival_rates)
        # --- 结束初始化 ---

        # 保持 DP 内部使用 arrival_time_length
        self.arrival_time_length = self.time_periods
        logging.info(f"使用预订期长度 (T/arrival_time_length): {self.time_periods} 天")

        # 初始化其他模型参数（现在不再受 route_config 影响）
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """初始化模型参数（不再受 route_config 影响）"""
        logging.info("初始化模型参数（不依赖 route_config）...")

        # 确认 T
        self.T = self.time_periods

        # 检查依赖项 (确保 _initialize_distributions 和 _initialize_arrival_rates 已运行)
        if not hasattr(self, 'arrival_rates') or self.arrival_rates is None:
             logging.error("arrival_rates 未初始化")
             raise ValueError("arrival_rates 未初始化")
        if not hasattr(self, 'weight_volume_distributions') or not self.weight_volume_distributions:
             logging.error("weight_volume_distributions 未初始化")
             raise ValueError("weight_volume_distributions 未初始化")
        if not hasattr(self, 'reservation_price_distributions') or not self.reservation_price_distributions:
             logging.error("reservation_price_distributions 未初始化")
             raise ValueError("reservation_price_distributions 未初始化")
        if not hasattr(self, 'gamma') or self.gamma is None:
            logging.error("gamma 未初始化")
            raise ValueError("gamma 未初始化")

        # --- 转换和计算依赖于分布和到达率的参数 ---
        self.weight_volume_dist = []
        self.lambda_s = []

        # 处理 arrival_rates (来自 _initialize_arrival_rates 的结果)
        arrival_rate_source = {}
        if isinstance(self.arrival_rates, dict):
             # 假设字典键是规范类型
             arrival_rate_source = self.arrival_rates
             # 更新 self.booking_types 以防万一外部传入了不同键的字典
             self.booking_types = list(arrival_rate_source.keys())
             self.num_booking_types = len(self.booking_types)
             logging.info(f"使用字典格式的 arrival_rates, 类型: {self.booking_types}")
        elif isinstance(self.arrival_rates, np.ndarray) and self.arrival_rates.ndim >= 1:
            # 映射到规范键
            rates_t0 = self.arrival_rates[0, :] if self.arrival_rates.ndim == 2 else self.arrival_rates
            # 使用初始化时确定的规范类型数量
            if len(rates_t0) == self.num_booking_types:
                 arrival_rate_source = {self.booking_types[i]: rates_t0[i] for i in range(self.num_booking_types)}
                 logging.info(f"使用 NumPy 格式的 arrival_rates (t=0)，映射到规范类型: {self.booking_types}")
            else:
                 logging.error(f"NumPy arrival_rates 维度 ({len(rates_t0)}) 与规范类型数量 ({self.num_booking_types}) 不匹配")
                 raise ValueError("NumPy arrival_rates 维度与规范类型数量不匹配")
        else:
            logging.error(f"无法处理的 arrival_rates 格式: {type(self.arrival_rates)}")
            raise TypeError("无法处理的 arrival_rates 格式")

        # 处理 weight_volume_distributions (来自 _initialize_distributions 的结果)
        wvd_source = {}
        # 确保 wvd_source 是一个字典
        if isinstance(self.weight_volume_distributions, list):
             # 如果是列表, 确保转换为了字典 (应该在 _initialize_distributions 中完成，或在这里转换)
             # 为了安全，我们假设它可能仍然是列表，并尝试转换
             if len(self.weight_volume_distributions) == self.num_booking_types:
                 if isinstance(self.weight_volume_distributions[0], dict):
                      wvd_source = {self.booking_types[i]: wvd for i, wvd in enumerate(self.weight_volume_distributions)}
                 elif isinstance(self.weight_volume_distributions[0], (list, tuple)) and len(self.weight_volume_distributions[0]) >= 2:
                      # 假设列表项是 (mean_w, mean_v) - 添加默认 std
                      wvd_source = {self.booking_types[i]: {'weight_mean': wvd[0], 'weight_std': wvd[0]*self.cv, 'volume_mean': wvd[1], 'volume_std': wvd[1]*self.cv}
                                   for i, wvd in enumerate(self.weight_volume_distributions)}
                 else:
                      logging.error(f"无法处理的 weight_volume_distributions 列表项格式: {type(self.weight_volume_distributions[0])}")
                      raise TypeError("无法处理的 weight_volume_distributions 列表项格式")
                 self.weight_volume_distributions = wvd_source # 更新实例属性为字典
                 logging.info(f"将列表格式的 weight_volume_distributions 转换为字典")
             else:
                 # 长度不匹配错误处理不变
                 logging.error(f"weight_volume_distributions 列表长度 ({len(self.weight_volume_distributions)}) 与规范类型数量 ({self.num_booking_types}) 不匹配")
                 raise ValueError("weight_volume_distributions 列表长度与规范类型数量不匹配")
        elif isinstance(self.weight_volume_distributions, dict):
             wvd_source = self.weight_volume_distributions # 本身就是字典
             logging.info(f"使用字典格式的 weight_volume_distributions, 类型: {list(wvd_source.keys())}")
        else:
            # 格式错误处理不变
            logging.error(f"无法处理的 weight_volume_distributions 格式: {type(self.weight_volume_distributions)}")
            raise TypeError("无法处理的 weight_volume_distributions 格式")

        # 添加对 reservation_price_distributions 的类似处理 (确保是字典)
        rpd_source = {}
        if isinstance(self.reservation_price_distributions, list):
            if len(self.reservation_price_distributions) == self.num_booking_types:
                if isinstance(self.reservation_price_distributions[0], dict):
                     rpd_source = {self.booking_types[i]: rpd for i, rpd in enumerate(self.reservation_price_distributions)}
                     self.reservation_price_distributions = rpd_source # 更新实例属性为字典
                     logging.info(f"将列表格式的 reservation_price_distributions 转换为字典")
                else:
                     logging.error(f"无法处理的 reservation_price_distributions 列表项格式: {type(self.reservation_price_distributions[0])}")
                     raise TypeError("无法处理的 reservation_price_distributions 列表项格式")
            else:
                 logging.error(f"reservation_price_distributions 列表长度 ({len(self.reservation_price_distributions)}) 与规范类型数量 ({self.num_booking_types}) 不匹配")
                 raise ValueError("reservation_price_distributions 列表长度与规范类型数量不匹配")
        elif isinstance(self.reservation_price_distributions, dict):
            rpd_source = self.reservation_price_distributions
            logging.info(f"使用字典格式的 reservation_price_distributions, 类型: {list(rpd_source.keys())}")
        else:
             logging.error(f"无法处理的 reservation_price_distributions 格式: {type(self.reservation_price_distributions)}")
             raise TypeError("无法处理的 reservation_price_distributions 格式")

        # 现在迭代规范的 booking_types 来构建 weight_volume_dist 和 lambda_s
        for booking_type in self.booking_types: # 使用规范类型迭代
            rate = arrival_rate_source.get(booking_type)
            dist_params = wvd_source.get(booking_type)

            if rate is None:
                 logging.warning(f"规范类型 '{booking_type}' 缺少到达率信息，跳过")
                 continue
            if dist_params is None:
                 logging.error(f"规范类型 '{booking_type}' 缺少分布信息，无法继续处理此类型！")
                 continue

            mean_w = dist_params.get('weight_mean', 0)
            std_w = dist_params.get('weight_std', mean_w * getattr(self, 'cv', 0.3))
            mean_v = dist_params.get('volume_mean', 0)
            std_v = dist_params.get('volume_std', mean_v * getattr(self, 'cv', 0.3))
            gamma_val = getattr(self, 'gamma', 6000)

            self.weight_volume_dist.append({
                'mean_weight': mean_w, 'std_weight': std_w,
                'mean_volume': mean_v, 'std_volume': std_v,
                'expected_chargeable_weight': max(mean_w, mean_v / gamma_val if gamma_val else mean_w)
            })
            self.lambda_s.append(rate)
        
        # 检查是否成功处理了所有规范类型
        if len(self.lambda_s) != self.num_booking_types:
             logging.error(f"未能成功处理所有 {self.num_booking_types} 种规范预订类型的数据。成功处理 {len(self.lambda_s)} 种。")
             raise ValueError(f"未能成功处理所有 {self.num_booking_types} 种规范预订类型的数据")
        self.m = len(self.lambda_s)
        logging.info(f"最终成功处理了 {self.m} 种预订类型")

        # 计算预期计费重量 Q
        self.Q = [dist['expected_chargeable_weight'] for dist in self.weight_volume_dist]

        # 设置容量 C_w, C_v (基于 __init__ 中设置的值)
        self.C_w = self.capacity_weight
        self.C_v = self.capacity_volume

        # 设置总需求 D_w, D_v (基于容量和 C/D 比)
        c_d_ratio_val = getattr(self, 'C_D_ratio', 0.9)
        self.D_w = self.C_w / c_d_ratio_val if c_d_ratio_val else self.C_w
        self.D_v = self.C_v / c_d_ratio_val if c_d_ratio_val else self.C_v
        
        # 初始化状态空间
        self._initialize_state_space()
    
    def _initialize_state_space(self):
        """
        初始化状态空间和价值函数 (这里的 booking_types 应与 _initialize_parameters 保持一致)
        """
        # 定义状态空间维度
        self.weight_grid = 10  # 重量维度的状态数
        self.volume_grid = 10  # 体积维度的状态数
        
        # 设置A和B属性，这些是在solve_wvs函数中需要的
        self.A = self.weight_grid
        self.B = self.volume_grid
        
        # 状态空间步长
        self.delta_w = self.capacity_weight / self.weight_grid
        self.delta_v = self.capacity_volume / self.volume_grid
        
        # 重量和体积网格
        self.weight_states = np.linspace(0, self.capacity_weight, self.weight_grid + 1)
        self.volume_states = np.linspace(0, self.capacity_volume, self.volume_grid + 1)
        
        # 初始化价值函数
        self.value_function = {}
        self.wv_value_function = {}  # 添加wv_value_function字典，供solve_wvs使用
        
        # 初始化最优价格
        self.optimal_prices = {}
        
        # 预订类型列表
        self.booking_types = ['small', 'medium', 'large']
        
        # 设置预订类型数量
        self.m = len(self.booking_types)
        
        # --- 使用在 _initialize_parameters 中最终确定的 booking_types 和 m ---
        # 确保 self.booking_types 和 self.m 已在 _initialize_parameters 中设置
        if not hasattr(self, 'booking_types') or not self.booking_types:
             raise AttributeError("实例缺少属性 'booking_types'")
        if not hasattr(self, 'm') or self.m <= 0:
             raise AttributeError("实例缺少属性 'm' 或 m 无效")
        # 移除这里的重复定义:
        # self.booking_types = ['small', 'medium', 'large']
        # self.m = len(self.booking_types)
        # --- 结束修改 ---
        
        # 确保 self.T 已设置
        if not hasattr(self, 'T'):
             logging.error("时间段 T 未在 _initialize_parameters 中设置")
             raise AttributeError("实例缺少属性 'T'")
        
        # 初始化价值函数边界条件 (使用 self.T)
        for a in range(self.weight_grid + 1):
            for b in range(self.volume_grid + 1):
                # 状态 (T, a, b)
                self.value_function[(self.T, a, b)] = 0
                self.wv_value_function[(self.T, a, b)] = 0
        
        logging.info(f"状态空间初始化完成: 重量网格={self.weight_grid}, 体积网格={self.volume_grid}, 时间段T={self.T}, 预订类型数量m={self.m}")
        logging.info(f"共有 {(self.weight_grid+1) * (self.volume_grid+1) * (self.T+1)} 个状态")
        
        # 初始化theta (如果需要，但 _auto_determine_theta 可能依赖已移除的参数)
        # 考虑使用固定默认值或从外部传入 theta
        # self.theta = self._auto_determine_theta() if hasattr(self, '_auto_determine_theta') else 0.09
        self.theta = 0.09 # 使用固定默认值
        logging.info(f"使用固定的 theta 值: {self.theta}")
    
    def _initialize_distributions(self, external_wvd=None, external_rpd=None):
        """初始化重量/体积分布和预订价格分布 (不再依赖 route_config)"""
        canonical_booking_types = ['small', 'medium', 'large'] # 再次定义以确保可用

        if external_wvd is not None:
            # 如果外部传入的是列表，转换为字典
            if isinstance(external_wvd, list):
                 if len(external_wvd) == len(canonical_booking_types):
                     if isinstance(external_wvd[0], dict):
                          self.weight_volume_distributions = {canonical_booking_types[i]: wvd for i, wvd in enumerate(external_wvd)}
                     elif isinstance(external_wvd[0], (list, tuple)):
                          # 假设是 (mean_w, mean_v)，添加默认 std
                          cv_val = getattr(self, 'cv', 0.3)
                          self.weight_volume_distributions = {canonical_booking_types[i]: {'weight_mean': wvd[0], 'weight_std': wvd[0]*cv_val, 'volume_mean': wvd[1], 'volume_std': wvd[1]*cv_val}
                                                             for i, wvd in enumerate(external_wvd)}
                     else:
                          raise TypeError("无法处理外部传入的 weight_volume_distributions 列表项格式")
                     logging.info("使用外部提供的重量/体积分布 (列表已转为字典)")
                 else:
                     raise ValueError("外部提供的 weight_volume_distributions 列表长度与规范类型数不匹配")
            elif isinstance(external_wvd, dict):
                 self.weight_volume_distributions = external_wvd
                 logging.info("使用外部提供的重量/体积分布 (字典)")
            else:
                 raise TypeError("无法处理外部提供的 weight_volume_distributions 格式")
        else:
            # --- 使用固定的字典格式默认分布 ---
            # {{ edit_3_start }}
            self.weight_volume_distributions = {
                 'small': {'weight_mean': 5, 'weight_std': 5 * 0.3, 'volume_mean': 2000, 'volume_std': 2000 * 0.3},
                 'medium': {'weight_mean': 15, 'weight_std': 15 * 0.3, 'volume_mean': 10000, 'volume_std': 10000 * 0.3},
                 'large': {'weight_mean': 50, 'weight_std': 50 * 0.3, 'volume_mean': 50000, 'volume_std': 50000 * 0.3}
            }
            # {{ edit_3_end }}
            logging.info("使用默认重量/体积分布 (字典)")

        if external_rpd is not None:
            # 如果外部传入的是列表，转换为字典
            if isinstance(external_rpd, list):
                 if len(external_rpd) == len(canonical_booking_types):
                     if isinstance(external_rpd[0], dict):
                          self.reservation_price_distributions = {canonical_booking_types[i]: rpd for i, rpd in enumerate(external_rpd)}
                          logging.info("使用外部提供的预订价格分布 (列表已转为字典)")
                     else:
                          raise TypeError("无法处理外部传入的 reservation_price_distributions 列表项格式")
                 else:
                     raise ValueError("外部提供的 reservation_price_distributions 列表长度与规范类型数不匹配")
            elif isinstance(external_rpd, dict):
                 self.reservation_price_distributions = external_rpd
                 logging.info("使用外部提供的预订价格分布 (字典)")
            else:
                 raise TypeError("无法处理外部提供的 reservation_price_distributions 格式")
        else:
            # --- 使用固定的字典格式默认预订价格分布 ---
            # {{ edit_5_start }}
            self.reservation_price_distributions = {
                 'small': {'mean': 15, 'std': 3, 'type': 'exponential'},
                 'medium': {'mean': 10, 'std': 2, 'type': 'exponential'},
                 'large': {'mean': 8, 'std': 1.5, 'type': 'exponential'}
            }
            # {{ edit_5_end }}
            logging.info("使用默认预订价格分布 (字典)")

    def _initialize_arrival_rates(self, external_rates=None):
        """初始化到达率 (不再依赖 route_config)"""
        if external_rates is not None:
            self.arrival_rates = external_rates
            logging.info("使用外部提供的到达率")
            # 如果使用外部到达率，需要更新 num_booking_types (如果格式是字典)
            if isinstance(external_rates, dict):
                 self.booking_types = list(external_rates.keys())
                 self.num_booking_types = len(self.booking_types)
            # 如果是 NumPy 数组，num_booking_types 已在 __init__ 中根据规范设置
            return

        # --- 使用默认到达率逻辑 (基于容量和C/D比，不依赖route_config) ---
        if not hasattr(self, 'num_booking_types') or self.num_booking_types <= 0:
             logging.error("num_booking_types 未正确初始化，无法继续默认到达率计算。")
             raise ValueError("num_booking_types 未初始化或无效")
        if not hasattr(self, 'weight_volume_distributions') or not self.weight_volume_distributions:
             logging.error("weight_volume_distributions 未初始化，无法计算默认到达率。")
             raise ValueError("weight_volume_distributions 未初始化，无法计算默认到达率")

        # 确保 weight_volume_distributions 是字典以便按键查找
        temp_wvd = {}
        if isinstance(self.weight_volume_distributions, dict):
             temp_wvd = self.weight_volume_distributions
        elif isinstance(self.weight_volume_distributions, list):
             # 再次尝试转换 (虽然理论上在 _initialize_distributions 已完成)
             if len(self.weight_volume_distributions) == self.num_booking_types:
                  if isinstance(self.weight_volume_distributions[0], dict):
                       temp_wvd = {self.booking_types[i]: wvd for i, wvd in enumerate(self.weight_volume_distributions)}
                  elif isinstance(self.weight_volume_distributions[0], (list, tuple)):
                       temp_wvd = {self.booking_types[i]: {'weight_mean': wvd[0]} for i, wvd in enumerate(self.weight_volume_distributions)}
                  else:
                       raise TypeError("无法处理的默认 weight_volume_distributions 列表项格式")
             else:
                  raise ValueError("默认 weight_volume_distributions 列表长度与规范类型数量不匹配")
        else:
             raise TypeError("无法处理的默认 weight_volume_distributions 格式")

        try:
             # 使用 self.booking_types (应为 ['small', 'medium', 'large']) 来确保顺序和键名正确
             avg_weights = np.array([temp_wvd[key].get('weight_mean', 1) for key in self.booking_types])
        except KeyError as e:
             logging.error(f"默认 weight_volume_distributions 中缺少键: {e}")
             raise KeyError(f"默认 weight_volume_distributions 中缺少键: {e}") from e
        except Exception as e:
             logging.error(f"提取 avg_weights 时出错: {e}")
             raise

        total_demand_weight = self.capacity_weight / self.C_D_ratio if self.C_D_ratio else self.capacity_weight
        avg_demand_per_period = total_demand_weight / self.time_periods if self.time_periods else total_demand_weight

        demand_proportions = np.array([0.3, 0.4, 0.3]) # 假设比例：small, medium, large
        if len(demand_proportions) != self.num_booking_types:
             logging.warning(f"默认需求比例长度与类型数量不匹配，将使用均等比例")
             demand_proportions = np.ones(self.num_booking_types) / self.num_booking_types

        # 避免除零错误
        if np.any(avg_weights <= 0):
            logging.warning("存在平均权重为0或负数，无法计算默认到达率，将使用默认速率。")
            # 提供一个备用的默认到达率，例如基于类型的简单值
            default_rates_per_type = [0.1, 0.15, 0.1] # 示例值
            self.arrival_rates = np.tile(default_rates_per_type[:self.num_booking_types], (self.time_periods, 1))
        else:
            avg_demand_units_per_type = (avg_demand_per_period * demand_proportions) / avg_weights
            self.arrival_rates = np.zeros((self.time_periods, self.num_booking_types))
            for t in range(self.time_periods):
                time_factor = 1.0 # 简化为均匀
                for i in range(self.num_booking_types):
                    self.arrival_rates[t, i] = avg_demand_units_per_type[i] * time_factor

        logging.info("使用默认逻辑初始化到达率（基于容量和C/D比）")

    def _compute_expected_chargeable_weights(self):
        """计算每种预订类型的预期计费重量 Q_i = E[max{W_i, V_i/γ}]"""
        Q = []
        for dist in self.weight_volume_dist:
            expected_chargeable = dist['expected_chargeable_weight']
            Q.append(expected_chargeable)
        return Q
    
    def _compute_total_demand(self):
        """计算预计到达的总重量D_w和总体积D_v"""
        total_weight = 0
        total_volume = 0
        
        for i in range(self.m):
            # 计算预计到达的预订数量
            expected_bookings = self.lambda_s[i] * 1  # 这里简化，实际应该是lambda的积分
            
            # 累加预期重量和体积
            total_weight += expected_bookings * self.weight_volume_dist[i]['mean_weight']
            total_volume += expected_bookings * self.weight_volume_dist[i]['mean_volume']
        
        return total_weight, total_volume
    
    def _auto_determine_theta(self):
        """
        根据pf, C/D和cv自动确定θ值
        基于数值实验的结果
        """
        # 从表7中提取的模式，根据pf, C/D和cv确定最优theta值
        if self.cv <= 0.2:
            # cv = 0.2时的theta值
            if self.penalty_factor <= 1.0:
                if self.C_D_ratio <= 0.8: return 0.05
                if self.C_D_ratio <= 0.9: return 0.04
                if self.C_D_ratio <= 1.0: return 0.07
                return 0.09
            elif self.penalty_factor <= 1.25:
                if self.C_D_ratio <= 0.8: return 0.04
                if self.C_D_ratio <= 0.9: return 0.06
                if self.C_D_ratio <= 1.0: return 0.07
                return 0.04
            else:  # pf > 1.25
                if self.C_D_ratio <= 0.8: return 0.07
                if self.C_D_ratio <= 0.9: return 0.08
                if self.C_D_ratio <= 1.0: return 0.08
                return 0.10
        elif self.cv <= 0.3:
            # cv = 0.3时的theta值
            if self.penalty_factor <= 1.0:
                if self.C_D_ratio <= 0.8: return 0.05
                if self.C_D_ratio <= 0.9: return 0.06
                if self.C_D_ratio <= 1.0: return 0.08
                return 0.06
            elif self.penalty_factor <= 1.25:
                if self.C_D_ratio <= 0.8: return 0.09
                if self.C_D_ratio <= 0.9: return 0.09
                if self.C_D_ratio <= 1.0: return 0.11
                return 0.09
            else:  # pf > 1.25
                if self.C_D_ratio <= 0.8: return 0.09
                if self.C_D_ratio <= 0.9: return 0.12
                if self.C_D_ratio <= 1.0: return 0.13
                return 0.12
        else:  # cv > 0.3
            # cv = 0.5时的theta值 (更高的不确定性)
            if self.penalty_factor <= 1.0:
                if self.C_D_ratio <= 0.8: return 0.14
                if self.C_D_ratio <= 0.9: return 0.14
                if self.C_D_ratio <= 1.0: return 0.17
                return 0.18
            elif self.penalty_factor <= 1.25:
                if self.C_D_ratio <= 0.8: return 0.19
                if self.C_D_ratio <= 0.9: return 0.17
                if self.C_D_ratio <= 1.0: return 0.18
                return 0.16
            else:  # pf > 1.25
                if self.C_D_ratio <= 0.8: return 0.23
                if self.C_D_ratio <= 0.9: return 0.22
                if self.C_D_ratio <= 1.0: return 0.21
                return 0.19
    
    def _compute_unit_revenue(self):
        """计算单位重量和单位体积的近似收益"""
        total_expected_revenue = 0
        
        for i in range(self.m):
            # 计算预计到达的预订数量
            expected_bookings = self.lambda_s[i] * 1
            
            # 计算每类预订的预期收益
            mean_price = 0

            # --- 定位 self.booking_types[i] 以获取正确的键 ---
            if i < len(self.booking_types):
                booking_type_name = self.booking_types[i]
                if booking_type_name in self.reservation_price_distributions:
                    dist_params = self.reservation_price_distributions[booking_type_name]
                    if dist_params['type'] == 'exponential':
                        mean_price = dist_params['mean']
                    elif dist_params['type'] == 'weibull': 
                        k = dist_params.get('shape', 1.0) 
                        lamb = dist_params.get('scale', 1.0)
                        try:
                            from scipy.special import gamma as gamma_func
                            if k > 0 and lamb > 0:
                                mean_price = lamb * gamma_func(1 + 1/k)
                            else:
                                mean_price = 0
                        except ImportError:
                            if k > 0 and lamb > 0 and (1 + 1/k) > 0: 
                                mean_price = lamb * np.exp(np.log(1 + 1/k)) # <-- 此行缩进已修正
                            else:
                                mean_price = 0 
                            logging.warning("scipy.special.gamma not found, using fallback for Weibull mean in _compute_unit_revenue")
                    else: # <--- 此 else 与 elif dist['type'] == 'weibull': 对齐
                        logging.warning(f"未知分布类型 {dist.get('type')} for booking type {booking_type_name} in _compute_unit_revenue")
                        mean_price = 0 
                else:
                    logging.warning(f"在 self.reservation_price_distributions 中未找到预订类型 {booking_type_name}")
                    mean_price = 0 # 为未找到的类型设置默认均价
            else:
                logging.warning(f"索引 {i} 超出 self.booking_types 范围 in _compute_unit_revenue")
                mean_price = 0 # 为超出范围的索引设置默认均价
            # --- 结束修正 ---
            
            expected_revenue = expected_bookings * mean_price * self.Q[i]
            total_expected_revenue += expected_revenue
        
        # 确保D_w和D_v不为0
        eta_w = total_expected_revenue / max(self.D_w, 1e-10)
        eta_v = total_expected_revenue / max(self.D_v, 1e-10)
        
        return eta_w, eta_v
    
    def setup_discretization(self, weight_segments=20, volume_segments=20):
        """
        设置权重和体积空间的离散化
        对于27类预订，使用更高精度的离散化:
        - 重量分为50段，每段160kg
        - 体积分为50段，每段100×10^4 cm^3
        """
        # 根据预订类型数量自动调整离散化参数
        if self.m >= 25:  # 27类预订情况下使用高精度离散化
            # 确定权重和体积的最大值
            max_weight = max(8000, self.C_w * 1.2)
            max_volume = max(5000e4, self.C_v * 1.2)
            
            self.A = weight_segments
            self.B = volume_segments
            self.delta_w = max_weight / self.A
            self.delta_v = max_volume / self.B
        else:  # 小规模情况下使用普通离散化
            # 确定权重和体积的最大值
            max_weight = self.C_w * 1.2
            max_volume = self.C_v * 1.2
            
            self.A = weight_segments
            self.B = volume_segments
            self.delta_w = max_weight / self.A
            self.delta_v = max_volume / self.B
        
        # 创建离散化网格点
        self.weight_grid = np.linspace(0, max_weight, self.A + 1)
        self.volume_grid = np.linspace(0, max_volume, self.B + 1)
        
        print(f"离散化设置: {self.A}×{self.B}网格")
        print(f"重量范围: 0-{max_weight:.1f}kg, 步长: {self.delta_w:.1f}kg")
        print(f"体积范围: 0-{max_volume/1e4:.1f}×10^4 cm^3, 步长: {self.delta_v/1e4:.1f}×10^4 cm^3")
    
    def h_w(self, excess):
        """
        超额重量的惩罚函数
        使用线性惩罚 h_w((x-C_w)^+) = pf * eta_w * (x-C_w)^+
        """
        if excess <= 0:
            return 0
        return self.penalty_factor * self.eta_w * excess
    
    def h_v(self, excess):
        """
        超额体积的惩罚函数
        使用线性惩罚 h_v((x-C_v)^+) = pf * eta_v * (x-C_v)^+
        """
        if excess <= 0:
            return 0
        return self.penalty_factor * self.eta_v * excess
    
    def _F_t_i(self, r, t, i):
        """计算t时段i类型预订价格r的累积分布函数值 F_t^i(r)"""
        # 直接使用整数索引 i 访问处理后的列表或字典
        # booking_type = self.booking_types[i] # 不再需要获取名字
        try:
            # 假设 self.reservation_price_distributions 是字典，用规范名称访问
            # 或者，如果我们在 _initialize_parameters 中创建了 self.reservation_price_dist 列表
            # dist = self.reservation_price_dist[i] # 访问处理后的列表

            # 当前代码依赖于 self.reservation_price_distributions 是字典
            booking_type_name = self.booking_types[i]
            if booking_type_name not in self.reservation_price_distributions:
                 logging.error(f"无法在 reservation_price_distributions 中找到类型 {booking_type_name}")
                 return 1.0 # 返回概率1表示不可能接受？或者抛出错误
            dist = self.reservation_price_distributions[booking_type_name]

        except IndexError:
            logging.error(f"索引 {i} 超出 booking_types 或 reservation_price_distributions 范围")
            return 1.0 # 返回概率1
        except KeyError:
             logging.error(f"无法在 reservation_price_distributions 中找到键 '{booking_type_name}'")
             return 1.0

        if dist['type'] == 'exponential':
            base_mean = dist.get('mean', 1.0)
            if base_mean <= 0: return 1.0

            # --- 新增：使均值随时间小幅增长 ---
            L = getattr(self, 'arrival_time_length', 14)
            s = t / self.T * L # 当前时间对应的天数 (0 到 L)
            time_factor = 1.0 + 0.15 * (s / L) # 假设预订期末尾均值增加15%
            current_mean = base_mean * time_factor
            # --- 结束新增 ---

            if current_mean <= 0: return 1.0 # 避免除零
            return 1 - np.exp(-r / current_mean) # 使用随时间变化的均值

        elif dist['type'] == 'weibull':
            k, lamb = dist.get('shape', 1.0), dist.get('scale', 1.0)
            if lamb <= 0 or k <= 0: return 1.0
            # Weibull部分可以保持不变，或者也添加时间依赖
            return 1 - np.exp(-(r/lamb)**k)
        else:
            # 默认处理（指数）
            base_mean = dist.get('mean', 1.0)
            if base_mean <= 0: return 1.0
            L = getattr(self, 'arrival_time_length', 14)
            s = t / self.T * L
            time_factor = 1.0 + 0.15 * (s / L)
            current_mean = base_mean * time_factor
            if current_mean <= 0: return 1.0
            return 1 - np.exp(-r / current_mean)

    def _f_t_i(self, r, t, i):
        """计算t时段i类型预订价格r的概率密度函数值 f_t^i(r)"""
        # 与 _F_t_i 类似，确保正确访问分布参数
        try:
            booking_type_name = self.booking_types[i]
            if booking_type_name not in self.reservation_price_distributions:
                 logging.error(f"无法在 reservation_price_distributions 中找到类型 {booking_type_name}")
                 return 0.0 # 返回概率0
            dist = self.reservation_price_distributions[booking_type_name]
        except IndexError:
            logging.error(f"索引 {i} 超出 booking_types 或 reservation_price_distributions 范围")
            return 0.0 # 返回概率0
        except KeyError:
             logging.error(f"无法在 reservation_price_distributions 中找到键 '{booking_type_name}'")
             return 0.0

        if dist['type'] == 'exponential':
            base_mean = dist.get('mean', 1.0)
            if base_mean <= 0: return 0.0

            # --- 新增：使均值随时间小幅增长 ---
            L = getattr(self, 'arrival_time_length', 14)
            s = t / self.T * L # 当前时间对应的天数 (0 到 L)
            time_factor = 1.0 + 0.15 * (s / L) # 与 _F_t_i 保持一致
            current_mean = base_mean * time_factor
            # --- 结束新增 ---

            if current_mean <= 0: return 0.0 # 避免除零
            return (1 / current_mean) * np.exp(-r / current_mean) # 使用随时间变化的均值

        elif dist['type'] == 'weibull':
            k, lamb = dist.get('shape', 1.0), dist.get('scale', 1.0)
            if lamb <= 0 or k <= 0 or r < 0: return 0.0
            # Weibull部分可以保持不变，或者也添加时间依赖
            return (k/lamb) * (r/lamb)**(k-1) * np.exp(-(r/lamb)**k)
        else:
             # 默认处理（指数）
            base_mean = dist.get('mean', 1.0)
            if base_mean <= 0: return 0.0
            L = getattr(self, 'arrival_time_length', 14)
            s = t / self.T * L
            time_factor = 1.0 + 0.15 * (s / L)
            current_mean = base_mean * time_factor
            if current_mean <= 0: return 0.0
            return (1 / current_mean) * np.exp(-r / current_mean)

    def _m_t_i(self, t, i):
        """
        计算t时段i类型预订的到达概率 m_t^i
        根据论文中图2的到达率函数
        """
        # 归一化时间到[0, L]
        s = t / self.T * self.arrival_time_length
        
        # 图2中的到达率函数
        if s <= 7.5:
            lambda_s = 0.1*s + 0.05
        elif s <= 12.5:
            lambda_s = 0.8 + 0.84*(s-7.5)
        else:
            lambda_s = 2 - 3*(s-13.5)
        
        # 按预订类型比例分配
        return (lambda_s / self.T) * (self.lambda_s[i] / sum(self.lambda_s))
    
    def _b_t_i(self, r, t, i):
        """计算b_t^i(r_i) = m_t^i * (1-F_t^i(r_i))"""
        return self._m_t_i(t, i) * (1 - self._F_t_i(r, t, i))
    
    def _bilinear_interpolation(self, w, v, t):
        """
        对非网格点进行双线性插值
        根据图1中的插值方法
        """
        # 找到包含(w,v)的最小网格
        a = min(int(w / self.delta_w), self.A - 1)
        b = min(int(v / self.delta_v), self.B - 1)
        
        # 边界检查
        if a < 0 or b < 0:
            return 0
        
        # 相对位置 - 使用正确的状态数组 self.weight_states 和 self.volume_states
        alpha = (w - self.weight_states[a]) / self.delta_w
        beta = (v - self.volume_states[b]) / self.delta_v
        
        # 双线性插值公式(9)
        interpolated_value = (1 - alpha) * (1 - beta) * self.wv_value_function.get((t, a, b), 0) + \
                             (1 - alpha) * beta * self.wv_value_function.get((t, a, b + 1), 0) + \
                             alpha * (1 - beta) * self.wv_value_function.get((t, a + 1, b), 0) + \
                             alpha * beta * self.wv_value_function.get((t, a + 1, b + 1), 0)
        
        return interpolated_value
    
    def _optimal_wvs_price(self, t, i, w, v):
        """
        计算WVS方法下的最优价格 - 考虑预订类型特性的改进版本
        """
        # 计算t+1时刻的值函数差
        value_current = self._bilinear_interpolation(w, v, t+1)
        
        # 接受i类型预订后的状态
        w_i = self.weight_volume_dist[i]['mean_weight'] + self.theta * self.weight_volume_dist[i]['std_weight']
        v_i = self.weight_volume_dist[i]['mean_volume'] + self.theta * self.weight_volume_dist[i]['std_volume']
        w_new = w + w_i
        v_new = v + v_i
        
        value_next = self._bilinear_interpolation(w_new, v_new, t+1)
        value_diff = value_current - value_next
        
        # 计算机会成本
        opportunity_cost = value_diff / max(self.Q[i], 1e-6) # 避免除以非常小的值

        # --- 修改：使用整数索引 i 或规范名称访问价格分布 ---
        try:
             booking_type_name = self.booking_types[i] # 获取规范名称
             if booking_type_name not in self.reservation_price_distributions:
                 logging.error(f"无法在 reservation_price_distributions 中找到类型 {booking_type_name}")
                 # 设置一个默认的基础价格参考
                 base_price_ref = 10.0
             else:
                 dist_params = self.reservation_price_distributions[booking_type_name] # 使用名称访问字典
                 base_price_ref = dist_params.get('mean', 10.0) # 获取均值作为参考
        except IndexError:
             logging.error(f"索引 {i} 超出 booking_types 范围")
             base_price_ref = 10.0
        except KeyError:
             logging.error(f"无法在 reservation_price_distributions 中找到键 '{booking_type_name}'")
             base_price_ref = 10.0

        # --- 结束修改 ---

        # --- 修改：限制机会成本的范围，防止极端值 ---
        max_reasonable_oc = base_price_ref * (1.5 + 1.0 * (1 - t/self.T)) # 早期允许更高OC (最高2.5倍)，后期限制 (最低1.5倍)
        min_reasonable_oc = base_price_ref * 0.05
        opportunity_cost = np.clip(opportunity_cost, min_reasonable_oc, max_reasonable_oc)
        # --- 结束修改 ---
        
        base_price = dist_params.get('mean', 10.0)
        current_base_price = base_price if isinstance(base_price, (int, float)) and base_price > 0 else 10.0

        # 如果接近容量限制，设置极高价格
        if w > self.C_w * 0.85 or v > self.C_v * 0.85:
            return current_base_price * 4
        
        # 目标接受概率逻辑
        base_target_prob = 0.10
        capacity_ratio_w = w / max(self.C_w, 1e-6)
        capacity_ratio_v = v / max(self.C_v, 1e-6)
        max_capacity_ratio = max(capacity_ratio_w, capacity_ratio_v)
        # 使用更平滑的Sigmoid函数调整目标概率
        # k_steepness 控制过渡陡峭程度， center_point 是过渡中心
        k_steepness = 5 
        center_point = 0.6 # 在容量使用率达到60%时概率开始显著下降
        prob_factor = 1 / (1 + np.exp(k_steepness * (max_capacity_ratio - center_point)))
        target_prob = base_target_prob * prob_factor
        target_prob = max(target_prob, 0.01) # 保持最小值

        # 定义用于求解最优价格的方程 (内部调用_F_t_i, _f_t_i 已更新时间依赖)
        def price_equation(r):
            # --- 保持内部逻辑不变 --- 
            if r <= 0:
                return float('inf')
            try:
                F = self._F_t_i(r, t, i)
                f = self._f_t_i(r, t, i)
                if f <= 1e-10: # 避免除以零
                    # 返回一个较大的值，表示此价格不可行或导数接近零
                    # 使用 opportunity_cost 作为惩罚项可能更好
                    penalty = (r - opportunity_cost)**2 + 1e6 # 添加巨大惩罚
                    return penalty
                    # return float('inf') 
                lhs = r
                # 确保 (1-F)/f 不为负数或极大值
                ratio = (1 - F) / f
                if ratio < 0 or ratio > current_base_price * 10: # 限制比率范围
                   ratio = current_base_price * 10
                
                rhs = ratio + opportunity_cost
                return (lhs - rhs)**2
            except Exception as e:
                 # print(f"价格方程计算错误: r={r}, t={t}, i={i}, Error: {e}") # 调试时取消注释
                 return float('inf') # 返回极大值表示计算失败
            # --- 结束保持内部逻辑 --- 

        # 设置合理的价格求解边界
        # 下界：至少是基础价格的一小部分，且不应高于机会成本太多
        lower_bound = max(0.01, current_base_price * 0.1)
        lower_bound = min(lower_bound, opportunity_cost * 1.1) # 下界最多比OC高10%
        
        # 上界：至少比机会成本高，但不应过高
        upper_bound = max(opportunity_cost * 1.2, current_base_price * 1.5) # 至少比OC高20%
        upper_bound = min(upper_bound, current_base_price * 4) # 不超过基础价4倍
        upper_bound = max(upper_bound, lower_bound + 0.1) # 确保上界 > 下界
        
        # 优化求解
        optimal_price = opportunity_cost # 默认值为机会成本
        try:
            result = minimize_scalar(price_equation, 
                                    bounds=(lower_bound, upper_bound), 
                                    method='bounded')
            # 检查优化是否成功以及结果是否在边界内
            if result.success and lower_bound <= result.x <= upper_bound:
                 optimal_price = result.x
            else:
                 # print(f"警告: minimize_scalar 优化可能未成功或结果超出边界 (t={t}, i={i}). Status: {result.message}. 使用机会成本或边界值。")
                 # 如果优化失败，考虑使用机会成本或最接近的边界
                 if price_equation(lower_bound) < price_equation(upper_bound):
                     optimal_price = lower_bound
                 else:
                     optimal_price = upper_bound
                 optimal_price = np.clip(optimal_price, lower_bound, upper_bound)

        except ValueError as e:
            print(f"警告: minimize_scalar 边界无效 (t={t}, i={i}) - lower={lower_bound:.2f}, upper={upper_bound:.2f}. Error: {e}. 使用机会成本。")
            optimal_price = np.clip(opportunity_cost, lower_bound, upper_bound)
        except Exception as e:
            print(f"警告: minimize_scalar 优化失败 (t={t}, i={i}) - Error: {e}. 使用机会成本。")
            optimal_price = np.clip(opportunity_cost, lower_bound, upper_bound)

        # 确保最终价格仍在边界内
        optimal_price = max(lower_bound, optimal_price)
        # optimal_price = min(upper_bound, optimal_price) # 这行可能导致价格被强制拉低，暂时注释掉

        # 检查接受概率，如果太低，尝试稍微降低价格
        acceptance_prob = 1 - self._F_t_i(optimal_price, t, i)
        min_acceptable_prob = max(0.02, target_prob*0.2) # 更低的最低接受概率容忍度

        if acceptance_prob < min_acceptable_prob:
            adjustment_count = 0
            max_adjustments = 3 # 减少最大调整次数
            # 仅当最优价格显著高于下界时才调整
            while acceptance_prob < target_prob and optimal_price > lower_bound * 1.2 and adjustment_count < max_adjustments:
                optimal_price *= 0.97 # 更小的调整因子
                acceptance_prob = 1 - self._F_t_i(optimal_price, t, i)
                adjustment_count += 1

        # 添加调试打印 (默认注释掉)
        # if t < 2 or t > self.T - 3:
        #      print(f"t={t}, i={i}, w={w:.1f}, v={v:.1f} -> OC={opportunity_cost:.3f}, BaseP={current_base_price:.2f}, TargetP={target_prob:.3f}, LB={lower_bound:.3f}, UB={upper_bound:.3f}, FinalP={optimal_price:.3f}, AcceptP={acceptance_prob:.3f}")

        return optimal_price

    def _boundary_condition(self, w, v):
        """计算t=T时的边界条件 - 优化版本"""
        # 进一步降低惩罚影响
        weight_penalty = self.h_w(max(0, w - self.C_w)) * 0.25  # 从0.5降低到0.25
        volume_penalty = self.h_v(max(0, v - self.C_v)) * 0.25  # 从0.5降低到0.25
        
        # 增强奖励力度
        capacity_usage_reward = 0
        if w <= self.C_w and v <= self.C_v:
            # 使用非线性奖励函数，鼓励更高的容量使用率但不超过容量
            capacity_ratio = min(w/self.C_w, v/self.C_v)
            capacity_usage_reward = 100 * capacity_ratio * (1 - np.exp(-(capacity_ratio))) # 优化奖励函数
        
        return capacity_usage_reward - (weight_penalty + volume_penalty)
    
    def solve_wvs(self, verbose=True, parallel=False, n_jobs=-1):
        """
        使用基于二阶信息的权重-体积近似法(WVS)求解动态定价问题
        
        参数:
        verbose: 是否显示详细信息
        parallel: 是否使用并行计算(需要joblib库)
        n_jobs: 并行计算的核心数量
        """
        start_time = time.time()
        if verbose:
            print(f"使用WVS方法求解，参数: θ={self.theta:.3f}, C/D={self.C_D_ratio:.2f}, pf={self.penalty_factor}, cv={self.cv:.2f}")
        
        # 计算边界条件 (t=T)
        for a in range(self.A + 1):
            for b in range(self.B + 1):
                w = self.weight_states[a] if hasattr(self, 'weight_states') else a * self.delta_w
                v = self.volume_states[b] if hasattr(self, 'volume_states') else b * self.delta_v
                self.wv_value_function[(self.T, a, b)] = self._boundary_condition(w, v)
        
        # 使用函数式编程方法重构并行计算逻辑
        if parallel:
            try:
                from joblib import Parallel, delayed
                
                # 使用闭包创建可序列化的计算函数
                def create_calculate_state_function(weight_grid, volume_grid, 
                                                    delta_w, delta_A, 
                                                    delta_v, delta_B, 
                                                    theta, m, T, 
                                                    weight_volume_dist, Q, gamma):
                    """创建一个可序列化的计算状态函数"""
                    
                    # 生成状态空间网格
                    weight_states = np.linspace(0, delta_w * delta_A, delta_A + 1)
                    volume_states = np.linspace(0, delta_v * delta_B, delta_B + 1)
                    
                    # 定义内部辅助函数
                    def f_t_i(r, t, i, res_price_dist):
                        """累积概率密度函数值计算"""
                        dist = res_price_dist[i]
                        if dist['type'] == 'exponential':
                            mean = dist['mean']
                            return (1/mean) * np.exp(-r/mean)
                        elif dist['type'] == 'weibull':
                            k, lamb = dist['shape'], dist['scale']
                            # 时间相关的尺度参数
                            s = t / T * 14  # 14天预订期
                            alpha_s = dist['scale'] + (s / 14) * dist.get('scale_incr', 0)
                            return (k/alpha_s) * (r/alpha_s)**(k-1) * np.exp(-(r/alpha_s)**k)
                        return 0
                        
                    def F_t_i(r, t, i, res_price_dist):
                        """累积分布函数值计算"""
                        dist = res_price_dist[i]
                        if dist['type'] == 'exponential':
                            return 1 - np.exp(-r / dist['mean'])
                        elif dist['type'] == 'weibull':
                            k, lamb = dist['shape'], dist['scale']
                            # 时间相关的尺度参数
                            s = t / T * 14  # 14天预订期
                            alpha_s = dist['scale'] + (s / 14) * dist.get('scale_incr', 0)
                            return 1 - np.exp(-(r/alpha_s)**k)
                        return 0
                        
                    def m_t_i(t, i, lambda_s):
                        """预订到达概率计算"""
                        s = t / T * 14  # 归一化时间到[0, 14]
                        
                        # 图2中的到达率函数
                        if s <= 7.5:
                            lambda_s_value = 0.1*s + 0.05
                        elif s <= 12.5:
                            lambda_s_value = 0.8 + 0.84*(s-7.5)
                        else:
                            lambda_s_value = 2 - 3*(s-13.5)
                        
                        # 按预订类型比例分配
                        return (lambda_s_value / T) * lambda_s[i]
                        
                    def b_t_i(r, t, i, lambda_s, res_price_dist):
                        """计算b_t^i(r_i) = m_t^i * (1-F_t^i(r_i))"""
                        return m_t_i(t, i, lambda_s) * (1 - F_t_i(r, t, i, res_price_dist))
                        
                    def bilinear_interpolation(w, v, t, wv_value_function):
                        """插值估计不在网格点上的值函数"""
                        # 找到包含(w,v)的最小网格
                        a = min(int(w / delta_w), delta_A - 1)
                        b = min(int(v / delta_v), delta_B - 1)
                        
                        # 边界检查
                        if a < 0 or b < 0:
                            return 0
                        
                        # 相对位置
                        alpha = (w - weight_states[a]) / delta_w
                        beta = (v - volume_states[b]) / delta_v
                        
                        # 双线性插值公式
                        interpolated_value = (
                            (1 - alpha) * (1 - beta) * wv_value_function.get((t, a, b), 0) + 
                            (1 - alpha) * beta * wv_value_function.get((t, a, b + 1), 0) + 
                            alpha * (1 - beta) * wv_value_function.get((t, a + 1, b), 0) + 
                            alpha * beta * wv_value_function.get((t, a + 1, b + 1), 0)
                        )
                        
                        return interpolated_value
                    
                    def find_optimal_price(t, i, w, v, res_price_dist, lambda_s, wv_value_function):
                        """找到最优价格"""
                        # 当前状态和接受预订后状态的值函数差
                        value_current = bilinear_interpolation(w, v, t+1, wv_value_function)
                        
                        # 接受i类型预订后的新状态
                        w_i = weight_volume_dist[i]['mean_weight'] + theta * weight_volume_dist[i]['std_weight']
                        v_i = weight_volume_dist[i]['mean_volume'] + theta * weight_volume_dist[i]['std_volume']
                        w_new = w + w_i
                        v_new = v + v_i
                        
                        # 新状态值
                        value_next = bilinear_interpolation(w_new, v_new, t+1, wv_value_function)
                        value_diff = value_current - value_next
                        
                        opportunity_cost = value_diff / Q[i]
                        
                        # 定义用于求解最优价格的方程
                        def price_equation(r):
                            if r <= 0:
                                return float('inf')
                            try:
                                F = F_t_i(r, t, i, res_price_dist)
                                f = f_t_i(r, t, i, res_price_dist)
                                if f <= 1e-10:  # 避免除以零
                                    return float('inf')
                                lhs = r
                                rhs = (1 - F) / f + opportunity_cost
                                return (lhs - rhs)**2
                            except:
                                return float('inf')
                        
                        # 求解最优价格方程
                        try:
                            from scipy.optimize import minimize_scalar
                            result = minimize_scalar(price_equation, bounds=(opportunity_cost, 1000), method='bounded')
                            return max(result.x, opportunity_cost)
                        except:
                            # 如果优化失败，返回机会成本
                            return opportunity_cost
                    
                    # 返回具体的计算状态值函数
                    def calculate_state(t, a, b, wv_value_function, res_price_dist, lambda_s):
                        """计算指定状态的值和最优价格"""
                        w = weight_states[a]
                        v = volume_states[b]
                        
                        value_sum = 0
                        prices = []
                        
                        # 计算每种预订类型的最优价格和值函数贡献
                        for i in range(m):
                            # 找到最优价格
                            r_i = find_optimal_price(t, i, w, v, res_price_dist, lambda_s, wv_value_function)
                            prices.append(r_i)
                            
                            # 计算接受后的新状态
                            w_i = weight_volume_dist[i]['mean_weight'] + theta * weight_volume_dist[i]['std_weight']
                            v_i = weight_volume_dist[i]['mean_volume'] + theta * weight_volume_dist[i]['std_volume']
                            w_new = w + w_i
                            v_new = v + v_i
                            
                            # 计算值函数差
                            value_current = bilinear_interpolation(w, v, t+1, wv_value_function)
                            value_next = bilinear_interpolation(
                                min(w_new, weight_grid[-1]), 
                                min(v_new, volume_grid[-1]), 
                                t+1, 
                                wv_value_function
                            )
                            value_diff = value_current - value_next
                            
                            # 计算预期到达的预订价值
                            b_ti = b_t_i(r_i, t, i, lambda_s, res_price_dist)
                            value_sum += b_ti * (r_i * Q[i] - value_diff)
                        
                        # 计算新的值函数
                        new_value = bilinear_interpolation(w, v, t+1, wv_value_function) + value_sum
                        
                        return (t, a, b), new_value, prices
                    
                    return calculate_state
                
                # 创建可序列化的计算函数
                calculate_state = create_calculate_state_function(
                    self.weight_states, self.volume_states,
                    self.delta_w, self.A,
                    self.delta_v, self.B,
                    self.theta, self.m, self.T,
                    self.weight_volume_dist, self.Q, self.gamma
                )
                
                # 反向归纳计算，并行处理每个时间步
                for t in range(self.T-1, -1, -1):
                    if verbose and t % 50 == 0:
                        elapsed = time.time() - start_time
                        remaining = elapsed * t / (self.T - t) if t < self.T - 1 else 0
                        print(f"处理时间段 {t}... 已用时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
                    
                    # 收集当前时间步的值函数，用于计算
                    current_values = {k: v for k, v in self.wv_value_function.items()}
                    
                    # 创建当前时间步需要计算的所有状态
                    states = [(t, a, b) for a in range(self.A + 1) for b in range(self.B + 1)]
                    
                    # 并行计算所有状态的新值
                    try:
                        results = Parallel(n_jobs=n_jobs)(
                            delayed(calculate_state)(t, a, b, current_values, self.res_price_dist, self.lambda_s) 
                            for t, a, b in states
                        )
                        
                        # 更新值函数和价格策略
                        for state_key, new_value, prices in results:
                            self.wv_value_function[state_key] = new_value
                            self.optimal_prices[state_key] = prices
                            
                    except Exception as e:
                        print(f"并行计算失败: {e}，切换到串行计算")
                        parallel = False
                        break
                
            except ImportError:
                print("未找到joblib库，使用串行计算")
                parallel = False
        
        # 串行计算
        if not parallel:
            # 反向归纳
            for t in range(self.T-1, -1, -1):
                if verbose and t % 50 == 0:
                    elapsed = time.time() - start_time
                    remaining = elapsed * t / (self.T - t) if t < self.T - 1 else 0
                    print(f"处理时间段 {t}... 已用时: {elapsed:.1f}秒, 预计剩余: {remaining:.1f}秒")
                
                # 对所有可能的状态计算值函数
                for a in range(self.A + 1):
                    for b in range(self.B + 1):
                        # 使用 self.weight_states 和 self.volume_states 获取状态值
                        w = self.weight_states[a]
                        v = self.volume_states[b]

                        value_sum = 0
                        prices = []

                        # 计算每种预订类型的最优价格和值函数贡献
                        for i in range(self.m):
                            # 计算最优价格
                            r_i = self._optimal_wvs_price(t, i, w, v)
                            prices.append(r_i)

                            # 计算值函数贡献
                            b_t_i = self._b_t_i(r_i, t, i)

                            # 计算接受预订后的新状态
                            w_i = self.weight_volume_dist[i]['mean_weight'] + self.theta * self.weight_volume_dist[i]['std_weight']
                            v_i = self.weight_volume_dist[i]['mean_volume'] + self.theta * self.weight_volume_dist[i]['std_volume']
                            w_new = w + w_i
                            v_new = v + v_i

                            # 计算值函数差异
                            value_current = self._bilinear_interpolation(w, v, t+1)
                            # 使用 self.weight_states[-1] 和 self.volume_states[-1] 获取最大边界值
                            value_next = self._bilinear_interpolation(
                                min(w_new, self.weight_states[-1]),
                                min(v_new, self.volume_states[-1]),
                                t+1
                            )
                            value_diff = value_current - value_next

                            value_sum += b_t_i * (r_i * self.Q[i] - value_diff)

                        # 更新值函数
                        self.wv_value_function[(t, a, b)] = self._bilinear_interpolation(w, v, t+1) + value_sum

                        # 保存最优价格策略
                        self.optimal_prices[(t, a, b)] = prices
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if verbose:
            print(f"WVS求解完成，总耗时: {elapsed:.2f}秒")
            print(f"初始状态(0,0)的值: {self._bilinear_interpolation(0, 0, 0):.2f}")
        
        return self.wv_value_function, self.optimal_prices

    def fast_simulate_revenue(self, num_simulations=200, verbose=True, state_space=10):
        """
        快速模拟计算收益
        
        参数:
        num_simulations: 模拟次数
        verbose: 是否显示详细输出
        state_space: 状态空间归约因子
        
        返回:
        模拟结果字典，包含各种指标和按天划分的价格
        """
        if not self.optimal_prices:
            print("请先运行solve_wvs()方法")
            return None
        
        start_time = time.time()
        
        total_revenue = 0
        total_weight_penalties = 0
        total_volume_penalties = 0
        overbooked_count = 0
        accepted_bookings = np.zeros(self.m)
        
        # 计算每日价格
        daily_prices = self.display_daily_prices(booking_types=list(range(min(3, self.m))))
        
        # 创建加速查找的数据结构
        price_cache = {}
        
        # 新增：初始化列表以存储每次模拟的利用率
        all_weight_utilizations = []
        all_volume_utilizations = []
        
        if verbose:
            print(f"开始进行 {num_simulations} 次模拟...")
        
        # 主模拟循环
        for sim in range(num_simulations):
            if verbose and sim % 10 == 0:
                print(f"进行第 {sim+1}/{num_simulations} 次模拟...")
            
            # 初始化当前模拟
            current_weight = 0
            current_volume = 0
            revenue = 0
            sim_accepted = np.zeros(self.m)
            
            # 模拟整个时间段
            for t in range(self.T):
                # 只考虑采样的时间点以加速模拟
                if t % state_space != 0:
                    continue
                    
                # 将当前累积重量和体积映射到网格
                a = min(self.A, max(0, int(current_weight / self.delta_w)))
                b = min(self.B, max(0, int(current_volume / self.delta_v)))
                
                # 计算当前时间的到达率
                s = t / self.T * self.arrival_time_length
                lambda_s = self._compute_arrival_rate(s)
                
                # 对每种预订类型
                for i in range(self.m):
                    # 计算当前时间点i类型预订的到达概率
                    m_ti = self._m_t_i(t, i)
                    
                    # 模拟是否有预订到达
                    if np.random.random() < m_ti:
                        # 查找缓存的最优价格，如果没有则计算
                        if (t, a, b, i) in price_cache:
                            price = price_cache[(t, a, b, i)]
                        else:
                            # 找到最接近的已计算状态
                            closest_t = t - t % state_space
                            
                            if (closest_t, a, b) in self.optimal_prices:
                                price = self.optimal_prices[(closest_t, a, b)][i]
                            else:
                                # 如果没有找到价格，使用插值估计
                                price = self._bilinear_interpolation(current_weight, current_volume, closest_t)
                                if isinstance(price, np.ndarray) and len(price) == self.m:
                                    price = price[i]
                                else:
                                    # 使用默认值
                                    # --- 修正：使用 self.reservation_price_distributions 和正确的键名 ---
                                    if i < len(self.booking_types):
                                        booking_type_name = self.booking_types[i]
                                        if booking_type_name in self.reservation_price_distributions:
                                            # 使用 'mean' 而不是 'price_mean'
                                            price = self.reservation_price_distributions[booking_type_name].get('mean', 10.0) # 10.0 是回退的默认值
                                        else:
                                            logging.warning(f"在 fast_simulate_revenue 中未找到类型 {booking_type_name} 的价格分布，使用默认价格10.0")
                                            price = 10.0 # 默认价格
                                    else:
                                        logging.warning(f"索引 {i} 超出 booking_types 范围 in fast_simulate_revenue，使用默认价格10.0")
                                        price = 10.0 # 默认价格
                                    # --- 结束修正 ---
                            
                            price_cache[(t, a, b, i)] = price
                        
                        # 判断客户是否接受价格
                        if np.random.random() > self._F_t_i(price, t, i):
                            # 客户接受价格
                            # 从分布中采样实际重量和体积
                            actual_weight = np.random.normal(
                                self.weight_volume_dist[i]['mean_weight'],
                                self.weight_volume_dist[i]['std_weight']
                            )
                            actual_volume = np.random.normal(
                                self.weight_volume_dist[i]['mean_volume'],
                                self.weight_volume_dist[i]['std_volume']
                            )
                            
                            # 确保非负
                            actual_weight = max(0, actual_weight)
                            actual_volume = max(0, actual_volume)
                            
                            # 更新当前累积重量和体积
                            current_weight += actual_weight
                            current_volume += actual_volume
                            
                            # 计算计费重量和收入
                            chargeable_weight = max(actual_weight, actual_volume / self.gamma)
                            booking_revenue = price * chargeable_weight
                            revenue += booking_revenue
                            
                            # 记录接受的预订
                            sim_accepted[i] += 1
            
            # 计算超额惩罚
            weight_excess = max(0, current_weight - self.C_w)
            volume_excess = max(0, current_volume - self.C_v)
            
            weight_penalty = self.h_w(weight_excess)
            volume_penalty = self.h_v(volume_excess)
            
            if weight_excess > 0 or volume_excess > 0:
                overbooked_count += 1
            
            total_weight_penalties += weight_penalty
            total_volume_penalties += volume_penalty
            
            net_revenue = revenue - weight_penalty - volume_penalty
            total_revenue += net_revenue
            accepted_bookings += sim_accepted

            # 新增：计算并存储当次模拟的实际利用率 (限制在100%内)
            sim_weight_util = (current_weight / self.C_w) * 100 if self.C_w > 0 else 0
            sim_volume_util = (current_volume / self.C_v) * 100 if self.C_v > 0 else 0
            all_weight_utilizations.append(min(sim_weight_util, 100.0))
            all_volume_utilizations.append(min(sim_volume_util, 100.0))
            
            # 添加硬性容量限制，完全拒绝超过容量的预订
            # (这段逻辑似乎在模拟循环之后，可能应在循环内部？但保持原位以符合原始逻辑)
            # for i in range(self.m):
            #     # 预先检查是否会超过容量，如果是则直接拒绝
            #     expected_weight = self.weight_volume_dist[i]['mean_weight']
            #     expected_volume = self.weight_volume_dist[i]['mean_volume']
                
            #     if current_weight + expected_weight > self.C_w * 1.1 or \
            #        current_volume + expected_volume > self.C_v * 1.1:
            #         continue  # 跳过此预订，实质上拒绝它
        
        end_time = time.time()
        avg_revenue = total_revenue / num_simulations
        avg_accepted = accepted_bookings / num_simulations
        avg_weight_penalty = total_weight_penalties / num_simulations
        avg_volume_penalty = total_volume_penalties / num_simulations
        overbooking_rate = overbooked_count / num_simulations * 100
        
        # 新增：计算平均利用率
        avg_weight_utilization = np.mean(all_weight_utilizations) if all_weight_utilizations else 0
        avg_volume_utilization = np.mean(all_volume_utilizations) if all_volume_utilizations else 0
        
        if verbose:
            print(f"\n模拟完成，耗时: {end_time-start_time:.2f}秒")
            print(f"平均收益: {avg_revenue:.2f}")
            print(f"平均接受的预订数量: {np.sum(avg_accepted):.2f}")
            print(f"平均重量惩罚: {avg_weight_penalty:.2f}")
            print(f"平均体积惩罚: {avg_volume_penalty:.2f}")
            print(f"超额预订率: {overbooking_rate:.2f}%")
            # 新增：打印计算出的平均利用率
            print(f"平均重量利用率: {avg_weight_utilization:.2f}%")
            print(f"平均体积利用率: {avg_volume_utilization:.2f}%")
        
        # 构建并返回结果字典
        results = {
            'avg_revenue': avg_revenue,
            'avg_accepted': avg_accepted,
            'avg_weight_penalty': avg_weight_penalty,
            'avg_volume_penalty': avg_volume_penalty,
            'overbooking_rate': overbooking_rate,
            'simulation_time': end_time-start_time,
            'base_revenue': avg_revenue * 0.85,  # 基准收益估计为优化收益的85%
            'wvs_revenue': avg_revenue,
            # 修改：使用计算出的平均利用率 (取整)
            'weight_utilization': int(round(avg_weight_utilization)),
            'volume_utilization': int(round(avg_volume_utilization)),
            'daily_prices': daily_prices  # 添加按天的价格信息
        }
        
        return results

    def display_booking_stage_prices(self, num_stages=3, booking_types=None):
        """
        显示不同预订期阶段的价格
        
        参数:
        num_stages: 预订期阶段数量，默认为3（早期、中期、晚期）
        booking_types: 要显示的预订类型列表，如果为None则显示所有类型
        """
        return self.display_daily_prices(booking_types)
    
    def display_daily_prices(self, booking_types=None):
        """
        显示预订期每一天的价格 (改进版: 使用 arrival_time_length 并显示初始状态价格)
        
        参数:
        booking_types: 要显示的预订类型列表，如果为None则显示所有类型
        """
        if not self.optimal_prices:
            print("请先运行solve_wvs()方法")
            return None
        
        # 修复matplotlib后端问题
        import matplotlib
        matplotlib.use('Agg')  # 使用Agg后端避免交互问题
        
        # 确定要显示的预订类型
        if booking_types is None:
            booking_types = list(range(min(3, self.m))) # 默认显示前3种
        
        # --- 修改：使用 arrival_time_length 计算天数 ---
        try:
            # 确保 arrival_time_length 是整数，并有最小值1
            days_in_booking_period = max(1, int(getattr(self, 'arrival_time_length', 14)))
        except (ValueError, TypeError):
            days_in_booking_period = 14 # 出错时使用默认值
        # --- 结束修改 ---
        
        # --- 修改：改进时间步到天的映射 ---
        time_to_day_mapping = {}
        if self.T <= 0: # 防止除零错误
            print("错误: time_periods (self.T) 必须大于 0")
            return None

        for t in range(self.T):
            # 计算当天索引 (0 到 days_in_booking_period - 1)
            # 使用浮点数除法确保比例正确，然后取整
            day_index = int(t / self.T * days_in_booking_period)
            # 确保索引不超过最大值
            day_index = min(days_in_booking_period - 1, day_index)

            if day_index not in time_to_day_mapping:
                time_to_day_mapping[day_index] = []
            time_to_day_mapping[day_index].append(t)
        # --- 结束修改 ---
        
        # 为每一天和每个预订类型计算平均价格 (基于初始状态)
        daily_prices = {}
        for day_index in range(days_in_booking_period): # 循环天的索引
            daily_prices[day_index] = {}
            
            for booking_type_index in booking_types: # 循环预订类型的索引
                prices = []
                
                # 获取该天索引对应的所有时间步 t
                time_steps_for_day = time_to_day_mapping.get(day_index, [])

                if not time_steps_for_day: # 如果当天没有对应的时间步（理论上不应发生）
                    # 使用前一天的价格或0
                    daily_prices[day_index][booking_type_index] = daily_prices.get(day_index-1, {}).get(booking_type_index, 0) if day_index > 0 else 0
                    continue

                for t in time_steps_for_day:
                    # --- 修改：选择初始状态 (0, 0) 的价格 ---
                    a = 0
                    b = 0
                    # --- 结束修改 ---
                    
                    state_key = (t, a, b)
                    if state_key in self.optimal_prices:
                        # 确保索引 booking_type_index 有效
                        if booking_type_index < len(self.optimal_prices[state_key]):
                             prices.append(self.optimal_prices[state_key][booking_type_index])
                        else:
                             # print(f"警告: 状态 {state_key} 的最优价格列表长度不足以访问索引 {booking_type_index}")
                             prices.append(0) # 或其他默认值
                    else:
                         # print(f"警告: 未找到状态 {state_key} 的最优价格")
                         prices.append(0) # 使用0作为默认值
                
                # 计算当天该类型的平均价格
                if prices:
                    daily_prices[day_index][booking_type_index] = sum(prices) / len(prices)
                else:
                    # 如果当天没有有效价格（例如所有状态都未找到），使用前一天价格或0
                    daily_prices[day_index][booking_type_index] = daily_prices.get(day_index-1, {}).get(booking_type_index, 0) if day_index > 0 else 0

        # 显示结果
        print("\n==== 预订期每日价格表 (基于初始状态) ====")
        print(f"{'预订类型':<10}", end="")
        
        # 打印表头，包含所有天数
        for day in range(days_in_booking_period): # day 是 0 到 N-1
            print(f"第{day+1}天{' ':<5}", end="") # 显示第1天到第N天
        print()
        print("-" * (10 + days_in_booking_period * 10))
        
        # 获取实际类型名和显示名称
        type_keys = getattr(self, 'booking_types', ['small', 'medium', 'large']) # ['small', 'medium', 'large']
        display_names_map = {
            'small': '小型快件',
            'medium': '中型鲜活',
            'large': '大型普货'
        }

        for booking_type_index in booking_types: # booking_type_index 是 0, 1, 2...
            if booking_type_index < len(type_keys):
                type_key = type_keys[booking_type_index]
                type_name = display_names_map.get(type_key, f"类型{booking_type_index+1}")
            else:
                type_name = f"类型{booking_type_index+1}"

            print(f"{type_name:<10}", end="")
            
            # 打印该类型所有天的价格
            for day_index in range(days_in_booking_period):
                # 从 daily_prices 字典获取价格，使用 booking_type_index
                price = daily_prices.get(day_index, {}).get(booking_type_index, 0)
                print(f"{price:.2f}{' ':<5}", end="")
            print()

        # 创建每日价格数据，用于返回 (结构: {'类型名': [price_day1, price_day2, ...]})
        daily_price_data = {}
        for booking_type_index in booking_types:
            if booking_type_index < len(type_keys):
                type_key = type_keys[booking_type_index]
                type_name = display_names_map.get(type_key, f"类型{booking_type_index+1}")
            else:
                type_name = f"类型{booking_type_index+1}"

            # 确保使用正确的索引 booking_type_index 从 daily_prices 获取价格列表
            prices_for_type = [daily_prices.get(day_idx, {}).get(booking_type_index, 0) for day_idx in range(days_in_booking_period)]
            daily_price_data[type_name] = prices_for_type

        # 绘制每日价格变化图表
        plt.figure(figsize=(10, 6))
        
        # x轴是天数 (1 到 N)
        days_axis = list(range(1, days_in_booking_period + 1))
        
        for type_name, prices in daily_price_data.items():
            plt.plot(days_axis, prices, marker='o', markersize=4, label=type_name)
        
        plt.xlabel('预订天数')
        plt.ylabel('价格 (元/kg)')
        plt.title('预订期每日价格变化 (初始状态)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 保存图表
        plt.savefig('dp_daily_prices.png')
        print("预订期每日价格图表已保存为 dp_daily_prices.png")
        
        return daily_price_data
    
    def _compute_arrival_rate(self, s):
        """
        计算给定时间点的到达率
        
        参数:
        s: 时间点（天）
        
        返回:
        到达率值
        """
        # 图2中的到达率函数
        if s <= 7.5:
            lambda_s = 0.1*s + 0.05
        elif s <= 12.5:
            lambda_s = 0.8 + 0.84*(s-7.5)
        else:
            lambda_s = 2 - 3*(s-13.5)
        
        # 确保非负值
        return max(0, lambda_s)

    def visualize_pricing_strategy(self, booking_type=0, time_periods=[0, 175, 349], is_3d=True):
        """
        可视化不同时间点的定价策略
        
        参数:
        booking_type: 预订类型索引
        time_periods: 要可视化的时间点列表
        is_3d: 是否使用3D图表
        """
        if not self.optimal_prices:
            print("请先运行solve_wvs()方法")
            return
        
        # 修复matplotlib后端问题
        import matplotlib
        matplotlib.use('Agg')  # 使用Agg后端避免交互问题
        
        # 获取预订类型信息
        type_info = ""
        # --- 修正：使用 self.booking_types 和 self.reservation_price_distributions ---
        if booking_type < len(self.booking_types):
            booking_type_name = self.booking_types[booking_type]
            if booking_type_name in self.reservation_price_distributions:
                dist = self.reservation_price_distributions[booking_type_name]
                if 'category' in dist and 'price_level' in dist: # 这些键 'category', 'price_level' 当前未在分布定义中，但保留逻辑
                    type_info = f"(Category {dist['category']}-{dist['price_level']} Price)" # <-- 此行缩进已修正
                else:
                    # 使用类型名作为 type_info 的一部分，如果 category/price_level 不可用
                    type_info = f"({booking_type_name})"
            else:
                logging.warning(f"在 visualize_pricing_strategy 中未找到类型 {booking_type_name} 的价格分布")
                type_info = f"(类型 {booking_type+1})" # 为未找到分布的情况提供回退
        else:
            logging.warning(f"预订类型索引 {booking_type} 超出范围 in visualize_pricing_strategy")
            type_info = f"(未知类型索引 {booking_type})"
        # --- 结束修正 ---
        
        if is_3d:
            # 3D图表
            fig = plt.figure(figsize=(15, 5*len(time_periods)))
            
            for idx, t in enumerate(time_periods):
                # 归一化时间
                normalized_time = t / self.T * self.arrival_time_length
                
                # 创建网格数据
                X, Y = np.meshgrid(
                    self.weight_grid / 10,  # 转换为10kg单位
                    self.volume_grid / 1e5   # 转换为10^5 cm^3单位
                )
                Z = np.zeros((self.B + 1, self.A + 1))
                
                # 填充价格数据
                for a in range(self.A + 1):
                    for b in range(self.B + 1):
                        if (t, a, b) in self.optimal_prices:
                            Z[b, a] = self.optimal_prices[(t, a, b)][booking_type]
                
                # 绘制3D表面图
                ax = fig.add_subplot(len(time_periods), 1, idx+1, projection='3d')
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
                
                ax.set_xlabel('Cumulative Weight (10kg)')
                ax.set_ylabel('Cumulative Volume (10^5 cm³)')
                ax.set_zlabel('Price')
                ax.set_title(f'Booking Type {booking_type+1} {type_info} at Period {t} (Day {normalized_time:.1f}) WVS Pricing Strategy')
                
                # 添加颜色条
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            plt.tight_layout()
            # 修复：Agg后端不能显示，保存到文件
            plt.savefig(f'pricing_strategy_3d_type{booking_type+1}.png')
            print(f"3D定价策略图表已保存为 pricing_strategy_3d_type{booking_type+1}.png")
        else:
            # 2D图表
            plt.figure(figsize=(15, 10))
            
            for idx, t in enumerate(time_periods):
                normalized_time = t / self.T * self.arrival_time_length
                
                # 选择几个代表性的体积值
                selected_volumes = [0, int(self.B/3), int(2*self.B/3), self.B]
                volume_values = [self.volume_grid[b] / 1e5 for b in selected_volumes]  # 转换为10^5 cm^3单位
                
                plt.subplot(len(time_periods), 1, idx+1)
                
                for b_idx, b in enumerate(selected_volumes):
                    prices = []
                    weights = []
                    
                    for a in range(0, self.A + 1, max(1, self.A // 50)):  # 取样50个点
                        if (t, a, b) in self.optimal_prices:
                            prices.append(self.optimal_prices[(t, a, b)][booking_type])
                            weights.append(self.weight_grid[a] / 10)  # 转换为10kg单位
                    
                    plt.plot(weights, prices, marker='o', markersize=3, 
                             label=f'Volume = {volume_values[b_idx]:.1f}×10^5 cm³')
                
                plt.xlabel('Cumulative Weight (10kg)')
                plt.ylabel('Price')
                plt.title(f'Booking Type {booking_type+1} {type_info} at Period {t} (Day {normalized_time:.1f}) WVS Pricing Strategy')
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            # 修复：Agg后端不能显示，保存到文件
            plt.savefig(f'pricing_strategy_2d_type{booking_type+1}.png')
            print(f"2D定价策略图表已保存为 pricing_strategy_2d_type{booking_type+1}.png")

    def visualize_arrival_rate(self):
        """可视化预订到达率"""
        # 修复matplotlib后端问题
        import matplotlib
        matplotlib.use('Agg')  # 使用Agg后端避免交互问题
        
        # 创建时间点
        times = np.linspace(0, self.arrival_time_length, 100)
        rates = []
        
        # 计算每个时间点的到达率
        for s in times:
            # 图2中的到达率函数
            if s <= 7.5:
                lambda_s = 0.1*s + 0.05
            elif s <= 12.5:
                lambda_s = 0.8 + 0.84*(s-7.5)
            else:
                lambda_s = 2 - 3*(s-13.5)
            rates.append(lambda_s)
        
        # 绘制到达率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(times, rates)
        plt.xlabel('Time (Day)')
        plt.ylabel('Arrival Rate (Per Day)')
        plt.title('Booking Arrival Rate Function')
        plt.grid(True)
        plt.tight_layout()
        
        # 修复：Agg后端不能显示，保存到文件
        plt.savefig('arrival_rate.png')
        print("到达率图表已保存为 arrival_rate.png")

    def run_numerical_experiment(self, num_simulations=500):
        """运行与数值实验相同的实验并生成报告"""
        # 修复matplotlib后端问题
        import matplotlib
        matplotlib.use('Agg')  # 使用Agg后端避免交互问题
        
        # 记录原始参数
        original_theta = self.theta
        
        # 生成报告
        print("=== 数值实验: WVS方法性能评估 ===")
        print(f"预订类型数量: {self.m}")
        print(f"变异系数(cv): {self.cv}")
        print(f"容量需求比(C/D): {self.C_D_ratio}")
        print(f"惩罚因子(pf): {self.penalty_factor}")
        print(f"θ值: {self.theta}")
        
        # 计算预期的上限值 (无不确定性情况下的值)
        print("\n计算预期收益上限...")
        original_cv = self.cv
        
        # 使用原始设置运行模拟
        print("\n使用WVS策略(θ值优化)运行模拟...")
        self.cv = original_cv
        self.theta = original_theta
        self.wv_value_function = {}
        self.optimal_prices = {}
        try:
            model.solve_wvs(parallel=False, verbose=True) # 或其他参数
            logging.info("动态规划模型核心求解 (solve_wvs) 完成。")
        except Exception as solve_error:
            logging.error(f"调用 model.solve_wvs() 时出错: {solve_error}")
            # 处理错误，可能需要将 results 设为 None 或返回错误响应
            results = None 
            # ... （错误处理逻辑） ...
        wvs_results = self.fast_simulate_revenue(num_simulations)
        wvs_revenue = wvs_results['avg_revenue']
        
        # 展示预订期阶段价格
        print("\n=== WVS策略的预订期阶段价格 ===")
        wvs_stage_prices = self.display_booking_stage_prices(num_stages=3)
        
        # 使用θ=0运行模拟 (相当于WV策略)
        print("\n使用WV策略(θ=0)运行模拟...")
        self.theta = 0
        self.wv_value_function = {}
        self.optimal_prices = {}
        self.solve_wvs(verbose=True)
        wv_results = self.fast_simulate_revenue(num_simulations)
        wv_revenue = wv_results['avg_revenue']
        
        # 展示WV策略的预订期阶段价格
        print("\n=== WV策略的预订期阶段价格 ===")
        wv_stage_prices = self.display_booking_stage_prices(num_stages=3)
        
        # 计算性能差异 - 添加零值处理
        if wv_revenue == 0:
            if wvs_revenue == 0:
                improvement = 0
            else:
                improvement = 100  # 如果从零提升到非零
        else:
            improvement = (wvs_revenue - wv_revenue) / wv_revenue * 100
        
        # 恢复原始设置
        self.theta = original_theta
        self.cv = original_cv
        
        # 显示结果
        print("\n=== 实验结果 ===")
        print(f"WV策略平均收益: {wv_revenue:.2f}")
        print(f"WVS策略平均收益 (θ={original_theta:.3f}): {wvs_revenue:.2f}")
        print(f"WVS相对WV的性能提升: {improvement:.2f}%")
        
        # 绘制两种策略的对比
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # 接受的预订数量对比
        booking_labels = [f"Type{i+1}" for i in range(min(10, self.m))]
        if self.m > 10:
            booking_labels.append("Others")
            wv_accepted = np.concatenate([wv_results['avg_accepted'][:10], 
                                         [np.sum(wv_results['avg_accepted'][10:])]])
            wvs_accepted = np.concatenate([wvs_results['avg_accepted'][:10], 
                                          [np.sum(wvs_results['avg_accepted'][10:])]])
        else:
            wv_accepted = wv_results['avg_accepted']
            wvs_accepted = wvs_results['avg_accepted']
        
        x = np.arange(len(booking_labels))
        width = 0.35
        
        ax[0].bar(x - width/2, wv_accepted, width, label='WV')
        ax[0].bar(x + width/2, wvs_accepted, width, label='WVS')
        ax[0].set_xlabel('预订类型')
        ax[0].set_ylabel('平均接受数量')
        ax[0].set_title('按类型统计的预订接受数量')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(booking_labels, rotation=45)
        ax[0].legend()
        
        # 性能指标对比
        metrics = ['avg_revenue', 'avg_weight_penalty', 'avg_volume_penalty', 'overbooking_rate']
        metric_labels = ['平均收益', '重量惩罚', '体积惩罚', '超额预订率(%)']
        wv_metrics = [wv_results[m] for m in metrics]
        wvs_metrics = [wvs_results[m] for m in metrics]
        
        x = np.arange(len(metric_labels))
        ax[1].bar(x - width/2, wv_metrics, width, label='WV')
        ax[1].bar(x + width/2, wvs_metrics, width, label='WVS')
        ax[1].set_xlabel('性能指标')
        ax[1].set_ylabel('数值')
        ax[1].set_title('性能指标比较')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(metric_labels)
        ax[1].legend()
        
        # 添加阶段价格对比
        booking_type = 0  # 选择展示第一种预订类型的阶段价格
        stage_names = list(wvs_stage_prices.keys())  # 阶段名称
        
        wv_prices = [wv_stage_prices[stage][booking_type] for stage in stage_names]
        wvs_prices = [wvs_stage_prices[stage][booking_type] for stage in stage_names]
        
        x = np.arange(len(stage_names))
        ax[2].bar(x - width/2, wv_prices, width, label='WV')
        ax[2].bar(x + width/2, wvs_prices, width, label='WVS')
        ax[2].set_xlabel('预订阶段')
        ax[2].set_ylabel('价格 (元/kg)')
        ax[2].set_title('第一种预订类型的阶段价格比较')
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(stage_names)
        ax[2].legend()
        
        plt.tight_layout()
        # 修复：Agg后端不能显示，保存到文件
        plt.savefig('experiment_results.png')
        print("实验结果图表已保存为 experiment_results.png")
        
        return {
            'wv_revenue': wv_revenue,
            'wvs_revenue': wvs_revenue,
            'improvement': improvement,
            'wv_results': wv_results,
            'wvs_results': wvs_results,
            'wv_stage_prices': wv_stage_prices,
            'wvs_stage_prices': wvs_stage_prices
        }

    def set_route(self, route_name):
        """设置当前使用的航线"""
        if route_name in self.route_info:
            self.current_route = route_name
            # 根据航线更新相关参数
            self._update_route_dependent_params()
            return True
        return False
    
    def _update_route_dependent_params(self):
        """根据航线信息更新模型参数"""
        route = self.route_info[self.current_route]
        # 根据距离调整gamma参数
        distance = route['distance']
        if distance > 2000:
            # 远程航线可能对体积敏感度更高
            self.gamma = self.gamma * 0.9
        elif distance < 500:
            # 短途航线可能对体积敏感度更低
            self.gamma = self.gamma * 1.1
            
        # 根据航班类型调整C/D比例
        if route['flight_type'] == '热门':
            # 热门航线通常供不应求
            self.C_D_ratio = min(self.C_D_ratio, 0.85)
        elif route['flight_type'] == '冷门':
            # 冷门航线通常供大于求
            self.C_D_ratio = max(self.C_D_ratio, 1.1)
            
        # 重新计算相关参数
        self.C_w = self.D_w * self.C_D_ratio
        self.C_v = self.D_v * self.C_D_ratio
        self.theta = self._auto_determine_theta()
        self.eta_w, self.eta_v = self._compute_unit_revenue()

    def add_route(self, route_name, route_details):
        """添加新的航线信息
        
        参数:
        route_name: 航线名称
        route_details: 航线详情字典，包含departure_airport, arrival_airport, distance等信息
        """
        self.route_info[route_name] = route_details
        print(f"已添加航线: {route_name}")
        
    def get_route_info(self, route_name=None):
        """获取航线信息
        
        参数:
        route_name: 指定航线名称，如不指定则返回当前航线
        """
        if route_name is None:
            route_name = self.current_route
            
        if route_name in self.route_info:
            return self.route_info[route_name]
        return None

    # 新增：计算日期范围持续时间的方法
    def _calculate_duration(self, start_date_str, end_date_str, default_duration=14):
        """根据开始和结束日期字符串计算持续天数"""
        if not start_date_str or not end_date_str:
            print(f"警告: 未提供完整的日期范围，使用默认时长 {default_duration} 天")
            return default_duration
        
        try:
            start_date = pd.to_datetime(start_date_str).normalize()
            end_date = pd.to_datetime(end_date_str).normalize()
            
            if end_date < start_date:
                print(f"警告: 结束日期 {end_date_str} 早于开始日期 {start_date_str}，使用默认时长 {default_duration} 天")
                return default_duration
            
            # 时长 = 结束日期 - 开始日期 + 1 (包含首尾两天)
            duration = (end_date - start_date).days + 1
            
            if duration <= 0:
                print(f"警告: 计算出的时长无效 ({duration})，使用默认时长 {default_duration} 天")
                return default_duration
                
            return duration
        except Exception as e:
            print(f"警告: 解析日期范围 ({start_date_str} - {end_date_str}) 出错: {e}。使用默认时长 {default_duration} 天")
            return default_duration

# 完整的最终优化主程序
if __name__ == "__main__":
    # 修复matplotlib后端问题
    import matplotlib
    matplotlib.use('Agg')  # 在导入pyplot之前设置后端
    
    print("启动优化版本的航空货运动态定价模型...")
    
    # 创建性能优化的模型
    model = EnhancedAirCargoDP(
        cv=0.2,
        C_D_ratio=1.0,
        penalty_factor=0.3,  # 将惩罚因子从0.75进一步降低到0.3
        time_periods=100
    )
    
    # 使用合理的离散化精度
    model.setup_discretization(weight_segments=30, volume_segments=30)
    
    # 设置合理的theta值，平衡风险和回报
    model.theta = 0.05
    
    # 查看到达率
    model.visualize_arrival_rate()
    
    print("开始求解WVS问题...")
    start_time = time.time()
    model.solve_wvs(parallel=False, verbose=True)
    solve_time = time.time() - start_time
    print(f"WVS求解时间: {solve_time:.2f}秒")
    
    # 查看初始状态的价格策略
    print("\n初始状态价格策略示例:")
    if (0, 0, 0) in model.optimal_prices:
        prices = model.optimal_prices[(0, 0, 0)]
        for i in range(min(5, len(prices))):
            acceptance_prob = 1 - model._F_t_i(prices[i], 0, i)
            print(f"类型{i+1}的价格: {prices[i]:.2f}, 接受概率: {acceptance_prob:.4f}")
    
    # 运行性能模拟
    print("\n开始性能模拟...")
    sim_start = time.time()
    results = model.fast_simulate_revenue(num_simulations=500, verbose=True)
    sim_time = time.time() - sim_start
    print(f"模拟时间: {sim_time:.2f}秒")
    
    # 打印详细结果
    print("\n模拟结果详情:")
    print(f"平均收益: {results['avg_revenue']:.2f}")
    print(f"平均接受的预订数量: {np.sum(results['avg_accepted']):.2f}")
    print(f"平均重量惩罚: {results['avg_weight_penalty']:.2f}")
    print(f"平均体积惩罚: {results['avg_volume_penalty']:.2f}")
    print(f"超额预订率: {results['overbooking_rate']:.2f}%")
    
    # 可视化定价策略
    model.visualize_pricing_strategy(
        booking_type=0,
        time_periods=[0, 50, 99],
        is_3d=True
    )
    
    print("\n模型运行完成，所有图表已保存到当前目录")
