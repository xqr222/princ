import os
import sys
import traceback  # 添加traceback模块用于输出详细错误
import json # Add for parsing params from DB if they are still JSON strings

# 解决PyCharm中matplotlib的显示问题
os.environ['MPLBACKEND'] = 'Agg'  # 通过环境变量设置后端为Agg

# 在任何其他导入之前设置matplotlib后端
import matplotlib
matplotlib.use('Agg', force=True)  # 强制使用Agg后端，避免显示问题
print("已强制设置matplotlib后端为Agg")

import numpy as np
from scipy.optimize import minimize

# 添加导入AircraftConfig
try:
    from aircraft_config import AircraftConfig
    AIRCRAFT_CONFIG_AVAILABLE = True
except ImportError:
    print("警告: aircraft_config模块未找到，将使用默认机型参数。")
    AIRCRAFT_CONFIG_AVAILABLE = False

# 将pandas相关的导入放在try-except块中
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("警告: pandas未安装，表格显示功能将不可用。")
    print("提示: 可以通过运行 'pip install pandas' 来安装pandas。")
    PANDAS_AVAILABLE = False

# 检测是否在PyCharm环境中
in_pycharm = 'PYCHARM_HOSTED' in os.environ or any('pycharm' in arg.lower() for arg in sys.argv)
if in_pycharm:
    print("检测到PyCharm环境")

# 将matplotlib相关的导入放在try-except块中
try:
    import matplotlib.pyplot as plt
    
    # 输出当前使用的后端
    print(f"Matplotlib版本: {matplotlib.__version__}")
    print(f"Matplotlib使用的后端: {matplotlib.get_backend()}")
    
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"警告: matplotlib导入错误，可视化功能将不可用。错误信息: {str(e)}")
    print("提示: 可以通过运行 'pip install matplotlib' 来安装matplotlib。")
    MATPLOTLIB_AVAILABLE = False
except Exception as e:
    print(f"警告: matplotlib初始化错误，可视化功能可能不正常。错误信息: {str(e)}")
    traceback.print_exc()  # 打印详细的错误堆栈
    MATPLOTLIB_AVAILABLE = False

# Attempt to import database utility
try:
    # Assuming database.py is in the same directory or accessible via PYTHONPATH
    from database import get_route_elasticity_params
    DATABASE_ACCESS_AVAILABLE = True
    print("成功导入 get_route_elasticity_params from database.py")
except ImportError as e:
    print(f"警告: 无法从 database.py 导入 get_route_elasticity_params - {e}")
    print("弹性模型将无法从数据库加载参数，将依赖外部传入或默认值。")
    DATABASE_ACCESS_AVAILABLE = False
    def get_route_elasticity_params(route_id): # Dummy function if import fails
        print(f"警告: 模拟的 get_route_elasticity_params 调用 route_id: {route_id} - 数据库访问不可用")
        return {"initial_prices": None, "coefficients": None, "base_demands": None}

class ElasticityBasedPricingModel:
    """基于价格弹性的航空货运定价模型"""
    
    def __init__(self, num_airlines=1, num_segments=1, num_cargo_types=3, 
                 route_config=None, booking_period=14, price_elasticity=None, 
                 initial_prices=None, base_demands=None, route_id=None):
        """
        初始化定价模型
        
        参数:
        num_airlines (int): 航空公司数量
        num_segments (int): 航段数量
        num_cargo_types (int): 货物类型数量 (例如: 快件、鲜活、普货)
        route_config (RouteConfig): 航线配置对象
        booking_period (int): 预订周期天数
        price_elasticity (numpy.ndarray, optional): 外部提供的价格弹性系数
        initial_prices (dict/numpy.ndarray, optional): 外部提供的初始价格 (按货物类型)
        base_demands (dict/numpy.ndarray, optional): 外部提供的基础需求 (按货物类型)
        route_id (int, optional): 航线ID，用于从数据库加载参数
        """
        # 设置模型维度
        self.m = num_airlines  # 航空公司数量
        self.n = num_segments  # 航段数量
        self.l = num_cargo_types  # 货物类型数量
        
        self.cargo_map = {'快件': 0, '鲜活': 1, '普货': 2} # Standardized cargo map
        # Ensure num_cargo_types matches the map size if map is fixed like this
        if self.l != len(self.cargo_map):
            print(f"警告:传入的 num_cargo_types ({self.l}) 与内置 cargo_map 大小 ({len(self.cargo_map)}) 不符。模型可能表现异常。")
            # Potentially adjust self.l or re-create self.cargo_map based on self.l
            # For now, we assume self.l is the authority if they differ, and cargo_map might be used for known types
            # A more robust approach would be to dynamically create cargo_map based on self.l or expected names
        
        # 保存传入的 route_config（可能包含机型信息），但后续不再用它计算核心参数
        self.route_config = route_config
        self.route_id_for_params = route_id # Store explicitly passed route_id
        if self.route_config and hasattr(self.route_config, 'route_id') and self.route_config.route_id is not None:
            # If route_config has an id, and no explicit route_id was passed, use it.
            if self.route_id_for_params is None:
                 self.route_id_for_params = self.route_config.route_id
            elif self.route_id_for_params != self.route_config.route_id:
                 print(f"警告: 传入的 route_id ({self.route_id_for_params}) 与 route_config 中的 ID ({self.route_config.route_id}) 不匹配。将优先使用直接传入的 route_id。")
        elif self.route_config and hasattr(self.route_config, 'id') and self.route_config.id is not None: # common way to store id in config obj
            if self.route_id_for_params is None:
                self.route_id_for_params = self.route_config.id
        
        # 设置预订周期 (直接使用传入的 booking_period)
        self.booking_period = booking_period
        
        # 初始化数组
        # P_S: 价格矩阵，维度为 (m, n, l)
        # d_S: 基础需求矩阵，维度为 (n, l)
        # price_elasticity: 价格弹性系数矩阵，维度为 (n, l)
        self.P_S_initial = np.zeros((self.m, self.n, self.l))
        self.d_S = np.zeros((self.n, self.l))
        self.price_elasticity = np.zeros((self.n, self.l))
        self.base_demands_array = np.zeros((self.n, self.l)) # New array for base demands
        
        # 为保证价格弹性计算有意义，设置最低需求比例
        self.min_demand_ratio = 0.1

        # 初始化参数，现在会尝试从数据库加载
        self._initialize_parameters_with_database_fallback(
            external_initial_prices=initial_prices, 
            external_price_elasticity=price_elasticity, 
            external_base_demands=base_demands
        )
        
        # 运行时生成的优化结果
        self.max_revenue = 0
        self.optimal_prices = None

        # 记录当前模型状态
        print(f"初始化完成: {self.m}家航空公司, {self.n}个航段, {self.l}种货物类型, {self.booking_period}天预订期")
        if self.route_config:
            route_info = self.route_config.get_route_info() if hasattr(self.route_config, 'get_route_info') else vars(self.route_config)
            print(f"航线: {route_info.get('origin','N/A')} - {route_info.get('destination','N/A')}, 距离: {route_info.get('distance','N/A')}km")
        elif self.route_id_for_params:
            print(f"将使用 route_id: {self.route_id_for_params} 尝试从数据库加载参数。")
        
    def _initialize_parameters_with_database_fallback(self, external_initial_prices=None, external_price_elasticity=None, external_base_demands=None):
        """初始化模型参数，优先从数据库加载，然后是外部传入，最后是默认值。"""
        db_params = None
        loaded_from_db = False
        if DATABASE_ACCESS_AVAILABLE and self.route_id_for_params is not None:
            print(f"尝试从数据库为 route_id: {self.route_id_for_params} 加载弹性参数...")
            db_params = get_route_elasticity_params(self.route_id_for_params)
            if db_params and (db_params.get('initial_prices') or db_params.get('coefficients') or db_params.get('base_demands')):
                print(f"成功从数据库加载参数: {db_params}")
                loaded_from_db = True
            else:
                print(f"未能从数据库为 route_id: {self.route_id_for_params} 加载有效参数，或参数为空。")
        else:
            if not DATABASE_ACCESS_AVAILABLE:
                print("数据库访问不可用，跳过从数据库加载参数。")
            if self.route_id_for_params is None:
                print("未提供 route_id，跳过从数据库加载参数。")

        # 使用加载到的数据库参数（如果可用），否则使用外部传入的参数
        # Cargo map is now self.cargo_map
        default_initial_prices_dict = {name: (12.0 if idx == 0 else (7.0 if idx == 1 else 6.0)) for name, idx in self.cargo_map.items()}
        default_elasticity_dict = {name: (-1.7 if idx == 0 else (-1.2 if idx == 1 else -1.5)) for name, idx in self.cargo_map.items()}
        default_base_demands_dict = {name: 60 for name in self.cargo_map.keys()}

        # 1. Initial Prices (P_S_initial)
        source_initial_prices = "默认值"
        final_initial_prices_dict = default_initial_prices_dict
        if loaded_from_db and db_params.get('initial_prices'):
            final_initial_prices_dict = db_params['initial_prices']
            source_initial_prices = "数据库"
        elif external_initial_prices and isinstance(external_initial_prices, dict):
            final_initial_prices_dict = external_initial_prices
            source_initial_prices = "外部传入(dict)"
        elif external_initial_prices and isinstance(external_initial_prices, np.ndarray):
            # Convert np.ndarray to dict if shape is compatible (m,n,l) or (n,l) or (l)
            # This part needs careful handling based on how external_initial_prices array is structured
            # For now, we assume if it's an array, it has to be converted to dict pérdidas outside or this logic needs expansion
            print(f"警告: 外部传入的 initial_prices 是 NumPy 数组，当前实现期望字典或从数据库加载。将尝试使用默认值。外部数组: {external_initial_prices}")
            # Fallback to default if direct dict conversion is not straightforward here
        print(f"初始价格将基于: {source_initial_prices}")
        for k in range(self.m): # airlines
            for i in range(self.n): # segments
                for cargo_name, idx in self.cargo_map.items():
                    if idx < self.l: # ensure cargo index is within bounds
                        self.P_S_initial[k, i, idx] = float(final_initial_prices_dict.get(cargo_name, default_initial_prices_dict.get(cargo_name, 0)))

        # 2. Price Elasticity (price_elasticity)
        source_elasticity = "默认值"
        final_elasticity_dict = default_elasticity_dict
        if loaded_from_db and db_params.get('coefficients'):
            final_elasticity_dict = db_params['coefficients']
            source_elasticity = "数据库"
        elif external_price_elasticity and isinstance(external_price_elasticity, dict):
            final_elasticity_dict = external_price_elasticity
            source_elasticity = "外部传入(dict)"
        elif external_price_elasticity is not None and isinstance(external_price_elasticity, np.ndarray):
            # 现在可以安全地访问 external_price_elasticity.shape
            if external_price_elasticity.shape == self.price_elasticity.shape:
                self.price_elasticity = external_price_elasticity.copy()
                source_elasticity = "外部传入(NumPy直接赋值)"
                print(f"价格弹性系数将基于: {source_elasticity}")
                # Skip dict-based assignment if array is directly assigned
                # for i in range(self.n): # This loop is redundant if array assigned
                #     for cargo_name, idx in self.cargo_map.items(): 
                #         pass # Values already set
            else: # external_price_elasticity is a numpy array but shape doesn't match
                print(f"警告: 外部传入的 price_elasticity (NumPy) 形状不匹配 ({external_price_elasticity.shape} vs {self.price_elasticity.shape})。将使用字典或默认值。")
                # source_elasticity remains "默认值" or "数据库" or "外部传入(dict)" from previous conditions, leading to dict-based assignment later
        # If external_price_elasticity was None or an unhandled type, source_elasticity will also lead to dict-based assignment.
        
        if source_elasticity != "外部传入(NumPy直接赋值)": # Only use dict if not directly assigned from numpy
            print(f"价格弹性系数将基于: {source_elasticity}")
            for i in range(self.n): # segments
                for cargo_name, idx in self.cargo_map.items():
                    if idx < self.l:
                        self.price_elasticity[i, idx] = float(final_elasticity_dict.get(cargo_name, default_elasticity_dict.get(cargo_name, 0)))

        # 3. Base Demands (base_demands_array, formerly d_S)
        source_base_demands = "默认值"
        final_base_demands_dict = default_base_demands_dict
        if loaded_from_db and db_params.get('base_demands'):
            final_base_demands_dict = db_params['base_demands']
            source_base_demands = "数据库"
        elif external_base_demands and isinstance(external_base_demands, dict):
            final_base_demands_dict = external_base_demands
            source_base_demands = "外部传入(dict)"
        elif external_base_demands and isinstance(external_base_demands, np.ndarray):
            # Similar to initial_prices, needs logic to map array to (n,l) structure if passed
            if external_base_demands.shape == self.base_demands_array.shape:
                self.base_demands_array = external_base_demands.copy()
                source_base_demands = "外部传入(NumPy直接赋值)"
                print(f"基础需求将基于: {source_base_demands}")
                # Skip dict-based assignment
        else:
                print(f"警告: 外部传入的 base_demands (NumPy) 形状不匹配。将尝试使用字典或默认值。")
        
        if source_base_demands != "外部传入(NumPy直接赋值)":
            print(f"基础需求将基于: {source_base_demands}")
            for i in range(self.n): # segments
                for cargo_name, idx in self.cargo_map.items():
                    if idx < self.l:
                        self.base_demands_array[i, idx] = float(final_base_demands_dict.get(cargo_name, default_base_demands_dict.get(cargo_name, 0)))
         
        # Ensure d_S (old name) is also updated if other parts of the code still use it.
        # self.d_S should be an alias or replaced by self.base_demands_array
        self.d_S = self.base_demands_array 

        # 移除或确保注释掉基于航线特性的调整调用，因为参数现在更直接
        # if self.route_config: ...

        print(f"参数初始化完成。 P_S_initial Min/Max: {np.min(self.P_S_initial)}/{np.max(self.P_S_initial)}")
        print(f"Price Elasticity Min/Max: {np.min(self.price_elasticity)}/{np.max(self.price_elasticity)}")
        print(f"Base Demands (d_S) Min/Max: {np.min(self.d_S)}/{np.max(self.d_S)}")
    
    def calculate_demand_with_elasticity(self, P_S, P_S_initial, segment_index, cargo_index):
        """
        基于价格弹性计算新的需求量
        P_S: 当前价格
        P_S_initial: 基准价格 (来自 self.P_S_initial)
        segment_index: 航段索引
        cargo_index: 货物类型索引
        """
        # 获取对应的基准价格
        # 注意：这里的 P_S_initial 参数是调用时传入的，通常应该是 self.P_S_initial[airline, segment, cargo]
        # 而不是整个 self.P_S_initial 数组。
        # 调用者需要传递正确的标量基准价格。
        
        # 获取基础需求 (D0)
        # 使用 self.d_S (已经被设置为 self.base_demands_array 的别名)
        base_demand_D0 = self.d_S[segment_index, cargo_index]
        
        # 计算价格变化百分比
        if P_S_initial <= 0: # 避免除以零
            # 如果初始价为0或负数，且新价也为0或负数，则需求为基础需求 (比例为1)
            # 如果初始价为0或负数，但新价为正，则需求为0 (价格从不存在变为存在，弹性模型可能不适用，极端情况)
            return base_demand_D0 * (1.0 if P_S <= 0 else self.min_demand_ratio) 
        
        price_change_ratio = (P_S - P_S_initial) / P_S_initial
        
        # 获取对应的价格弹性系数
        elasticity = self.price_elasticity[segment_index, cargo_index]
        
        # 基于弹性系数计算需求变化比例 (1 + E * (dP/P))
        demand_ratio = 1 + elasticity * price_change_ratio
        
        # 应用最低需求保障，并乘以基础需求 D0
        # 新需求 D = D0 * max(min_ratio, (1 + E * (dP/P)))
        actual_demand = base_demand_D0 * max(self.min_demand_ratio, demand_ratio) 
        return actual_demand
    
    def calculate_revenue(self, P_S_flat):
        """
        计算给定临时价格下的总收益
        """
        try:
            expected_size = self.m * self.n * self.l
            if len(P_S_flat) != expected_size:
                # 尝试修正大小 (如果可能)
                P_S_flat = self._fix_array_size(P_S_flat, expected_size, self.P_S_initial)
                if len(P_S_flat) != expected_size:
                    print(f"错误: 无法修正价格数组大小以匹配预期 {expected_size}")
                    return -1e10 # 返回一个很大的负数

            P_S = P_S_flat.reshape((self.m, self.n, self.l))
            total_revenue = 0
            
            for k in range(self.m):
                for i in range(self.n):
                    for j in range(self.l):
                        # calculate_demand_with_elasticity returns the actual demand, not a ratio
                        calculated_demand = self.calculate_demand_with_elasticity(
                            P_S[k, i, j], 
                            self.P_S_initial[k, i, j], 
                            i, j
                        )
                        # Revenue is Price * Actual_Calculated_Demand
                        current_revenue = P_S[k, i, j] * calculated_demand
                        total_revenue += current_revenue
            
            return -total_revenue # 返回负收益用于最小化
            
        except Exception as e:
            print(f"计算收益时出错: {str(e)}")
            traceback.print_exc()
            return -1e10 
    
    def optimize_prices(self):
        """
        优化价格以最大化总收益 (考虑所有航段和货物类型)
        假设航空公司数量m=1

        返回:
        numpy.ndarray: 每个航段每种货物的最优价格, 形状 (n, l)
        float: 最大优化后收益
        float: 初始总收益
        list: 需求详情列表
        list: 每日价格列表
        """
        if self.m != 1:
            print("警告: 当前价格优化仅支持单一航空公司 (m=1)")
            num_segments = self.n if hasattr(self, 'n') else 1
            num_cargo_types = self.l if hasattr(self, 'l') else 3
            empty_prices = np.zeros((num_segments, num_cargo_types))
            return empty_prices, 0, 0, [], [] # Added 0 for initial_revenue

        initial_prices_flat = self.P_S_initial[0].flatten()
        objective_func = lambda p_flat: -self.calculate_revenue(p_flat) # calculate_revenue already returns -total_revenue
        # So objective_func will be -(-total_revenue) = total_revenue. We want to MINIMIZE -total_revenue.
        # Correct objective_func for MINIMIZE should be self.calculate_revenue (which returns -total_revenue)
        objective_func = self.calculate_revenue # CORRECTED

        bounds = []
        for p_init in initial_prices_flat:
            lower_bound = max(0.1, p_init * 0.5)
            upper_bound = p_init * 3.0
            bounds.append((lower_bound, upper_bound))

        print("执行价格优化 (弹性模型)...")
        result = minimize(objective_func, initial_prices_flat, method='SLSQP', bounds=bounds)

        initial_total_revenue = 0
        # Calculate initial total revenue based on self.P_S_initial and self.d_S
        # Assuming m=1 (first airline)
        for i_seg in range(self.n): # segments
            for j_cargo in range(self.l): # cargo types
                initial_price = self.P_S_initial[0, i_seg, j_cargo]
                base_demand_for_cargo = self.d_S[i_seg, j_cargo] # d_S is base_demands_array
                initial_total_revenue += initial_price * base_demand_for_cargo
        print(f"计算得到的初始总收益: {initial_total_revenue:.2f}")

        if result.success:
            self.optimal_prices = result.x.reshape((self.n, self.l))
            self.max_revenue = -result.fun # result.fun is -total_optimized_revenue, so this is total_optimized_revenue
            
            print(f"基础价格优化成功! 最优收益: {self.max_revenue:.2f}, 初始收益: {initial_total_revenue:.2f}")
            
            daily_prices_list = []
            # Use self.cargo_map keys for names, fallback to "类型X" if map is not aligned with self.l
            cargo_type_names_map_keys = list(self.cargo_map.keys())
            cargo_type_names = [cargo_type_names_map_keys[i] if i < len(cargo_type_names_map_keys) else f'类型{i+1}' for i in range(self.l)]

            for c_idx in range(self.l):
                prices_for_cargo = []
                current_segment_idx_for_daily = 0 
                if self.n == 0:
                     print("错误: 航段数量n为0，无法计算每日价格。")
                
                start_price = self.P_S_initial[0, current_segment_idx_for_daily, c_idx] 
                end_price = self.optimal_prices[current_segment_idx_for_daily, c_idx]   

                for day in range(1, self.booking_period + 1):
                    price_today = start_price + (end_price - start_price) * (day / self.booking_period)
                    prices_for_cargo.append({'day': day, 'price': round(price_today, 2)})
                
                cargo_name = cargo_type_names[c_idx] # Use the derived cargo_type_names
                daily_prices_list.append({'cargoType': cargo_name, 'dailyPrices': prices_for_cargo})

            demand_details_list = []
            for i in range(self.n): # segments
                for j in range(self.l): # cargo types
                    cargo_name = cargo_type_names[j] # Use the derived cargo_type_names
                    calculated_demand = self.calculate_demand_with_elasticity(
                        P_S=self.optimal_prices[i, j],
                        P_S_initial=self.P_S_initial[0, i, j],
                        segment_index=i, 
                        cargo_index=j
                    )
                    avg_weight_per_unit = 10 # kg, placeholder
                    demand_in_kg = calculated_demand * avg_weight_per_unit
                    
                    # Get initial price for this cargo type and segment
                    initial_price_for_cargo = self.P_S_initial[0, i, j]

                    demand_details_list.append({
                        "cargo_type": cargo_name,
                        "segment": i,
                        "initial_price_unit": round(initial_price_for_cargo, 2), # ADDED
                        "optimal_price_unit": round(self.optimal_prices[i, j], 2), # Renamed for clarity
                        "calculated_demand_units": round(calculated_demand, 2),
                        "estimated_demand_kg": round(demand_in_kg, 2)
                    })
            
            print(f"每日价格计算完成。")
            return self.optimal_prices, self.max_revenue, initial_total_revenue, demand_details_list, daily_prices_list
        else:
            print(f"警告: 优化器未能成功收敛或遇到问题: {result.message}")
            print(f"优化器结果详情: {result}")
            
            # 回退到初始价格
            initial_prices_flat_for_opt = np.array(initial_prices_flat, dtype=float)
            print(f"由于优化失败，回退到使用初始价格: {initial_prices_flat_for_opt}")

            demand_details_list_on_failure = []
            daily_prices_list_on_failure = []
            
            # 确定货物名称列表
            cargo_names_ordered = ['快件', '鲜活', '普货'] 
            if self.l < len(cargo_names_ordered):
                 cargo_names_ordered = cargo_names_ordered[:self.l]
            elif self.l > len(cargo_names_ordered):
                 for i_generic in range(len(cargo_names_ordered), self.l):
                     cargo_names_ordered.append(f"货物类型{i_generic+1}")

            # 为每个货物类型创建空的每日价格列表和基础的demand_details
            for j in range(self.l):
                cargo_name = cargo_names_ordered[j] if j < len(cargo_names_ordered) else f"未知货物{j+1}"
                
                daily_prices_list_on_failure.append({
                    "cargoType": cargo_name,
                    "dailyPrices": [] # 空的每日价格
                })
                
                # 假设单航司 (m=0), 单航段 (n=0)
                initial_price_unit = 0.0
                base_demand_unit = 0.0
                if self.P_S_initial.ndim == 3 and self.P_S_initial.shape[0] > 0 and self.P_S_initial.shape[1] > 0 and self.P_S_initial.shape[2] > j:
                    initial_price_unit = self.P_S_initial[0, 0, j]
                if self.base_demands_array.ndim == 2 and self.base_demands_array.shape[0] > 0 and self.base_demands_array.shape[1] > j:
                    base_demand_unit = self.base_demands_array[0, j]
                    
                demand_details_list_on_failure.append({
                    "segment": 0, 
                    "cargo_type_index": j,
                    "cargo_type": cargo_name,
                    "initial_price_unit": initial_price_unit,
                    "optimal_price_unit": initial_price_unit, # 回退到初始价格
                    "initial_demand_unit": base_demand_unit,
                    "optimal_demand_unit": 0, # 优化失败，假设需求为0或基础需求
                    "initial_revenue_unit": 0, # 可以根据初始价格和需求估算
                    "optimal_revenue_unit": 0
                })

            initial_total_revenue_on_failure, _ = self.calculate_revenue(self.P_S_initial)
            
            # 将回退的flat价格数组重塑
            optimal_prices_array_fallback = np.array(initial_prices_flat_for_opt, dtype=float).reshape((1, self.n, self.l))

            print(f"优化失败，返回: optimal_prices={optimal_prices_array_fallback}, optimal_revenue={initial_total_revenue_on_failure}, initial_revenue={initial_total_revenue_on_failure}, demand_details={demand_details_list_on_failure}, daily_prices={daily_prices_list_on_failure}")
            return optimal_prices_array_fallback, initial_total_revenue_on_failure, initial_total_revenue_on_failure, demand_details_list_on_failure, daily_prices_list_on_failure

    def _format_daily_price_data(self, daily_prices_list):
        """格式化每日价格数据以供API返回"""
        cargo_types = ["快件", "鲜活", "普货"]
        self.daily_price_data = {}
        if not daily_prices_list: return # 如果列表为空则不处理
        num_days = len(daily_prices_list)

        for j in range(self.l):
            cargo_type = cargo_types[j] if j < len(cargo_types) else f"类型{j+1}" 
            prices_for_cargo = []
            for day_idx in range(num_days):
                try:
                    avg_price = np.mean(daily_prices_list[day_idx][:, :, j])
                    prices_for_cargo.append(round(float(avg_price), 2))
                except IndexError:
                     print(f"警告: 在格式化第 {day_idx+1} 天, 货物类型 {j} 的价格时发生索引错误。")
                     prices_for_cargo.append(0.0) # 添加默认值
            self.daily_price_data[cargo_type] = prices_for_cargo

    def _fix_array_size(self, input_array, expected_size, reference_array):
        """修正数组大小以匹配预期大小"""
        actual_size = len(input_array)
        if actual_size == expected_size:
            return input_array
        
        fixed_array = np.zeros(expected_size)
        if actual_size > expected_size:
            fixed_array = input_array[:expected_size]
        else: # actual_size < expected_size
            fixed_array[:actual_size] = input_array
            # 尝试用参考数组的平均值填充剩余部分
            if reference_array is not None and reference_array.size > 0:
                 fill_value = np.mean(reference_array)
            else:
                 fill_value = np.mean(input_array) if actual_size > 0 else 5.0 # 默认值
            fixed_array[actual_size:] = fill_value
        return fixed_array

    def _plot_daily_prices(self):
        """生成并保存每日价格图表"""
        if not MATPLOTLIB_AVAILABLE or not self.daily_price_data:
            print("警告: Matplotlib不可用或无每日价格数据，无法生成图表。")
            return

        try:
            plt.figure(figsize=(12, 6))
            days = list(range(1, self.booking_period + 1))
            
            for cargo_type, prices in self.daily_price_data.items():
                if len(prices) == len(days):
                    plt.plot(days, prices, marker='o', markersize=4, label=cargo_type)
                else:
                    print(f"警告: 货物类型 '{cargo_type}' 的价格数据长度 ({len(prices)}) 与天数 ({len(days)}) 不匹配，跳过绘图。")
            
            plt.title("预订期每日价格变化")
            plt.xlabel("预订天数")
            plt.ylabel("平均价格 (元/kg)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            filename = 'daily_booking_prices_elasticity.png'
            plt.savefig(filename)
            print(f"\n预订期每日价格变化图表已保存为 {filename}")
            plt.close()

        except Exception as e:
            print(f"绘制每日价格图表时出错: {str(e)}")
            traceback.print_exc()

    def _print_optimal_prices(self):
        """打印最优价格和收益结果"""
        cargo_types = ["快件", "鲜活", "普货"]
        
        print("\n最优价格结果:")
        print("-"*40)
        
        for k in range(self.m):
            print(f"\n航空公司{k+1}:")
            
            # 计算该航空公司的总收益
            airline_revenue = 0
            for i in range(self.n):
                for j in range(self.l):
                    # 计算价格弹性导致的需求变化
                    demand_ratio = self.calculate_demand_with_elasticity(
                        self.optimal_prices[k, i, j],
                        self.P_S_initial[k, i, j],
                        i, j
                    )
                    # 计算收益
                    revenue = self.optimal_prices[k, i, j] * demand_ratio * self.d_S[i, j]
                    airline_revenue += revenue
            
            # 创建价格表格
            price_data = []
            for j in range(self.l):
                cargo = cargo_types[j]
                old_price = np.mean(self.P_S_initial[k, :, j])
                new_price = np.mean(self.optimal_prices[k, :, j])
                change = (new_price - old_price) / old_price * 100
                
                price_data.append([
                    cargo, 
                    f"{old_price:.2f}元/kg", 
                    f"{new_price:.2f}元/kg", 
                    f"{change:+.2f}%"
                ])
            
            # 显示价格表格
            if PANDAS_AVAILABLE:
                price_df = pd.DataFrame(
                    price_data, 
                    columns=["货物类型", "初始价格", "最优价格", "变化百分比"]
                )
                print(price_df)
            else:
                print("货物类型  |  初始价格  |  最优价格  |  变化百分比")
                print("-" * 50)
                for row in price_data:
                    print(f"{row[0]}  |  {row[1]}  |  {row[2]}  |  {row[3]}")
            
            print(f"总收益: {airline_revenue:.2f}元")
    
    def analyze_elasticity_impact(self):
        """分析不同弹性系数对最优价格的影响"""
        cargo_types = ["快件", "鲜活", "普货"]
        
        print("\n价格弹性对最优定价的影响分析:")
        print("-"*50)
        
        for j in range(self.l):
            cargo = cargo_types[j]
            elasticity = np.mean(self.price_elasticity[:, j])
            
            # 计算该货物类型的平均最优价格
            optimal_prices = []
            for k in range(self.m):
                for i in range(self.n):
                    optimal_prices.append(self.optimal_prices[k, i, j])
            avg_optimal_price = np.mean(optimal_prices)
            
            # 计算初始价格
            initial_prices = []
            for k in range(self.m):
                for i in range(self.n):
                    initial_prices.append(self.P_S_initial[k, i, j])
            avg_initial_price = np.mean(initial_prices)
            
            price_change = (avg_optimal_price - avg_initial_price) / avg_initial_price * 100
            
            print(f"{cargo}:")
            print(f"  价格弹性系数: {elasticity:.2f}")
            print(f"  初始平均价格: {avg_initial_price:.2f}元/kg")
            print(f"  最优平均价格: {avg_optimal_price:.2f}元/kg")
            print(f"  价格变化: {price_change:+.2f}%")
            print(f"  分析: {'价格应下调' if price_change < 0 else '价格应上调'}, " +
                 f"弹性{abs(elasticity):.1f}{'较高' if abs(elasticity) > 1.5 else '中等' if abs(elasticity) > 1 else '较低'}")
            print()
    
    def create_simple_charts(self):
        """创建简单图表，避免使用plt.show()导致的问题"""
        if not MATPLOTLIB_AVAILABLE:
            print("警告: matplotlib未安装或初始化失败，无法生成可视化图表。")
            print("请安装matplotlib后再使用此功能（pip install matplotlib）。")
            return False
        
        try:
            print("开始生成图表...")
            cargo_types = ["快件", "鲜活", "普货"]
            output_dir = os.getcwd()
            chart_files = []
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            try:
                chinese_font = FontProperties(family='SimHei')
            except:
                chinese_font = None
                print("警告: 无法使用中文字体，图表标题可能显示为乱码")
            
            for j in range(self.l):
                cargo = cargo_types[j]
                elasticity = self.price_elasticity[0, j]
                
                # 文件名
                img_file = os.path.join(output_dir, f'elasticity_cargo_{j+1}_{cargo}.png')
                chart_files.append(img_file)
                
                print(f"生成{cargo}的弹性分析图表...")
                
                # 创建新图形
                plt.figure(figsize=(12, 5))
                
                # 生成价格变化范围
                base_price = self.P_S_initial[0, 0, j]
                price_ratios = np.linspace(0.7, 1.3, 100)
                
                # 计算需求和收益数据
                demand_ratios = []
                revenues = []
                for price in price_ratios:
                    price_change_percent = (price - 1) * 100
                    demand_change_percent = elasticity * price_change_percent
                    demand_ratio = max(self.min_demand_ratio, 1 + demand_change_percent / 100)
                    demand_ratios.append(demand_ratio)
                    
                    revenue = price * base_price * demand_ratio * self.d_S[0, j]
                    revenues.append(revenue)
                
                # 创建子图
                plt.subplot(1, 2, 1)
                plt.plot(price_ratios, demand_ratios, 'b-', linewidth=2)
                plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
                plt.grid(True, alpha=0.3)
                plt.title(f"{cargo}价格变化对需求的影响 (弹性系数: {elasticity:.2f})")
                plt.xlabel("价格比例 (相对于初始价格)")
                plt.ylabel("需求比例")
                
                # 收益曲线子图
                plt.subplot(1, 2, 2)
                plt.plot(price_ratios, revenues, 'g-', linewidth=2)
                
                # 标记最大收益点
                max_revenue_idx = np.argmax(revenues)
                max_revenue_price_ratio = price_ratios[max_revenue_idx]
                max_revenue = revenues[max_revenue_idx]
                
                plt.plot(max_revenue_price_ratio, max_revenue, 'ro', markersize=8)
                plt.annotate(f"最大收益点\n价格比例: {max_revenue_price_ratio:.2f}\n收益: {max_revenue:.0f}",
                             xy=(max_revenue_price_ratio, max_revenue),
                             xytext=(max_revenue_price_ratio+0.05, max_revenue-max_revenue*0.1),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
                
                plt.title(f"{cargo}价格变化对收益的影响")
                plt.xlabel("价格比例 (相对于初始价格)")
                plt.ylabel("收益")
                plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
                plt.grid(True, alpha=0.3)
                
                # 调整布局
                plt.tight_layout()
                
                # 保存图表
                plt.savefig(img_file, dpi=300, bbox_inches='tight')
                print(f"图表已保存到: {img_file}")
                
                # 关闭当前图形，避免内存泄漏
                plt.close()
            
            # 创建最终的组合图
            print("生成组合图表...")
            final_output = os.path.join(output_dir, 'elasticity_effects_on_demand_and_revenue.png')
            
            # 创建包含所有货物类型的大图
            plt.figure(figsize=(16, 6*self.l))
            
            for j in range(self.l):
                cargo = cargo_types[j]
                elasticity = self.price_elasticity[0, j]
                
                # 生成价格变化范围
                base_price = self.P_S_initial[0, 0, j]
                price_ratios = np.linspace(0.7, 1.3, 100)
                
                # 计算需求和收益数据
                demand_ratios = []
                revenues = []
                for price in price_ratios:
                    price_change_percent = (price - 1) * 100
                    demand_change_percent = elasticity * price_change_percent
                    demand_ratio = max(self.min_demand_ratio, 1 + demand_change_percent / 100)
                    demand_ratios.append(demand_ratio)
                    
                    revenue = price * base_price * demand_ratio * self.d_S[0, j]
                    revenues.append(revenue)
                
                # 需求曲线
                plt.subplot(self.l, 2, j*2+1)
                plt.plot(price_ratios, demand_ratios, 'b-', linewidth=2)
                plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
                plt.grid(True, alpha=0.3)
                plt.title(f"{cargo}价格变化对需求的影响 (弹性系数: {elasticity:.2f})")
                plt.xlabel("价格比例 (相对于初始价格)")
                plt.ylabel("需求比例")
                
                # 收益曲线
                plt.subplot(self.l, 2, j*2+2)
                plt.plot(price_ratios, revenues, 'g-', linewidth=2)
                
                # 标记最大收益点
                max_revenue_idx = np.argmax(revenues)
                max_revenue_price_ratio = price_ratios[max_revenue_idx]
                max_revenue = revenues[max_revenue_idx]
                
                plt.plot(max_revenue_price_ratio, max_revenue, 'ro', markersize=8)
                plt.annotate(f"最大收益点\n价格比例: {max_revenue_price_ratio:.2f}\n收益: {max_revenue:.0f}",
                            xy=(max_revenue_price_ratio, max_revenue),
                            xytext=(max_revenue_price_ratio+0.05, max_revenue-max_revenue*0.1),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
                
                plt.title(f"{cargo}价格变化对收益的影响")
                plt.xlabel("价格比例 (相对于初始价格)")
                plt.ylabel("收益")
                plt.axvline(x=1, color='r', linestyle='--', alpha=0.5)
                plt.grid(True, alpha=0.3)
            
            # 调整布局
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.tight_layout(pad=4.0)
            
            # 保存组合图
            plt.savefig(final_output, dpi=300, bbox_inches='tight')
            print(f"组合图表已保存到: {final_output}")
            
            # 关闭图形
            plt.close()
            
            print("所有图表生成完成！")
            return True
            
        except Exception as e:
            print(f"生成图表时出错: {str(e)}")
            traceback.print_exc()
            print("\n可能的解决方案:")
            print("1. 确保matplotlib已正确安装: pip install matplotlib")
            print("2. 检查是否有足够的磁盘空间保存图像")
            print("3. 尝试更新matplotlib: pip install --upgrade matplotlib")
            return False
            
    def visualize_elasticity_effects(self):
        """可视化价格弹性对需求和收益的影响"""
        if not MATPLOTLIB_AVAILABLE:
            print("警告: matplotlib未安装或初始化失败，无法生成可视化图表。")
            print("请安装matplotlib后再使用此功能（pip install matplotlib）。")
            return
            
        print("尝试使用简化图表生成方法...")
        success = self.create_simple_charts()
        
        if success:
            output_path = os.path.join(os.getcwd(), 'elasticity_effects_on_demand_and_revenue.png')
            print(f"请使用图像查看器打开此文件查看完整图表: {output_path}")
        else:
            print("图表生成失败，请查看错误信息")

    def get_route_config(self):
        """获取当前的航线配置"""
        return self.route_config

    def set_route_config(self, route_config):
        """设置新的航线配置并更新相关参数"""
        self.route_config = route_config
        # 重新初始化参数
        self._initialize_parameters_with_database_fallback()
        # 清除之前的优化结果
        self.optimal_prices = None
        self.max_revenue = 0

    def set_booking_period(self, period):
        """设置预订期长度并更新相关参数
        
        参数:
        period (int): 预订期长度（天数）
        """
        if period < 1:
            print("警告: 预订期长度必须大于0，将使用默认值14")
            period = 14
        
        self.booking_period = period
        # 更新预订期相关的参数
        self.booking_stages = min(5, self.booking_period // 3)
        self.booking_stage_names = ["早期", "中期", "晚期", "临近", "紧急"][:self.booking_stages]
        
        print(f"预订期已更新为{period}天，分为{self.booking_stages}个阶段")
        return self

# 使用示例
if __name__ == "__main__":
    print(f"Python版本: {sys.version}")
    print(f"NumPy版本: {np.__version__}")
    
    # 打印环境变量和工作目录信息
    try:
        print(f"当前工作目录: {os.getcwd()}")
        # 检查图像存储路径是否可写
        image_path = os.path.join(os.getcwd(), 'elasticity_effects_on_demand_and_revenue.png')
        print(f"图像将保存到: {image_path}")
    except Exception as e:
        print(f"获取系统信息时出错: {str(e)}")
    
    try:
        # 创建仅基于需求弹性的定价模型
        model = ElasticityBasedPricingModel(num_airlines=1, num_segments=2, num_cargo_types=3)
        
        # 运行价格优化
        print("计算基于需求弹性的最优价格...")
        model.optimize_prices()
        
        # 分析价格弹性对最优定价的影响
        model.analyze_elasticity_impact()
        
        # 可视化价格弹性对需求和收益的影响
        # 如果matplotlib不可用，此函数会显示警告但不会中断程序
        model.visualize_elasticity_effects()
        
        print("\n基于需求弹性的定价模型运行完成!")
        print("\n图表已保存为图像文件，请打开以下文件查看:")
        print(f"{image_path}")
    except Exception as e:
        print(f"程序运行中出现未处理的异常: {str(e)}")
        traceback.print_exc() 