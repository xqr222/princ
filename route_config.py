from aircraft_config import AircraftConfig
import math

class RouteConfig:
    """航线配置类，用于统一管理航线参数"""
    
    def __init__(self, route_info, aircraft_type=None, aircraft_config=None):
        """
        初始化航线配置
        
        参数:
        route_info: 航线信息字典，包含始发地、目的地等信息
        aircraft_type: 指定机型，如'B777F'
        aircraft_config: 直接传入的机型配置实例
        """
        # 基本航线信息
        self.route_info = {
            'origin': route_info.get('origin', '未知'),
            'destination': route_info.get('destination', '未知'),
            'distance': route_info.get('distance', 0),
            'flight_type': route_info.get('flight_type', '干线'),
            'competition_level': route_info.get('competition_level', 'medium'),
            'popularity': route_info.get('popularity', 'medium'),
            'market_share': route_info.get('market_share', 0.5),
            'flight_frequency': route_info.get('flight_frequency', 7),
            'season_factor': route_info.get('season_factor', 1.0)
        }
        
        # 设置机型配置
        self.aircraft = None
        if aircraft_config:
            # 优先使用传入的机型配置
            self.aircraft = aircraft_config
        elif aircraft_type:
            # 如果指定了机型，创建机型配置
            self.aircraft = AircraftConfig(aircraft_type)
            
        # 计算航线特征
        self._calculate_route_characteristics()
    
    def _calculate_route_characteristics(self):
        """计算航线特征参数"""
        # 基于距离的特征
        distance = self.route_info['distance']
        if distance <= 800:
            self.route_info['distance_category'] = 'short'
        elif distance <= 2000:
            self.route_info['distance_category'] = 'medium'
        else:
            self.route_info['distance_category'] = 'long'
            
        # 基于航班频率的市场规模
        freq = self.route_info['flight_frequency']
        if freq <= 3:
            self.route_info['market_size'] = 'small'
        elif freq <= 7:
            self.route_info['market_size'] = 'medium'
        else:
            self.route_info['market_size'] = 'large'
            
        # 计算基础需求系数
        self.route_info['demand_coefficient'] = (
            1.0 +
            (0.2 if self.route_info['popularity'] == 'high' else
             -0.2 if self.route_info['popularity'] == 'low' else 0) +
            (0.1 if self.route_info['market_size'] == 'large' else
             -0.1 if self.route_info['market_size'] == 'small' else 0)
        )
    
    def set_aircraft(self, aircraft_type_or_config):
        """设置或更改机型"""
        if isinstance(aircraft_type_or_config, AircraftConfig):
            self.aircraft = aircraft_type_or_config
        else:
            self.aircraft = AircraftConfig(aircraft_type_or_config)
        return True
    
    def calculate_operating_cost(self):
        """计算航线运营成本(简化版)"""
        if not self.aircraft:
            raise ValueError("未设置机型")
            
        # 使用简化的成本计算方法
        # 基于载重量的基础成本估算
        capacity = self.aircraft.get_capacity()
        distance = self.route_info['distance']
        
        # 使用简单成本模型：载重量×距离×基础成本因子
        base_cost_factor = 0.0005  # 简化的成本因子
        base_cost = capacity['max_payload'] * distance * base_cost_factor
        
        # 根据航线特征调整成本
        cost_factors = {
            'competition_level': {
                'high': 1.1,    # 高竞争环境成本增加
                'medium': 1.0,
                'low': 0.9      # 低竞争环境成本降低
            },
            'popularity': {
                'high': 1.05,   # 热门航线维护成本略高
                'medium': 1.0,
                'low': 0.95
            },
            'flight_type': {
                '干线': 1.0,
                '支线': 1.1     # 支线航线单位成本较高
            }
        }
        
        # 应用成本调整因子
        adjusted_cost = base_cost * (
            cost_factors['competition_level'][self.route_info['competition_level']] *
            cost_factors['popularity'][self.route_info['popularity']] *
            cost_factors['flight_type'][self.route_info['flight_type']]
        )
        
        return adjusted_cost
    
    def estimate_demand(self, price_level='medium'):
        """估算航线需求
        
        参数:
        price_level: 价格水平('low', 'medium', 'high')
        
        返回:
        预计需求(kg)
        """
        if not self.aircraft:
            raise ValueError("未设置机型")
            
        # 基础需求（以载重的70%为基准）
        base_demand = self.aircraft.get_capacity()['max_payload'] * 0.7
        
        # 价格弹性调整
        price_factors = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.8
        }
        
        # 应用需求系数和价格因子
        adjusted_demand = (base_demand * 
                         self.route_info['demand_coefficient'] * 
                         price_factors[price_level])
        
        return adjusted_demand
    
    def get_route_info(self):
        """获取完整的航线信息"""
        info = self.route_info.copy()
        if self.aircraft:
            info['aircraft'] = {
                'type': self.aircraft.aircraft_type,
                'capacity': self.aircraft.get_capacity()
            }
        return info
    
    def to_dict(self):
        """返回完整的配置字典"""
        return {
            'route': self.route_info,
            'aircraft': self.aircraft.to_dict() if self.aircraft else None
        }

# 使用示例
if __name__ == "__main__":
    # 创建航线配置
    route_info = {
        'origin': '北京',
        'destination': '上海',
        'distance': 1200,
        'flight_type': '干线',
        'competition_level': 'high',
        'popularity': 'high',
        'flight_frequency': 14
    }
    
    # 方法1：通过机型字符串创建
    route1 = RouteConfig(route_info, 'B737F')
    
    # 方法2：通过机型配置实例创建
    aircraft_config = AircraftConfig('B777F')
    aircraft_config.set_payload(100000).set_volume(640000000)  # 自定义设置参数
    route2 = RouteConfig(route_info, aircraft_type=None, aircraft_config=aircraft_config)
    
    # 比较两种方式的容量差异
    print(f"方法1 - B737F容量: {route1.aircraft.get_capacity()}")
    print(f"方法2 - 自定义B777F容量: {route2.aircraft.get_capacity()}") 