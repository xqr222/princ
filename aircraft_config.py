class AircraftConfig:
    """机型配置类，用于统一管理机型参数"""
    
    def __init__(self, aircraft_type, config=None):
        self.aircraft_type = aircraft_type
        
        # 默认配置 - 仅保留机型、最大载重和最大体积
        self.default_config = {
            # 波音系列
            'B737F': {
                'max_payload': 20000,      # 最大商载(kg)
                'max_volume': 135000000    # 最大容积(cm³)
            },
            'B757F': {
                'max_payload': 39000,
                'max_volume': 250000000
            },
            'B777F': {
                'max_payload': 102000,
                'max_volume': 650000000
            },
            # 空客系列
            'A330F': {
                'max_payload': 70000,
                'max_volume': 475000000
            },
            'A320': {
                'max_payload': 6964,        # 最大商载(kg)
                'max_volume': 500000        # 最大容积(cm³)，500立方分米 = 500,000立方厘米
            }
        }
        
        # 使用提供的配置或默认配置
        self.config = config if config else self.default_config.get(aircraft_type)
        if not self.config:
            raise ValueError(f"未知的机型: {aircraft_type}")
        
        # 设置属性便于直接访问
        self.payload = self.config['max_payload']
        self.volume = self.config['max_volume']
    
    def set_payload(self, value):
        """设置最大载重"""
        self.payload = value
        self.config['max_payload'] = value
        return self
    
    def set_volume(self, value):
        """设置最大体积"""
        self.volume = value
        self.config['max_volume'] = value
        return self
    
    def get_capacity(self):
        """获取机型的容量信息"""
        return {
            'max_payload': self.payload,
            'max_volume': self.volume
        }
    
    def to_dict(self):
        """返回完整的配置字典"""
        return self.config.copy()

# 使用示例
if __name__ == "__main__":
    # 创建B777F的配置
    b777f = AircraftConfig('B777F')
    
    # 获取容量信息
    capacity = b777f.get_capacity()
    print(f"B777F容量: {capacity['max_payload']}kg, {capacity['max_volume']}cm³")
    
    # 自定义设置参数
    b777f.set_payload(100000).set_volume(640000000)
    capacity = b777f.get_capacity()
    print(f"自定义B777F容量: {capacity['max_payload']}kg, {capacity['max_volume']}cm³") 