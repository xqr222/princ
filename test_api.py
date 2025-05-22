import requests
import json

def test_health():
    """测试API健康状态"""
    response = requests.get("http://localhost:8000/api/health")
    print("\n健康检查结果:")
    try:
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def test_elasticity_model():
    """测试弹性定价模型"""
    url = "http://localhost:8000/api/elasticity/calculate"
    data = {
        "origin": "PVG",
        "destination": "LAX",
        "distance": 10000,
        "competition_level": "medium",
        "popularity": "high",
        "aircraft_type": "B777F",
        "cargo_types": {"express": True, "fresh": True, "regular": True},
        "price_elasticity": {"express": -1.7, "fresh": -1.2, "regular": -1.5}
    }
    
    print("\n弹性定价模型测试:")
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\n价格表:")
            for item in result.get("priceTable", []):
                print(f"  {item['cargoType']}: 初始价格 {item['initialPrice']:.2f} -> 最优价格 {item['optimalPrice']:.2f} (变化: {item['change']*100:.1f}%)")
            
            print(f"\n初始收益: {result.get('initialRevenue', 0):.2f}")
            print(f"最优收益: {result.get('optimalRevenue', 0):.2f}")
            print(f"收益增长: {result.get('revenueIncrease', 0):.1f}%")
        else:
            print(f"响应错误: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

def test_gametheory_model():
    """测试博弈论定价模型"""
    url = "http://localhost:8000/api/gametheory/calculate"
    data = {
        "origin": "PVG",
        "destination": "LAX",
        "distance": 10000,
        "competition_level": "medium",
        "popularity": "high",
        "aircraft_type": "B777F",
        "companies": [
            {"name": "公司1", "capacity": 80000, "initialPrice": 5.8},
            {"name": "公司2", "capacity": 65000, "initialPrice": 7.0}
        ],
        "time_periods": 11
    }
    
    print("\n博弈论定价模型测试:")
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\n公司收益:")
            for company in result.get("companies", []):
                print(f"  {company['name']}: 初始收益 {company['initialRevenue']:.2f} -> 最优收益 {company['optimalRevenue']:.2f}")
            
            print(f"\n总初始收益: {result.get('initialRevenue', 0):.2f}")
            print(f"总最优收益: {result.get('optimalRevenue', 0):.2f}")
            print(f"收益增长: {result.get('revenueIncrease', 0):.1f}%")
            
            print("\n价格表 (部分):")
            for period in result.get("priceTable", [])[:3]:
                print(f"  时段{period['period']}: 公司1价格 {period['price1']:.2f}, 公司2价格 {period['price2']:.2f}")
        else:
            print(f"响应错误: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")
        
def test_dynamic_model():
    """测试动态规划定价模型"""
    url = "http://localhost:8000/api/dynamicdp/calculate"
    data = {
        "origin": "PVG",
        "destination": "LAX",
        "distance": 10000,
        "competition_level": "medium",
        "popularity": "high",
        "aircraft_type": "B777F",
        "booking_period": 14,
        "gamma": 6000,
        "C_D_ratio": 0.9
    }
    
    print("\n动态规划定价模型测试:")
    try:
        response = requests.post(url, json=data)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n基础收益: {result.get('baseRevenue', 0):.2f}")
            print(f"WVS优化收益: {result.get('wvsRevenue', 0):.2f}")
            print(f"收益增长: {result.get('improvement', 0):.1f}%")
            print(f"重量利用率: {result.get('weightUtilization', 0)}%")
            print(f"体积利用率: {result.get('volumeUtilization', 0)}%")
            
            print("\n不同货物类型的价格表 (部分):")
            for item in result.get("bookingDaysPriceTable", []):
                print(f"  {item['bookingType']}:")
                for day_price in item['dailyPrices'][:3]:  # 只显示前3天
                    print(f"    第{day_price['day']}天: {day_price['price']:.2f}")
                print("    ...")
        else:
            print(f"响应错误: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    # 测试API健康状态
    test_health()
    
    # 测试弹性定价模型
    test_elasticity_model()
    
    # 测试博弈论定价模型
    test_gametheory_model()
    
    # 测试动态规划定价模型
    test_dynamic_model() 