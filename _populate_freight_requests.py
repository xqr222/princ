'''
Script to GENERATE simulated freight request data and print it.
Database interaction is removed for this step.
'''
import random
import datetime
import sqlite3 # 添加 sqlite3 导入
import traceback # 用于更详细的错误输出

# --- CONFIGURATION ---
NUM_REQUESTS_PER_ROUTE = 20
TIMESTAMP_LOOKBACK_DAYS = 7
# Assume route_ids would be fetched or predefined. For generation, we can use a sample.
PREDEFINED_ROUTE_IDS = [1, 2, 3, 4] # Matching your previous logs

CARGO_TYPE_DISTRIBUTION = {
    '快件': 0.2,
    '鲜活': 0.3,
    '普货': 0.5
}

CARGO_SPECS = {
    '快件': {
        'weight_range_kg': (1, 50),
        'density_range_cm3_kg': (5000, 9000)
    },
    '鲜活': {
        'weight_range_kg': (20, 200),
        'density_range_cm3_kg': (4000, 7000)
    },
    '普货': {
        'weight_range_kg': (50, 1000),
        'density_range_cm3_kg': (3000, 8000)
    }
}
# --- END CONFIGURATION ---

# 数据库文件名 (与 database.py 和 _insert_generated_data.py 一致)
DATABASE_NAME = 'pricing_data.db'

def ensure_freight_requests_table_exists(cursor):
    """确保 freight_requests 表存在，如果不存在则创建。不会清空数据。"""
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS freight_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        route_id INTEGER NOT NULL,
        cargo_type TEXT NOT NULL,
        weight_kg REAL NOT NULL,
        volume_cm3 REAL NOT NULL,
        FOREIGN KEY (route_id) REFERENCES routes (id)
    );
    ''')
    print("INFO: `freight_requests` table checked/created (if not exists).")

def get_random_cargo_type():
    rand_val = random.random()
    cumulative_prob = 0
    for cargo_type, prob in CARGO_TYPE_DISTRIBUTION.items():
        cumulative_prob += prob
        if rand_val <= cumulative_prob:
            return cargo_type
    return list(CARGO_TYPE_DISTRIBUTION.keys())[-1]

def generate_and_insert_requests(num_requests_per_route_override=None, lookback_days_override=None):
    """生成货运请求并将其插入数据库。"""
    print("INFO: Starting data generation and insertion...")
    
    # 允许通过参数覆盖配置，方便外部调用
    num_to_generate = num_requests_per_route_override if num_requests_per_route_override is not None else NUM_REQUESTS_PER_ROUTE
    lookback_days = lookback_days_override if lookback_days_override is not None else TIMESTAMP_LOOKBACK_DAYS
    
    all_generated_requests = []
    route_ids = PREDEFINED_ROUTE_IDS

    print(f"INFO: Will generate data for route_ids: {route_ids}")
    print(f"INFO: Generating {num_to_generate} requests per route.")

    for route_id in route_ids:
        for i in range(num_to_generate):
            cargo_type = get_random_cargo_type()
            specs = CARGO_SPECS[cargo_type]
            weight_kg = round(random.uniform(specs['weight_range_kg'][0], specs['weight_range_kg'][1]), 2)
            density = random.uniform(specs['density_range_cm3_kg'][0], specs['density_range_cm3_kg'][1])
            volume_cm3 = round(weight_kg * density, 2)
            
            days_ago = random.uniform(0, lookback_days)
            delta = datetime.timedelta(days=days_ago, 
                                     hours=random.uniform(0,24),
                                     minutes=random.uniform(0,60),
                                     seconds=random.uniform(0,60))
            request_time = datetime.datetime.now() - delta
            request_timestamp_str = request_time.strftime('%Y-%m-%d %H:%M:%S')
            
            request_tuple = (request_timestamp_str, route_id, cargo_type, weight_kg, volume_cm3)
            all_generated_requests.append(request_tuple)

    print(f"INFO: Data generation complete. Total requests generated: {len(all_generated_requests)}")

    if not all_generated_requests:
        print("INFO: No requests generated, skipping database insertion.")
        return

    conn = None
    inserted_count = 0
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        print(f"INFO: Database '{DATABASE_NAME}' connected.")

        ensure_freight_requests_table_exists(cursor) # 确保表存在

        print(f"INFO: Attempting to insert {len(all_generated_requests)} generated requests into 'freight_requests' table...")
        for req_data_tuple in all_generated_requests:
            try:
                cursor.execute('''
                INSERT INTO freight_requests 
                    (request_timestamp, route_id, cargo_type, weight_kg, volume_cm3)
                VALUES (?, ?, ?, ?, ?);
                ''', req_data_tuple)
                inserted_count += 1
            except sqlite3.Error as item_error:
                print(f"ERROR: Could not insert item {req_data_tuple}: {item_error}")
        
        conn.commit()
        print(f"SUCCESS: Committed {inserted_count}/{len(all_generated_requests)} requests to the database.")

    except sqlite3.Error as e:
        print(f"ERROR: SQLite Database error: {e}")
        print(traceback.format_exc())
        if conn:
            conn.rollback()
            print("INFO: Transaction rolled back.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        print(traceback.format_exc())
        if conn:
            conn.rollback()
            print("INFO: Transaction rolled back.")
    finally:
        if conn:
            conn.close()
            print(f"INFO: Database connection to '{DATABASE_NAME}' closed.")
        print("INFO: Data generation and insertion script finished.")
    
    return inserted_count # 返回实际插入的数量

if __name__ == '__main__':
    # generate_request_data() # 旧的调用方式
    generate_and_insert_requests() 