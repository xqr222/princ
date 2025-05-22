import sqlite3
import json # Import json for handling JSON data

DATABASE_NAME = 'pricing_data.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create Routes Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            distance INTEGER,
            competition_level TEXT,
            popularity TEXT,
            season_factor REAL,
            flight_type TEXT,
            aircraft_id INTEGER,
            default_booking_period_days INTEGER,
            elasticity_initial_prices_json TEXT,
            elasticity_coefficients_json TEXT,
            elasticity_base_demands_json TEXT,
            dp_default_gamma REAL,
            dp_default_cd_ratio REAL,
            gt_k_value REAL,
            gt_demand_base_factor REAL,
            gt_price_sensitivity_factor REAL,
            gt_cross_price_sensitivity_factor REAL,
            gt_cap_util_sensitivity_factor REAL,
            FOREIGN KEY (aircraft_id) REFERENCES aircrafts(id)
        )
    ''')

    # Create Aircrafts Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS aircrafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL UNIQUE,
            max_payload INTEGER,
            max_volume INTEGER
        )
    ''')

    conn.commit()
    conn.close()
    print("Database tables created or already exist.")

def seed_initial_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    default_initial_prices = json.dumps({'快件': 12.0, '鲜活': 7.0, '普货': 6.0})
    default_coefficients = json.dumps({'快件': -1.7, '鲜活': -1.2, '普货': -1.5})
    default_base_demands = json.dumps({'快件': 60, '鲜活': 60, '普货': 60})

    # Default DP parameters
    default_booking_period = 14
    default_gamma = 6000.0
    default_cd_ratio = 0.9

    # Default Game Theory parameters
    default_gt_k_value = 2.0
    default_gt_demand_base_factor = 1.0
    default_gt_price_sensitivity_factor = 1.0
    default_gt_cross_price_sensitivity_factor = 1.0
    default_gt_cap_util_sensitivity_factor = 1.0

    # Seed aircrafts first to get A320 ID if needed
    cursor.execute("SELECT COUNT(*) FROM aircrafts")
    a320_id = None
    if cursor.fetchone()['COUNT(*)'] == 0:
        initial_aircrafts = [
            ('B737F', 20000, 135000000),
            ('B757F', 39000, 250000000),
            ('B777F', 102000, 650000000),
            ('A330F', 70000, 475000000),
            ('A320', 6964, 500000) # Ensure A320 is in this list
        ]
        cursor.executemany('''
            INSERT INTO aircrafts (type, max_payload, max_volume)
            VALUES (?, ?, ?)
        ''', initial_aircrafts)
        print(f"{len(initial_aircrafts)} initial aircrafts seeded.")
        
        cursor.execute("SELECT id FROM aircrafts WHERE type = ?", ('A320',))
        a320_id_row = cursor.fetchone()
        if a320_id_row:
            a320_id = a320_id_row['id']
            print(f"Retrieved A320 ID for initial route seeding: {a320_id}")
        else:
            print("Warning: A320 aircraft type not found after seeding. Will attempt to use first available aircraft ID.")
            cursor.execute("SELECT id FROM aircrafts LIMIT 1")
            fallback_id_row = cursor.fetchone()
            if fallback_id_row:
                a320_id = fallback_id_row['id']
                print(f"Using fallback aircraft ID for initial route seeding: {a320_id}")
            else:
                print("Warning: No aircrafts found to associate with routes during initial seeding.")
    else: # Aircrafts table already has data, try to get A320 ID
        cursor.execute("SELECT id FROM aircrafts WHERE type = ?", ('A320',))
        a320_id_row = cursor.fetchone()
        if a320_id_row:
            a320_id = a320_id_row['id']
        else: # A320 not found, use first available as fallback
            print("Warning: A320 aircraft type not found in existing data. Will attempt to use first available aircraft ID.")
            cursor.execute("SELECT id FROM aircrafts LIMIT 1")
            fallback_id_row = cursor.fetchone()
            if fallback_id_row:
                a320_id = fallback_id_row['id']
                print(f"Using fallback aircraft ID from existing aircrafts: {a320_id}")
            else:
                 print("Warning: No aircrafts found in existing data to associate with routes.")


    cursor.execute("SELECT COUNT(*) FROM routes")
    if cursor.fetchone()['COUNT(*)'] == 0:
        initial_routes = [
            ('大连', '广州', 2500, 'high', 'high', 1.0, '干线', a320_id, default_booking_period, default_initial_prices, default_coefficients, default_base_demands, default_gamma, default_cd_ratio, default_gt_k_value, default_gt_demand_base_factor, default_gt_price_sensitivity_factor, default_gt_cross_price_sensitivity_factor, default_gt_cap_util_sensitivity_factor),
            ('北京', '上海', 1200, 'high', 'high', 1.2, '干线', a320_id, default_booking_period, default_initial_prices, default_coefficients, default_base_demands, default_gamma, default_cd_ratio, default_gt_k_value, default_gt_demand_base_factor, default_gt_price_sensitivity_factor, default_gt_cross_price_sensitivity_factor, default_gt_cap_util_sensitivity_factor),
            ('广州', '成都', 1500, 'medium', 'medium', 1.0, '干线', a320_id, default_booking_period, default_initial_prices, default_coefficients, default_base_demands, default_gamma, default_cd_ratio, default_gt_k_value, default_gt_demand_base_factor, default_gt_price_sensitivity_factor, default_gt_cross_price_sensitivity_factor, default_gt_cap_util_sensitivity_factor),
            ('厦门', '昆明', 1800, 'low', 'low', 0.8, '支线', a320_id, default_booking_period, default_initial_prices, default_coefficients, default_base_demands, default_gamma, default_cd_ratio, default_gt_k_value, default_gt_demand_base_factor, default_gt_price_sensitivity_factor, default_gt_cross_price_sensitivity_factor, default_gt_cap_util_sensitivity_factor)
        ]
        cursor.executemany('''
            INSERT INTO routes (
                origin, destination, distance, competition_level, popularity, season_factor, flight_type, 
                aircraft_id, default_booking_period_days,
                elasticity_initial_prices_json, elasticity_coefficients_json, elasticity_base_demands_json,
                dp_default_gamma, dp_default_cd_ratio,
                gt_k_value, gt_demand_base_factor, gt_price_sensitivity_factor, 
                gt_cross_price_sensitivity_factor, gt_cap_util_sensitivity_factor
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', initial_routes)
        print(f"{len(initial_routes)} initial routes seeded.")

    conn.commit()
    conn.close()

# --- Routes CRUD --- 
def add_route(route_data: dict): # Expect a dictionary
    conn = get_db_connection()
    cursor = conn.cursor()
    # Ensure all keys are present, provide defaults for new JSON fields if missing
    route_data.setdefault('elasticity_initial_prices_json', json.dumps({'快件': 10.0, '鲜活': 6.0, '普货': 5.0}))
    route_data.setdefault('elasticity_coefficients_json', json.dumps({'快件': -1.5, '鲜活': -1.0, '普货': -1.2}))
    route_data.setdefault('elasticity_base_demands_json', json.dumps({'快件': 50, '鲜活': 70, '普货': 80}))
    # Add defaults for new DP and aircraft_id fields
    route_data.setdefault('aircraft_id', None) # Caller should provide a valid aircraft_id or handle None
    route_data.setdefault('default_booking_period_days', 14)
    route_data.setdefault('dp_default_gamma', 6000.0)
    route_data.setdefault('dp_default_cd_ratio', 0.9)
    # Add defaults for new Game Theory fields
    route_data.setdefault('gt_k_value', 2.0)
    route_data.setdefault('gt_demand_base_factor', 1.0)
    route_data.setdefault('gt_price_sensitivity_factor', 1.0)
    route_data.setdefault('gt_cross_price_sensitivity_factor', 1.0)
    route_data.setdefault('gt_cap_util_sensitivity_factor', 1.0)

    cursor.execute('''
        INSERT INTO routes (
            origin, destination, distance, competition_level, popularity, 
            season_factor, flight_type, 
            aircraft_id, default_booking_period_days,
            elasticity_initial_prices_json, elasticity_coefficients_json, elasticity_base_demands_json,
            dp_default_gamma, dp_default_cd_ratio,
            gt_k_value, gt_demand_base_factor, gt_price_sensitivity_factor, 
            gt_cross_price_sensitivity_factor, gt_cap_util_sensitivity_factor
        )
        VALUES (
            :origin, :destination, :distance, :competition_level, :popularity, 
            :season_factor, :flight_type,
            :aircraft_id, :default_booking_period_days,
            :elasticity_initial_prices_json, :elasticity_coefficients_json, :elasticity_base_demands_json,
            :dp_default_gamma, :dp_default_cd_ratio,
            :gt_k_value, :gt_demand_base_factor, :gt_price_sensitivity_factor, 
            :gt_cross_price_sensitivity_factor, :gt_cap_util_sensitivity_factor
        )
    ''', route_data)
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return get_route_by_id(new_id)

def get_all_routes():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM routes")
    routes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return routes

def get_route_by_id(route_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM routes WHERE id = ?", (route_id,))
    row = cursor.fetchone() # fetchone() returns a single row or None
    conn.close()
    return dict(row) if row else None # Convert to dict only if row is not None

def update_route(route_id, route_data: dict): # Expect a dictionary
    conn = get_db_connection()
    cursor = conn.cursor()
    
    set_clauses = []
    params = {'id': route_id}
    
    valid_keys = [
        'origin', 'destination', 'distance', 'competition_level', 
        'popularity', 'season_factor', 'flight_type', 
        'aircraft_id', 'default_booking_period_days',
        'elasticity_initial_prices_json', 'elasticity_coefficients_json', 
        'elasticity_base_demands_json',
        'dp_default_gamma', 'dp_default_cd_ratio',
        'gt_k_value', 'gt_demand_base_factor', 'gt_price_sensitivity_factor',
        'gt_cross_price_sensitivity_factor', 'gt_cap_util_sensitivity_factor'
    ]

    for key, value in route_data.items():
        if key in valid_keys:
            set_clauses.append(f"{key} = :{key}")
            params[key] = value
            
    if not set_clauses:
        conn.close()
        return get_route_by_id(route_id) # Or raise an error: no valid fields to update

    sql = f"UPDATE routes SET {', '.join(set_clauses)} WHERE id = :id"
    
    cursor.execute(sql, params)
    conn.commit()
    conn.close()
    return get_route_by_id(route_id)

def delete_route(route_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM routes WHERE id = ?", (route_id,))
    conn.commit()
    deleted_count = cursor.rowcount
    conn.close()
    return deleted_count > 0

# --- New function to get elasticity parameters ---
def get_route_elasticity_params(route_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            elasticity_initial_prices_json, 
            elasticity_coefficients_json, 
            elasticity_base_demands_json 
        FROM routes 
        WHERE id = ?
    ''', (route_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        try:
            initial_prices = json.loads(row['elasticity_initial_prices_json']) if row['elasticity_initial_prices_json'] else None
            coefficients = json.loads(row['elasticity_coefficients_json']) if row['elasticity_coefficients_json'] else None
            base_demands = json.loads(row['elasticity_base_demands_json']) if row['elasticity_base_demands_json'] else None
            
            # Basic validation: check if they are dictionaries (or None)
            if initial_prices is not None and not isinstance(initial_prices, dict): initial_prices = None
            if coefficients is not None and not isinstance(coefficients, dict): coefficients = None
            if base_demands is not None and not isinstance(base_demands, dict): base_demands = None

            return {
                "initial_prices": initial_prices,
                "coefficients": coefficients,
                "base_demands": base_demands
            }
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for route_id {route_id}: {e}")
            return {"initial_prices": None, "coefficients": None, "base_demands": None} # Return defaults on error
    return {"initial_prices": None, "coefficients": None, "base_demands": None} # Route not found or no params

# --- New function to get DP parameters ---
def get_route_dp_params(route_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Initialize with None to indicate data not found or not applicable
    dp_params = {
        "aircraft_max_payload": None,
        "aircraft_max_volume": None,
        "default_booking_period_days": None,
        "dp_default_gamma": None,
        "dp_default_cd_ratio": None
    }
    
    cursor.execute('''
        SELECT 
            r.aircraft_id, 
            r.default_booking_period_days, 
            r.dp_default_gamma, 
            r.dp_default_cd_ratio,
            a.max_payload AS aircraft_max_payload,
            a.max_volume AS aircraft_max_volume
        FROM routes r
        LEFT JOIN aircrafts a ON r.aircraft_id = a.id
        WHERE r.id = ?
    ''', (route_id,))
    row = cursor.fetchone()
    
    if row:
        # Only populate if values are not None from DB to distinguish from "not set"
        if row["aircraft_max_payload"] is not None:
            dp_params["aircraft_max_payload"] = row["aircraft_max_payload"]
        if row["aircraft_max_volume"] is not None:
            dp_params["aircraft_max_volume"] = row["aircraft_max_volume"]
        if row["default_booking_period_days"] is not None:
            dp_params["default_booking_period_days"] = row["default_booking_period_days"]
        if row["dp_default_gamma"] is not None:
            dp_params["dp_default_gamma"] = row["dp_default_gamma"]
        if row["dp_default_cd_ratio"] is not None:
            dp_params["dp_default_cd_ratio"] = row["dp_default_cd_ratio"]
            
    conn.close()
    return dp_params

# --- New function to get Game Theory parameters ---
def get_route_gametheory_params(route_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Initialize with None to indicate data not found or not applicable
    gt_params = {
        "gt_k_value": None,
        "gt_demand_base_factor": None,
        "gt_price_sensitivity_factor": None,
        "gt_cross_price_sensitivity_factor": None,
        "gt_cap_util_sensitivity_factor": None,
        "default_booking_period_days": None # Also fetch this for T_periods consistency
    }
    
    cursor.execute('''
        SELECT 
            r.gt_k_value, 
            r.gt_demand_base_factor, 
            r.gt_price_sensitivity_factor, 
            r.gt_cross_price_sensitivity_factor,
            r.gt_cap_util_sensitivity_factor,
            r.default_booking_period_days 
        FROM routes r
        WHERE r.id = ?
    ''', (route_id,))
    row = cursor.fetchone()
    
    if row:
        # Populate if values are not None from DB
        if row["gt_k_value"] is not None:
            gt_params["gt_k_value"] = row["gt_k_value"]
        if row["gt_demand_base_factor"] is not None:
            gt_params["gt_demand_base_factor"] = row["gt_demand_base_factor"]
        if row["gt_price_sensitivity_factor"] is not None:
            gt_params["gt_price_sensitivity_factor"] = row["gt_price_sensitivity_factor"]
        if row["gt_cross_price_sensitivity_factor"] is not None:
            gt_params["gt_cross_price_sensitivity_factor"] = row["gt_cross_price_sensitivity_factor"]
        if row["gt_cap_util_sensitivity_factor"] is not None:
            gt_params["gt_cap_util_sensitivity_factor"] = row["gt_cap_util_sensitivity_factor"]
        if row["default_booking_period_days"] is not None: # For T_periods
            gt_params["default_booking_period_days"] = row["default_booking_period_days"]
            
    conn.close()
    return gt_params

# --- Aircrafts CRUD --- 
def add_aircraft(aircraft_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO aircrafts (type, max_payload, max_volume)
            VALUES (:type, :max_payload, :max_volume)
        ''', aircraft_data)
        conn.commit()
        new_id = cursor.lastrowid
        return get_aircraft_by_id(new_id)
    except sqlite3.IntegrityError: # For UNIQUE constraint on type
        return None # Or raise a custom error
    finally:
        conn.close()

def get_all_aircrafts():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM aircrafts")
    aircrafts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return aircrafts

def get_aircraft_by_id(aircraft_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM aircrafts WHERE id = ?", (aircraft_id,))
    aircraft = dict(cursor.fetchone()) if cursor.fetchone() else None
    conn.close()
    return aircraft

def update_aircraft(aircraft_id, aircraft_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE aircrafts SET 
                type = :type, max_payload = :max_payload, max_volume = :max_volume
            WHERE id = :id
        ''', {**aircraft_data, 'id': aircraft_id})
        conn.commit()
        return get_aircraft_by_id(aircraft_id)
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def delete_aircraft(aircraft_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM aircrafts WHERE id = ?", (aircraft_id,))
    conn.commit()
    deleted_count = cursor.rowcount
    conn.close()
    return deleted_count > 0

if __name__ == '__main__':
    print("Initializing database...")
    create_tables()
    seed_initial_data()
    print("--- Sample Data ---")
    print("All Routes:", get_all_routes())
    print("All Aircrafts:", get_all_aircrafts())
    print("\\n--- Sample Elasticity Params for Route 1 ---")
    sample_params = get_route_elasticity_params(1)
    if sample_params and (sample_params.get('initial_prices') or sample_params.get('coefficients') or sample_params.get('base_demands')):
        print(f"Route 1 Initial Prices: {sample_params.get('initial_prices')}")
        print(f"Route 1 Coefficients: {sample_params.get('coefficients')}")
        print(f"Route 1 Base Demands: {sample_params.get('base_demands')}")
    else:
        print("No elasticity parameters found for Route 1 or route does not exist.") 