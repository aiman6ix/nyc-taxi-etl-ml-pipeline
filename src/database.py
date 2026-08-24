from sqlalchemy import create_engine, text

# SQLAlchemy engine created to establish a connection between the Pandas Dataframe and PostgreSQL table 
def get_engine():
    DB_user = "postgres"
    DB_pass = "postgres"
    DB_host = "localhost"
    DB_port = "5432"
    DB_name = "taxi_pipeline"

    return create_engine(f"postgresql://{DB_user}:{DB_pass}@{DB_host}:{DB_port}/{DB_name}")
   

def Database_creation(engine):
#SQL table created to accept all the information from the pandas Dataframe
    create_sql_table = """
        CREATE TABLE IF NOT EXISTS ny_taxi_prediction_fare_results (
        prediction_id serial PRIMARY KEY,

        --identifiers of each trip/row
        trip_key VARCHAR(50) NOT NULL,
        pickup_date TIMESTAMPTZ NOT NULL,

        --Accuracy of the predicitons
        predicted_fare NUMERIC(6,2) NOT NULL,
        actual_fare NUMERIC(6,2),
        predicted_fare_error NUMERIC(6,2),

        --Additional prediction information
        total_distance NUMERIC(6,2),
        passenger_count SMALLINT,
    
        -- Flag Featture
        from_to_jfk_airport SMALLINT,
        from_to_laguardia_airport SMALLINT,
        is_manhattan_centric_traffic SMALLINT,
        is_rush_hour SMALLINT,
        is_night_trip SMALLINT,


        --time the prediction was done and prediction model used
        model_version VARCHAR(20) DEFAULT 'lgbm_v1.0',
        processed TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
   
    )
    """

    with engine.connect() as connection:
        connection.execute(text(create_sql_table))
        connection.commit()
    print('table schema created')