
# Imports required
import pandas as pd
from src.database import get_engine,Database_creation
from src.etl import extract_and_clean_data,engineer_features
from src.model import train_test_model

def main():
    #database creation
    engine = get_engine()
    Database_creation(engine)
    #extraction of the dataset
    df_raw = extract_and_clean_data('archive/train.csv')
    df = engineer_features(df_raw)
    #model training and testing based on the given Dataframe
    x_test,y_test,predictions = train_test_model(df)

    #new dataframe since predicitons values come from 20% of the original data given
    df_clean = pd.DataFrame(index=x_test.index)
    #reattaching the dropped columns used for the new SQL schema table
    df_clean['pickup_date'] = df.loc[x_test.index,'pickup_datetime']
    df_clean['trip_key'] = df.loc[x_test.index,'key']
    df_clean['actual_fare'] = y_test
    #populating the rest of the values in the table
    df_clean['predicted_fare'] = predictions
    df_clean['predicted_fare_error'] = (df_clean['actual_fare'] - df_clean['predicted_fare']).abs()
    df_clean['total_distance'] = x_test['total_distance']
    df_clean['passenger_count'] = x_test['passenger_count']
    df_clean['from_to_jfk_airport'] = x_test['from_to_JFK_Airport']
    df_clean['from_to_laguardia_airport'] = x_test['from_to_LaGuardia_Airport']
    df_clean['is_manhattan_centric_traffic'] = x_test['is_manhattan_centric_traffic']
    df_clean['is_rush_hour'] = x_test['is_rush_hour']
    df_clean['is_night_trip'] = x_test['is_night_trip']

    #df_clean holds the results of the predicitons as well as other relevant data

    #all the information from df_clean is piped to the PostgreSQL Schema table
    df_clean.to_sql(
        name='ny_taxi_prediction_fare_results',
        con=engine,
        if_exists='append',
        index=False
    )
    print("Pipeline execution completed successfully.")
if __name__ == "__main__":
    main()