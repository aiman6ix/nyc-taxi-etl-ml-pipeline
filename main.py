
# Imports required
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb      #prediction model used
from sqlalchemy import create_engine, text

# SQLAlchemy engine created to establish a connection between the Pandas Dataframe and PostgreSQL table 

DB_user = "postgres"
DB_pass = "postgres"
DB_host = "localhost"
DB_port = "5432"
DB_name = "taxi_pipeline"

engine = create_engine(f"postgresql://{DB_user}:{DB_pass}@{DB_host}:{DB_port}/{DB_name}")
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


    --  time the prediction was done and prediction model used
    model_version VARCHAR(20) DEFAULT 'lgbm_v1.0',
    processed TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
   
)
"""

with engine.connect() as connection:
    connection.execute(text(create_sql_table))
    connection.commit()
print('table schema created')

data =  pd.read_csv('archive/train.csv')

df = pd.DataFrame(data)
#kepts only rows without any 0 
df = df.loc[
(df['pickup_latitude']>=40.5) & (df['pickup_latitude'] <=40.9) &  #NY latitude stretches from 40.5 to 40.9
(df['dropoff_latitude'] >=40.5) & (df['dropoff_latitude'] <=40.9) &  
(df['pickup_longitude']<= -73.7) & (df['pickup_longitude'] >= -74.3) & #NY longitude strethces from 73.7 to 74.3
(df['dropoff_longitude'] <= -73.7) & (df['dropoff_longitude'] >=-74.3) &
(df['passenger_count']>= 1) &  #always needs at least 1 passenger
(df['fare_amount'].between(1,100))     #no free rides
]


def calculate_haversine(df):
    """haversine formula is used to find the distance in Km between 2 points (lon/lat) using earths Radius"""
    #Turn them into radians from degrees
    lon1,lon2 = np.radians(df['pickup_longitude']),np.radians(df['dropoff_longitude'])
    lat1,lat2 = np.radians( df['pickup_latitude']),np.radians(df['dropoff_latitude'])
    #final lat/long - initial lat/long
    lon_diff = lon2-lon1
    lat_diff = lat2-lat1
    
    a = np.sin(lat_diff/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(lon_diff/2)**2
    c = 2*np.arctan(np.sqrt(a),np.sqrt(1-a))
    r = 6371    #earths radius in km

    #final distance
    d = r*c
    return d
#took apart pickup_datetime into multiple parts such as year,month etc
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['Year'] = df['pickup_datetime'].dt.year
df['Month'] = df['pickup_datetime'].dt.month
df['Dayofweek'] = df['pickup_datetime'].dt.dayofweek
df['Hour'] = df['pickup_datetime'].dt.hour
#using the haversine formula to calculate the total distance traveled
df['total_distance'] = calculate_haversine(df)

#Calculating manhattan distance (how many blocks the taxi drove)

df['abs_lat_diff'] = (df['pickup_latitude'] - df['dropoff_latitude']).abs()
df['abs_lon_diff'] = (df['pickup_longitude'] - df['dropoff_longitude']).abs()
#convert 1 latitude to 111 km and 1 longitude is 84 km
df['manhattan_distance'] = (df['abs_lat_diff'] *111) + (df['abs_lon_diff']*84)


#column for people coming and going to JFK airport since no matter the distance, a flat rate is charged. 
df['from_to_JFK_Airport'] = 0
df.loc[
       (((df['dropoff_latitude'].between(40.6199,40.6599))) &
       (df['dropoff_longitude'].between(-73.7987,-73.7587))) |

       (((df['pickup_latitude'].between(40.6199,40.6599)))  &
       (df['pickup_longitude'].between(-73.7987,-73.7587)))  ,'from_to_JFK_Airport'
] = 1

#Trips to and from LaGuardia Airport have a $ 5.00 supercharge 
df['from_to_LaGuardia_Airport'] = 0
df.loc[
 ((df['dropoff_latitude'].between(40.752,40.792)) &
 (df['dropoff_longitude'].between(-73.8926,-73.8526))) |

 ((df['pickup_latitude'].between(40.752,40.792)) &
 (df['pickup_longitude'].between(-73.8926,-73.8526)))
 ,'from_to_LaGuardia_Airport'
] = 1
#If pickup/dropoff occurs in manhattan centric traffic, fare would be more
df['is_manhattan_centric_traffic'] = 0
df.loc[
    ((df['dropoff_latitude'].between(40.698,40.882)) &
    (df['dropoff_longitude'].between(-74.025,-73.905))) |

    ((df['pickup_latitude'].between(40.698,40.882)) & 
    (df['pickup_longitude'].between(-74.025,-73.905)))
,'is_manhattan_centric_traffic'] = 1

#Rush hour NYC rush hour generally spans 6:00 a.m. to 10:00 a.m. and 4:00 p.m. to 8:00 p.m. on weekdays

df['is_rush_hour'] = 0
df.loc[
 (df['Dayofweek'] <5) & (((df['Hour']>=6) & (df['Hour'] <=10)) | ((df['Hour']>=16) & (df['Hour']<=20) ))
,'is_rush_hour'] = 1

#Nighttime extra fee
df['is_night_trip'] = 0
df.loc[
  (df['Hour'] >= 20) | (df['Hour'] <= 6) , 'is_night_trip'
] = 1

#dropped pickup_datetime column to remove any noise the model might run into
df_new=df.drop(columns=['pickup_datetime','key'])
#dropped any rows that have a fare amount,total distance or passenger count <=0
df_new = df_new.loc[(df_new['total_distance']> 0 )  & (df_new['fare_amount']>0) & (df_new['passenger_count']>0)]

#splitting the data
x_features = df_new.drop(columns = 'fare_amount')
y_target = df_new['fare_amount']
#model will be used to make the predicitons on 20% of the original data
x_train,x_test,y_train,y_test = train_test_split(x_features,y_target,test_size=0.2,train_size=0.8)
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=12,
    random_state=42,
    n_jobs = -1
)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
accuracy = root_mean_squared_error(y_test,predictions)
print(accuracy)
print(df_new.columns)
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
