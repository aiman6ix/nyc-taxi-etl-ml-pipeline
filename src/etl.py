import pandas as pd
import numpy as np

def extract_and_clean_data(csv):
    data =  pd.read_csv(csv)

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
    return df

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

def engineer_features(df):
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

    
    #dropped any rows that have a fare amount,total distance or passenger count <=0
    df = df.loc[(df['total_distance']> 0 )  & (df['fare_amount']>0) & (df['passenger_count']>0)]

    return df
