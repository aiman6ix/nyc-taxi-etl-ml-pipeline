from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb      #prediction model used

def train_test_model(df):
    #splitting the data
    x_features = df.drop(columns =['pickup_datetime','key','fare_amount'])
    y_target = df['fare_amount']
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
    print(df.columns)
    return x_test,y_test,predictions