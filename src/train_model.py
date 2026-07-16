import pandas as pd
import joblib
from preprocessing import load_data, cleaned_df


df = load_data()
clean_df = cleaned_df(df)


x = clean_df.drop("Price",axis = 1)
y = clean_df[["Price"]]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Doing binary mapping
x_train["Parking"]=x_train["Parking"].astype(int)
x_test["Parking"]= x_test["Parking"].astype(int)
x_train["Warehouse"]=x_train["Warehouse"].astype(int)
x_test["Warehouse"]=x_test["Warehouse"].astype(int)
x_train["Elevator"]=x_train["Elevator"].astype(int)
x_test["Elevator"]=x_test["Elevator"].astype(int)

#Encoding one hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
encoded_train = encoder.fit_transform(x_train[["Address"]])
encoded_test = encoder.transform(x_test[["Address"]])
encoded_train_df = pd.DataFrame(
    encoded_train,
    columns=encoder.get_feature_names_out(["Address"]),
    index=x_train.index
)
encoded_test_df = pd.DataFrame(
    encoded_test,
    columns=encoder.get_feature_names_out(["Address"]),
    index=x_test.index
)

x_train = x_train.drop("Address",axis=1)
x_train = pd.concat([x_train,encoded_train_df],axis=1)
x_test = x_test.drop("Address",axis=1)
x_test = pd.concat([x_test,encoded_test_df],axis=1)

joblib.dump(encoder,"models/encoder.pkl")

#Scaling data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
joblib.dump(scaler,"models/scaler.pkl")

#Training model

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

lr_model = LinearRegression()
dt_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor()
models = {
    "LinearRegression":lr_model,
    "DecisionTreeRegressor":dt_model,
    "RandomForestRegressor":rf_model
}

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
best_r2 = float("-inf")
best_model = None
best_metrics = None

for name,model in models.items():
    
    model.fit(x_train_scaled,y_train)
    y_pred = model.predict(x_test_scaled)

    mae = round(mean_absolute_error(y_test,y_pred),2)
    mse = round(mean_squared_error(y_test,y_pred),2)
    model_r2 = round(r2_score(y_test,y_pred))
    print(name)
    print(mae)
    print(mse)
    print(r2_score)

   
    
   
    if model_r2 > best_r2:
        best_r2 = model_r2
        best_model = model
        best_metrics={

            "MAE":mae,
            "MSE":mse,
            "R2 score": model_r2
        }
print(f"The best model is {best_model},with an r2 score of {best_r2} ")

joblib.dump(best_model,"models/best_model.pkl")
joblib.dump(best_metrics,"models/best_metrics.pkl")



