import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pickle

def create_model(data):
    X = data.drop(columns = ['selling_price'], axis=1)
    y = data['selling_price']
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rfr = RandomForestRegressor(max_depth=20, min_samples_leaf=2, random_state=42)
    rfr.fit(x_train, y_train)

    y_pred = rfr.predict(x_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", root_mean_squared_error(y_test, y_pred))

    return rfr


def clean_data():
    data = pd.read_csv("E:\\23881A66E2\\Projects\\Car_Price_Prediction\\data\\cardekho_dataset.csv")
    data = data.drop(columns = ['Unnamed: 0', 'car_name', 'brand', 'model'])
    data = pd.get_dummies(data, columns=['fuel_type', 'seller_type', 'transmission_type'], drop_first=True)
    data['power_per_cc'] = data['max_power'] / data['engine']
    data['diesel_auto'] = (data['fuel_type_Diesel'] == 1) & (data['transmission_type_Manual'] == 0)
    data['diesel_auto'] = data['diesel_auto'].astype(int) 

    return data


def main():
    data = clean_data()
    model = create_model(data)

    with open('E:\\23881A66E2\\Projects\\Car_Price_Prediction\\model\\model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
