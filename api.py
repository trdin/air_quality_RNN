# %%
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# %%
model = tf.keras.models.load_model('best_model.h5')
pm_10_scaler = joblib.load('pm10_scaler.joblib')
other_scaler = joblib.load('other_scaler.joblib')


# %%
def new_variables(all_data):
    all_data['NO2_PM2.5_Interact'] = all_data['NO2'] * all_data['PM2.5']
    all_data['Day'] = all_data['Date'].dt.day
    all_data['Month'] = all_data['Date'].dt.month
    all_data['Hour'] = all_data['Date'].dt.hour
    all_data['Day_of_Week'] = all_data['Date'].dt.dayofweek
    all_data['Hour_NO2_Interaction'] = all_data['Hour'] * all_data['NO2']
    all_data['Hour_PM2.5_Interaction'] = all_data['Hour'] * all_data['PM2.5']
    return all_data

# %%
def create_multivariate_dataset_with_steps(time_series, look_back=1, step=1):
    X, y = [], []
    for i in range(0, len(time_series) - look_back, step):
        X.append(time_series[i:(i + look_back), :])
        y.append(time_series[i + look_back, 0]) 
    return np.array(X), np.array(y)


# %%
def vaja6_predict(data):
    try:
        required_features = ['Date','Latitude', 'Longitude', 'Altitude', 'NO2', 'PM2.5', 'O3', 'PM10']
        for obj in data:
            for feature in required_features:
                if feature not in obj:
                    return {'error': f'Missing feature: {feature}'}, 400


        data = pd.DataFrame(data)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by=['Date'])

        data = new_variables(data)

        right_skew_columns = ["NO2", "PM2.5", "NO2_PM2.5_Interact", "Hour_NO2_Interaction", "Hour_PM2.5_Interaction"]
        for col in right_skew_columns:
            data[col] = np.log(data[col]+1 )

        selected_features = ['PM2.5', 'NO2_PM2.5_Interact', 'Hour_PM2.5_Interaction', 'NO2', 'Hour_NO2_Interaction', 'Day', 'Month']

        learn_features = data[list(selected_features) + ['PM10']]
        learn_features = learn_features.values

        

        pm10 = np.array(learn_features[:,7])
    
        pm10_normalized = pm_10_scaler.transform(pm10.reshape(-1, 1))

        other = np.array(learn_features[:,:7])
        other_normalized = other_scaler.transform(other)


        normalized_data = np.column_stack([pm10_normalized, other_normalized])

        X_train = normalized_data   	
        look_back = 48
        step = 1

        #X_train, y_train = create_multivariate_dataset_with_steps(normalized_data, look_back, step)
        #X_test, y_test = create_multivariate_dataset_with_steps(test_normalized, look_back, step)

        print(X_train.shape)

        X_train = X_train.reshape(1, X_train.shape[1], X_train.shape[0])
       # X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])
        
        
        

        """ target_feature = 'PM10'
        pm10_series = np.array(data[target_feature].values.reshape(-1, 1))

        pm10_series = pm_10_scaler.transform(pm10_series) """

        prediction = model.predict(X_train)
        prediction =  pm_10_scaler.inverse_transform(prediction)
        

        return {'prediction': prediction.tolist()}
    except Exception as e:
        return {'error': str(e)}, 400

# %%
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_naloga6():
    data = request.get_json()
    result = vaja6_predict(data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=123)


