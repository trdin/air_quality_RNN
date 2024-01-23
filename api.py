# %%
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import requests


# %%
model = tf.keras.models.load_model('best_model_final.h5')
pm_10_scaler = joblib.load('pm10_scaler.joblib')
other_scaler = joblib.load('other_scaler.joblib')


# %%
def weather_data(lat, lon, start_date, end_date):
  start_date = start_date.strftime('%Y-%m-%d')
  end_date = end_date.strftime('%Y-%m-%d')

  base_url = "https://archive-api.open-meteo.com/v1/archive"
  params = {
    "timezone": "GMT",
    "latitude": lat,
    "longitude": lon,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "weather_code", "wind_speed_10m", "soil_temperature_100_to_255cm"],
  }

  response = requests.get(base_url, params=params)
  data = response.json()


  df = pd.DataFrame(data['hourly'])
  df['Date'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M')
  df['Date'] = df['Date'].dt.tz_localize('UTC')
  df['Date_Hour'] = df['Date'].dt.floor('H')
  df.drop(columns=['time', 'Date'], inplace=True)




  return df

# %%
def new_variables(all_data):
    all_data['NO2_PM2.5_Interact'] = all_data['NO2'] * all_data['PM2.5']
    all_data['Day'] = all_data['Date'].dt.day
    all_data['Month'] = all_data['Date'].dt.month
    all_data['Hour'] = all_data['Date'].dt.hour
    all_data['Day_of_Week'] = all_data['Date'].dt.dayofweek
    all_data['Hour_NO2_Interaction'] = all_data['Hour'] * all_data['NO2']
    all_data['Hour_PM2.5_Interaction'] = all_data['Hour'] * all_data['PM2.5']

    weather_df = weather_data(all_data['Latitude'].iloc[0], all_data['Longitude'].iloc[0], all_data['Date'].iloc[0], all_data['Date'].iloc[-1])
    all_data['Date_Hour'] = all_data['Date'].dt.floor('H')
    merged_data = pd.merge(all_data, weather_df, on='Date_Hour', how='left')
    merged_data.drop(columns=['Date_Hour'], inplace=True)
    all_data = merged_data.copy()

    all_data['relative_humidity_NO2_Interaction'] = all_data['relative_humidity_2m'] * all_data['NO2']
    all_data['relative_humidity_PM2.5_Interaction'] = all_data['relative_humidity_2m'] * all_data['PM2.5']
    return all_data

# %%
def predict(data):
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

        left_skew_columns = ["relative_humidity_2m", "soil_temperature_100_to_255cm"]
        for col in left_skew_columns:
            data[col] = np.square(data[col])

        right_skew_columns = ["NO2", "PM2.5", "NO2_PM2.5_Interact", "Hour_NO2_Interaction", "Hour_PM2.5_Interaction", "precipitation", "rain", "wind_speed_10m", "relative_humidity_NO2_Interaction", "relative_humidity_PM2.5_Interaction"]
        for col in right_skew_columns:
            data[col] = np.log(data[col]+1 )


        selected_features = ['PM2.5',
        'NO2_PM2.5_Interact',
        'relative_humidity_PM2.5_Interaction',
        'Hour_PM2.5_Interaction',
        'NO2',
        'relative_humidity_NO2_Interaction',
        'Hour_NO2_Interaction',
        'soil_temperature_100_to_255cm',
        'temperature_2m']

        learn_features = data[['PM10'] + list(selected_features) ]
        learn_features = learn_features.values

        

        pm10 = np.array(learn_features[:,0])
    
        pm10_normalized = pm_10_scaler.transform(pm10.reshape(-1, 1))

        other = np.array(learn_features[:,1:])
        other_normalized = other_scaler.transform(other)


        normalized_data = np.column_stack([pm10_normalized, other_normalized])

        X_predict = normalized_data   	
       
        X_predict = X_predict.reshape(1, X_predict.shape[1], X_predict.shape[0])
       

        prediction = model.predict(X_predict)
        prediction =  pm_10_scaler.inverse_transform(prediction)
        

        return {'prediction': prediction.tolist()}
    except Exception as e:
        return {'error': str(e)}, 400

# %%
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_air():
    data = request.get_json()
    result = predict(data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=123)


