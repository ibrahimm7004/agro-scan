from flask import Flask, render_template, request
import pickle
import pandas as pd
import joblib

app = Flask(__name__)

# loading stored models:
irrigation_model = joblib.load('models/irrigation_model.pkl')
yield_model = joblib.load('models/yield-estimate-model.pkl')
soil_type_model = joblib.load('models/soil_type_model.pkl')

scaler = joblib.load('models/soil_scaler.pkl')

# Mapping of cluster numbers to real soil names
soil_type_mapping = {
    0: 'Sandy Loam',
    1: 'Clay Soil',
    2: 'Loamy Soil'
}


with open('models/wheat_suitability_model.pkl', 'rb') as f:
    clf = pickle.load(f)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/page1', methods=['GET', 'POST'])
def wheat_growth():
    prediction = None
    if request.method == 'POST':
        try:
            user_data = pd.DataFrame([{
                'Nitrogen_N_mg/kg': float(request.form['nitrogen']),
                'Phosphorus_P_mg/kg': float(request.form['phosphorus']),
                'Potassium_K_mg/kg': float(request.form['potassium']),
                'Temperature_C': float(request.form['temperature']),
                'Humidity_%': float(request.form['humidity']),
                'Rainfall_mm': float(request.form['rainfall']),
                'Soil_Moisture_%': float(request.form['soil_moisture'])
            }])

            prediction = clf.predict(user_data)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('page1.html', title='Wheat Growth', prediction=prediction)


@app.route('/page2', methods=['GET', 'POST'])
def irrigation_prediction():
    irrigation_result = None
    if request.method == 'POST':
        try:
            new_data = pd.DataFrame([{
                'Temperature_C': float(request.form['temperature']),
                'Humidity_%': float(request.form['humidity']),
                'Rainfall_mm': float(request.form['rainfall']),
                'Soil_Moisture_%': float(request.form['soil_moisture'])
            }])
            irrigation_result = round(irrigation_model.predict(new_data)[0], 4)
        except Exception as e:
            irrigation_result = f"Error: {str(e)}"
    return render_template('page2.html', title='Irrigation Prediction', prediction=irrigation_result)


@app.route('/page3', methods=['GET', 'POST'])
def crop_yield_prediction():
    yield_result = None
    if request.method == 'POST':
        try:
            sample_data = pd.DataFrame([{
                'Nitrogen_N_mg/kg': float(request.form['nitrogen']),
                'Phosphorus_P_mg/kg': float(request.form['phosphorus']),
                'Potassium_K_mg/kg': float(request.form['potassium']),
                'Temperature_C': float(request.form['temperature']),
                'Humidity_%': float(request.form['humidity']),
                'Rainfall_mm': float(request.form['rainfall']),
                'Soil_Moisture_%': float(request.form['soil_moisture']),
            }])
            yield_result = round(yield_model.predict(sample_data)[0], 2)
        except Exception as e:
            yield_result = f"Error: {str(e)}"

    return render_template('page3.html', title='Crop Yield Predictor', prediction=yield_result)


@app.route('/page4')
def feature_importance():
    return render_template('page4.html', title='Feature Importance')


@app.route('/page5', methods=['GET', 'POST'])
def soil_type_detection():
    soil_result = None
    if request.method == 'POST':
        try:
            # Get user inputs
            new_data = pd.DataFrame([{
                'Nitrogen_N_mg/kg': float(request.form['nitrogen']),
                'Phosphorus_P_mg/kg': float(request.form['phosphorus']),
                'Potassium_K_mg/kg': float(request.form['potassium']),
                'Soil_Moisture_%': float(request.form['soil_moisture'])
            }])

            # Standardize the inputs using stored soil scaler file
            new_data_scaled = scaler.transform(new_data)

            # Predict cluster
            cluster_label = soil_type_model.predict(new_data_scaled)[0]

            # Map to real soil name
            soil_result = soil_type_mapping[cluster_label]

        except Exception as e:
            soil_result = f"Error: {str(e)}"

    return render_template('page5.html', title='Soil Type Detection', prediction=soil_result)


if __name__ == '__main__':
    app.run(debug=True)
