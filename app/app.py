from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open(r'wheat_suitability_model.pkl', 'rb') as f:
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


if __name__ == '__main__':
    app.run(debug=True)
