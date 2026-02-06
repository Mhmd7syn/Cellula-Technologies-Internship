from flask import Flask, render_template, request
import pickle
import datetime
import pandas as pd

app = Flask(__name__)

# Load the trained encoder
with open('D:\\Cellula Internship\\Trip Fare Prediction\\encoder.pkl', 'rb') as en:
    encoder = pickle.load(en)

# Load the trained scaler
with open('D:\\Cellula Internship\\Trip Fare Prediction\\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load your trained model
with open('D:\\Cellula Internship\\Trip Fare Prediction\\NN_model.pkl', 'rb') as file:
    model = pickle.load(file)

def encoding(df):
    car_c_sorted = ['Bad', 'Good', 'Very Good', 'Excellent']
    car_condition_encoded = [car_c_sorted.index(x) for x in df['car_condition']]
    df.insert(df.columns.get_loc('car_condition') + 1, 'car_condition_encoded', car_condition_encoded)

    traffic_c_sorted = ['Congested Traffic', 'Dense Traffic', 'Flow Traffic']
    traffic_condition_encoded = [traffic_c_sorted.index(x) for x in df['traffic_condition']]
    df.insert(df.columns.get_loc('traffic_condition') + 1, 'traffic_condition_encoded', traffic_condition_encoded)

    weather_encoded = encoder.transform(df[['weather']])
    weather_encoded = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out())
    for column in weather_encoded.columns:
        df.insert(df.columns.get_loc('weather') + 1, column.removeprefix('weather_'), weather_encoded[column])

    for column in ['car_condition', 'traffic_condition', 'weather']:
        df.drop(column, axis=1, inplace=True)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Extract data from the form
        passenger_count = int(request.form.get('passenger_count'))
        distance = float(request.form.get('distance'))
        jfk_dist = float(request.form.get('jfk_dist'))
        lga_dist = float(request.form.get('lga_dist'))
        car_condition = request.form.get('car_condition')
        traffic_condition = request.form.get('traffic_condition')
        weather = request.form.get('weather')
        bearing = float(request.form.get('bearing'))
        datetime_info = request.form.get('datetime')  # e.g., "2025-03-13T17:19"

        # Process datetime info
        dt = datetime.datetime.fromisoformat(datetime_info)
        hour = dt.hour
        day = dt.day
        month = dt.month
        year = dt.year
        weekday = dt.weekday()  # Monday = 0, Sunday = 6


        df = pd.DataFrame(
            columns=['car_condition', 'weather', 'traffic_condition', 'passenger_count', 'hour', 'day', 'month', 'weekday', 'year', 'jfk_dist', 'lga_dist', 'distance', 'bearing'],
            data=[[car_condition, weather, traffic_condition, passenger_count, hour, day, month, weekday, year, jfk_dist, lga_dist, distance, bearing]]  # Enclose data in a list
        )

        encoding(df)

        # Scale the features
        scaled_features = scaler.transform(df)

        # Predict the fare amount
        prediction = model.predict(scaled_features)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)