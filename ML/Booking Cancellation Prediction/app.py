from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load the trained scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load your trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

def Encoding(df):
    df['type_of_meal_encoded'] = [0 if x == 'Not Selected' else int( x.replace('Meal Plan ', '')) for x in df['type of meal']]
    df['room_type_encoded'] = [int( x.replace('Room_Type ', '')) for x in df['room type']]
    df['market_segment_type_encoded'] = LabelEncoder().fit_transform(df['market segment type'])
    df.rename(columns={'average price': 'average price '}, inplace=True)
    df.drop(['type of meal', 'room type', 'market segment type'], axis=1, inplace=True)
    for column in df.columns:
        if column == 'average price ':
            df[column] = df[column].astype('float64')
        elif column != 'date of reservation':
            df[column] = df[column].astype('int64')

def new_features(df):
    df.insert(df.columns.get_loc('number of children') + 1, 'family_size', df['number of adults'] + df['number of children'])

    df.insert(df.columns.get_loc('number of week nights') + 1, 'total_nights', df['number of weekend nights'] + df['number of week nights'])

    df.insert(df.columns.get_loc('P-not-C') + 1, 'pre_reserv', df['P-C'] + df['P-not-C'])
    df.insert(df.columns.get_loc('pre_reserv') + 1, 'c_per_reserv', df['P-C'] / df['pre_reserv'] * 100) #the division by zero will generate nulls 
    df.fillna({'c_per_reserv': 0}, inplace=True)  #replace generated nulls with 0

    df['date_new_form'] = [[3, 1, 2018] if date == '2018-2-29' else date.split(sep='/') if '/' in date else date.split(sep='-') for date in df['date of reservation']]
    df['reservation_day'] = [int(x[1]) for x in df['date_new_form']]
    df['reservation_month'] = [int(x[0]) for x in df['date_new_form']]
    df['reservation_year'] = [int(x[2]) for x in df['date_new_form']]
    df.drop(['date_new_form', 'date of reservation'], axis=1, inplace=True)

def scaling(df):
    categorical_columns = ['car parking space', 'repeated', 'type_of_meal_encoded', 'room_type_encoded', 'market_segment_type_encoded']
    numerical_columns = [x for x in df.columns if x not in categorical_columns]
    # Scaling numerical columns
    scaled_data = scaler.transform(df[numerical_columns])
    scaled_df = pd.DataFrame(scaled_data, columns=numerical_columns)

    #join scaled numberical columns with categorical columns
    return scaled_df.join(df[categorical_columns])

def preprocessing(df):
    Encoding(df)
    new_features(df)
    return scaling(df)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form submission
    data = request.form

    features = [
        'number of adults', 'number of children', 'number of weekend nights',
        'number of week nights', 'type of meal',  'room type', 'lead time',
        'market segment type', 'car parking space', 'repeated', 'P-C', 'P-not-C', 'average price', 'special requests',
        'date of reservation'
    ]

    # Ensure the order of features is correct and convert to numpy array
    input_data = {feature: [data[feature]] for feature in features}
    input_features = pd.DataFrame(input_data)
    
    input_features = preprocessing(input_features)

    # Make a prediction
    prediction = model.predict(input_features)

    # Determine the result
    result = "canceled" if prediction[0] == 1 else "not-canceled"

    # Render the template with the prediction result
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
