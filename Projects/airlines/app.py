from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the trained model
with open('linear_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    airline = request.form['Airline']
    source = request.form['Source']
    destination = request.form['Destination']
    duration = request.form['Duration']
    total_stops = int(request.form['Total_Stops'])
    
    # Prepare the data for prediction
    input_data = pd.DataFrame([[airline, source, destination, duration, total_stops]],
                              columns=['Airline', 'Source', 'Destination', 'Duration', 'Total_Stops'])
    
    # Perform prediction
    prediction = model.predict(input_data)
    
    # Send the response
    return render_template('index.html', prediction_text='Predicted Price: ${:.2f}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
