import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import json

# Assuming result_dict is a dictionary with meaningful keys


app = Flask(__name__)
app.secret_key = "ABCDE"  # Replace with your own secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model("resources/model.tf")

# Function to check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File uploaded successfully')

        # Process the uploaded CSV file and make predictions
        result_filename,result_dict = process_uploaded_file(filename)
        result_json = json.dumps(result_dict)  # Convert to JSON string
        return render_template('result.html', result=result_json,result_filename=result_filename)
       
    else:
        flash('Invalid file type. Please upload a CSV file.')
        return redirect(request.url)

def process_uploaded_file(filename):
    # Load the uploaded CSV file into a DataFrame
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Extract features (modify as needed to match your dataset)
    features = df[['month_1', 'month_2', 'month_3', 'day_off', 'danger_level', 'nedbor',
                   'vind_styrke', 'temperatur_mean', 'aval_probability_id_0', 'aval_probability_id_3',
                   'aval_probability_id_5', 'aval_probability_id_7', 'aval_probability_id_10', 'aval_probability_id_30',
                   'aval_probability_id_45', 'aval_probability_id_50']]

    # Normalize the input data using the same MinMaxScaler used during training
    scaler = MinMaxScaler()
    scaler.fit(features)
    input_data_normalized = scaler.transform(features)

    # Make predictions
    predictions = model.predict(input_data_normalized)

    # Determine the class label for each row
    class_labels = ['Avalanche' if prediction[0] > prediction[1] else 'No Avalanche' for prediction in predictions]

    # Add the class labels to the DataFrame
    df['Prediction'] = class_labels

    # Create a dictionary with the prediction results
    result_dict = df[['month_1', 'month_2', 'month_3', 'day_off', 'danger_level', 'nedbor',
                   'vind_styrke', 'temperatur_mean', 'aval_probability_id_0', 'aval_probability_id_3',
                   'aval_probability_id_5', 'aval_probability_id_7', 'aval_probability_id_10', 'aval_probability_id_30',
                   'aval_probability_id_45', 'aval_probability_id_50', 'Prediction']].to_dict(orient='records')

    # Save the updated DataFrame to a new CSV file
    result_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions_' + filename)
    df.to_csv(result_filename, index=False)

    return result_filename,result_dict


@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
