import os
import joblib
import locale
import numpy as np
import mysql.connector
import tensorflow as tf
from joblib import load
from dotenv import load_dotenv
from datetime import datetime
from flask import Flask, request, jsonify

load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

# Connect to the database
def connect_to_database():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
# Flask app
app = Flask(__name__)

# Paths untuk file lokal
model_path = "models/financial_goal_model.h5"
scaler_X_path = "models/scaler_X.pkl"
scaler_y_path = "models/scaler_y.pkl"

# URLs untuk file di Google Cloud Storage
model_url = "https://storage.googleapis.com/fingoal-storage/model-h5/financial_goal_model.h5"
scaler_X_url = "https://storage.googleapis.com/fingoal-storage/model-h5/scaler_X.pkl"
scaler_y_url = "https://storage.googleapis.com/fingoal-storage/model-h5/scaler_y.pkl"

# Fungsi download file jika belum ada
def download_file_if_needed(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"{local_path} downloaded successfully!")
        else:
            raise Exception(f"Failed to download {url}, status code: {response.status_code}")
    else:
        print(f"{local_path} already exists. Skipping download.")

# Download file jika diperlukan
os.makedirs("models", exist_ok=True)
download_file_if_needed(model_url, model_path)
download_file_if_needed(scaler_X_url, scaler_X_path)
download_file_if_needed(scaler_y_url, scaler_y_path)

# Load model
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Load scaler files
# menggunakan joblib
scaler_X = load(scaler_X_path)
scaler_y = load(scaler_y_path)

print("Scalers loaded successfully!")


# Fungsi untuk normalisasi input
def normalize_input(input_data, scaler_min, scaler_max):
    return (input_data - scaler_min) / (scaler_max - scaler_min)

# Fungsi pembulatan
def rounded_up_to_nearest(value, nearest=100000):
    """
    Membulatkan nilai ke kelipatan terdekat.
    """
    return (int(value / nearest) + 1) * nearest if value % nearest != 0 else int(value)

# Fungsi format mata uang
def format_currency(value):
    """
    Memformat nilai ke dalam bentuk mata uang Rupiah.
    """
    locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8') 
    return locale.currency(value, grouping=True, symbol=True)

# Fungsi untuk denormalisasi output
def denormalize_output(output_data, scaler_min, scaler_max):
    return output_data * (scaler_max - scaler_min) + scaler_min


@app.route('/money/goal/predict', methods=['POST'])
def predict():
    try:
     
        data = request.get_json()
 
        id_goal = data.get("id_goal")

        connection = connect_to_database()
        get_goal = connection.cursor()

        get_goal.execute( "SELECT * FROM goals WHERE id_goal = %s", (id_goal,))
        goal = get_goal.fetchone()
        goal_amount = goal[4]
        target = goal[5]
        current_savings = goal[7]
        created_at = goal[9]


        # Hitung selisih bulan
        goal_duration = (target.year - created_at.year) * 12 + (target.month - created_at.month)
        print(goal_duration)
        print(goal)

        
        # Siapkan data input untuk model
        input_data = np.array([[goal_amount, goal_duration, current_savings]])

       
        # Normalisasi input
        normalized_input = scaler_X.transform(input_data)

        # Prediksi dengan model
        predicted_normalized = model.predict(normalized_input)
        denormalized_prediction = scaler_y.inverse_transform(predicted_normalized)

        # Post-processing hasil prediksi
        predicted_roundedup = rounded_up_to_nearest(denormalized_prediction[0][0])
        predicted_formatted = format_currency(predicted_roundedup)

        # Kembalikan hasil prediksi
        response_data = {
            "prediction": predicted_formatted,
            "raw_prediction": float(denormalized_prediction[0][0]),
            "message": f"Dengan melihat tujuan keuangan anda, kami merekomendasikan anda untuk menyisihkan sebesar {predicted_formatted} setiap bulannya."
        }

        # Simpan hasil ke database
        try:
            connection = connect_to_database()
            update_goal = connection.cursor()

            result= f"Dengan melihat tujuan keuangan anda, kami merekomendasikan anda untuk menyisihkan sebesar {predicted_formatted} setiap bulannya."

            update_goal.execute("UPDATE goals set predict_goal = %s WHERE id_goal = %s", (result,id_goal))
            
        
            connection.commit()

        except Exception as db_error:
            print(f"Database error: {db_error}")
            
        return jsonify(response_data,goal_amount,current_savings,goal_duration)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    PORT = int(os.environ.get("PORT"))
    app.run(host="0.0.0.0", port=PORT, debug=True)