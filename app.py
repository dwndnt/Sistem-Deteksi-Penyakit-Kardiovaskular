from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__, template_folder='templates')

# Memuat model
model = joblib.load('model_xgboost.pkl')

# Routing untuk halaman landing
@app.route('/')
def landing():
    return render_template('landing.html')

# Routing untuk halaman utama
@app.route('/index')
def index():
    return render_template('index.html') 

# Routing untuk melakukan prediksi
@app.route('/prediction', methods=['POST'])
def predict():
    # Mengambil data dari form input
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    ca = int(request.form['ca'])
    cp_0 = int(request.form['cp_0'])
    cp_1 = int(request.form['cp_1'])
    cp_2 = int(request.form['cp_2'])
    cp_3 = int(request.form['cp_3'])
    slope_0 = int(request.form['slope_0'])
    slope_1 = int(request.form['slope_1'])
    slope_2 = int(request.form['slope_2'])
    thal_0 = int(request.form['thal_0'])
    thal_1 = int(request.form['thal_1'])
    thal_2 = int(request.form['thal_2'])

    # Membuat dataframe untuk data pengguna
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'ca': [ca],
        'cp_0': [cp_0],
        'cp_1': [cp_1],
        'cp_2': [cp_2],
        'cp_3': [cp_3],
        'slope_0': [slope_0],
        'slope_1': [slope_1],
        'slope_2': [slope_2],
        'thal_0': [thal_0],
        'thal_1': [thal_1],
        'thal_2': [thal_2]
    })

    # Melakukan prediksi menggunakan model
    prediction = model.predict(user_data)

    # Menampilkan hasil prediksi
    if prediction[0] == 1:
        result = 'Kardiovaskular'
    else:
        result = 'Non-Kardiovaskular'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
