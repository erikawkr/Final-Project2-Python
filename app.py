from flask import Flask, render_template, request, send_from_directory
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Logistic Regression Pickle
model1 = pickle.load(open("lreg_model.pkl", "rb"))
# Scaler initialization
scaler = StandardScaler()

app = Flask(__name__, template_folder="templates")


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.root_path + '/static/', filename)

@app.route("/")
def main():
    return render_template('index3.html')


@app.route('/predict_1', methods=['POST'])
def predict_1():
    '''
    For rendering results on HTML GUI
    '''
    MinTemp = float(request.form["MinTemp"])
    MaxTemp = float(request.form["MaxTemp"])
    Rainfall = float(request.form["Rainfall"])
    WindGustSpeed = float(request.form["WindGustSpeed"])
    WindSpeed9am = float(request.form["WindSpeed9am"])
    WindSpeed3pm = float(request.form["WindSpeed3pm"])
    Pressure9am = float(request.form["Pressure9am"])
    Pressure3pm = float(request.form["Pressure3pm"])
    Temp9am = float(request.form["Temp9am"])
    Temp3pm = float(request.form["Temp3pm"])
    # Location = int(request.form["Location"])
    Humidity9am = float(request.form["Humidity9am"])
    Humidity3pm = float(request.form["Humidity3pm"])
    RainToday = float(request.form["RainToday"])

    data_list = [[
        MinTemp, MaxTemp,
        Rainfall,
        WindGustSpeed, WindSpeed9am, WindSpeed3pm,
        Humidity3pm, Humidity9am, Pressure9am, Pressure3pm, Temp9am, Temp3pm,
        RainToday
    ]]
    pred_scaller = scaler.fit_transform(data_list)
    prediction = model1.predict(pred_scaller)

    output = {
        0: "Tidak Hujan",
        1: "Hujan"
    }

    return render_template('index3.html', prediction_text='Prediksi Cuaca Hari Besok adalah : {}'.format(output[prediction[0]]))


if __name__ == '__main__':
    app.run(debug=True)
