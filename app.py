from flask import Flask, render_template, request, session, redirect, url_for
from markupsafe import Markup
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pickle

import tensorflow as tf

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql

# Check for GPU availability
if tf.config.list_physical_devices('GPU'):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trained_model.h5')
    classifier = load_model(model_path)
else:
    print("GPU not available, loading model on CPU.")
    with tf.device('/CPU:0'):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trained_model.h5')
        classifier = load_model(model_path)
# The following line is no longer needed in the latest versions of TensorFlow/Keras

crop_recommendation_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Crop_Recommendation.pkl')
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure secret key

# MySQL database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Manojc@localhost:3306/login_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Create the database and tables if they don't exist
with app.app_context():
    # Connect to MySQL server without specifying database to create it first
    from sqlalchemy import create_engine
    engine = create_engine('mysql+pymysql://root:Manojc@localhost:3306')
    conn = engine.connect()
    from sqlalchemy import text
    conn.execute(text("CREATE DATABASE IF NOT EXISTS login_db"))
    conn.close()
    db.create_all()

@ app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'Crop_NPK.csv'))

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)


def pred_pest(pest):
    try:
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict_classes(test_image)
        return result
    except:
        return 'x'

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = 'Invalid username or password'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if User.query.filter_by(username=username).first():
            error = 'Username already exists'
            return render_template('register.html', error=error)
        if password != confirm_password:
            error = 'Passwords do not match'
            return render_template('register.html', error=error)
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        session['username'] = username
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route("/")
@app.route("/index.html")
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)
        if pred == 'x':
            return render_template('unaptfile.html')
        if pred[0] == 0:
            pest_identified = 'aphids'
        elif pred[0] == 1:
            pest_identified = 'armyworm'
        elif pred[0] == 2:
            pest_identified = 'beetle'
        elif pred[0] == 3:
            pest_identified = 'bollworm'
        elif pred[0] == 4:
            pest_identified = 'earthworm'
        elif pred[0] == 5:
            pest_identified = 'grasshopper'
        elif pred[0] == 6:
            pest_identified = 'mites'
        elif pred[0] == 7:
            pest_identified = 'mosquito'
        elif pred[0] == 8:
            pest_identified = 'sawfly'
        elif pred[0] == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)

@ app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')

@app.route("/wheather")
def wheather():
    return render_template("wheather.html")

if __name__ == '__main__':
    app.run(debug=True)
