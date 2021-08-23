from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# create flask app
app = Flask(__name__)

# Load pickle model
model = pickle.load(open("model/best_model_rf.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    output = {0: 'lose', 1: 'win'}

    return render_template("index.html", prediction_text="Blue team will {}".format(output[prediction[0]]))


if __name__ == '__main__':
    app.run(debug=True)
