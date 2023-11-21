from flask import request, render_template, flash
import numpy as np
import joblib
from app import app


@app.route('/')
def home():
    return render_template('base.html')

@app.route("/prediction", methods = ['POST'])
def prediction():
    model = joblib.load('app/model_regression.joblib')

    if request.method ==  'POST':
        val_a_predire = request.form.get("experience")
        try:
            # Tentative de conversion en nombre réel
            val_a_predire = float(val_a_predire)
        except ValueError:
            error_message = "Vous n'avez pas renseigné un nombre réel!!!"
            return render_template('base.html', error_message=error_message)
           

        final_features = np.array(val_a_predire).reshape(-1,1)
        prediction = model.predict(final_features)
        prediction = float(prediction)
        output = round(prediction)
        output = "{:,}".format(output)
        output = output.replace(',', ' ')

    
    
    return render_template('base.html', prediction_text='Le Salaire de cet employé devrais etre de {} FCFA'.format(output))

