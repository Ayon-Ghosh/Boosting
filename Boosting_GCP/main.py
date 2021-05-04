# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pandas as pd
import pickle
from xgboost import XGBClassifier

app = Flask(__name__) # initializing a flask app


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Pregnancies=float(request.json['number_of_times_pregnant'])
            Glucose = float(request.json['plasma_glucose_concentration'])
            BloodPressure = float(request.json['diastolic_blood_pressure'])
            SkinThickness  = float(request.json['triceps_skinfold_thickness'])
            Insulin  = float(request.json['serum_insulin'])
            BMI  = float(request.json['body_mass_index'])
            Diabetes_pedigree_function = float(request.json['diabetes_pedigree_function'])
            Age = float(request.json['age'])

            #converting it into a datafframe
            to_pred = pd.DataFrame(
                {'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BloodPressure': BloodPressure, 'SkinThickness': SkinThickness,
                 'Insulin': Insulin, 'BMI': BMI,
                 'Diabetes pedigree function': Diabetes_pedigree_function, 'Age': Age}, index=[1])
            print(f'to_pred: {to_pred}')

            # Loading the saved models into memory
            filename = 'xgboost_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb'))

            # predictions using the loaded model file

            pred=loaded_model.predict(to_pred)
            print('prediction is', pred[0])
            if pred[0]==1:
                result= 'The Patient is Diabetic'
            else:
                result = 'The Patient is not Diabetic'
            # showing the prediction results in a UI
            return jsonify(result)
        except Exception as e:
            print('The Exception message is: ',e)
            return jsonify('error: Something is wrong')
    # return render_template('results.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
   app.run(debug=True) # running the app
