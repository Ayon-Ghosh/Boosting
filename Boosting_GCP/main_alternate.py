from wsgiref import simple_server
from flask import Flask, request, app, request,jsonify
from flask_cors import CORS,cross_origin
from flask import Response
import pandas as pd
import pickle
from xgboost import XGBClassifier

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True


#@app.route('/from_postman',methods=['POST']) # route to show the predictions in a web UI
#@cross_origin()
#def from_postman():
@app.route("/predict_alternate", methods=['POST'])
def predictRoute():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is:     ', data)
            res = predict_log(data)
            print('result is     ',res)
            return res
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)

def predict_log(dict_pred):

    #data_df = pd.DataFrame(dict_pred["data"], index=[1])
    to_pred = pd.DataFrame(dict_pred,index=[1])
    to_pred = to_pred.rename(columns={'number_of_times_pregnant': 'Pregnancies', 'plasma_glucose_concentration': 'Glucose',
                                'diastolic_blood_pressure': 'BloodPressure',
                                'triceps_skinfold_thickness': 'SkinThickness',
                                'serum_insulin': 'Insulin',
                                'body_mass_index': 'BMI',
                                'diabetes_pedigree_function': 'Diabetes pedigree function', 'age': 'Age'})
    print(to_pred)
    print(to_pred.dtypes)
    filename = 'xgboost_model.pickle'
    loaded_model = pickle.load(open(filename, 'rb'))
    pred = loaded_model.predict(to_pred)
    print('prediction is', pred[0])
    if pred[0] == 1:
        result = 'The Patient is Diabetic'
    else:
        result = 'The Patient is not Diabetic'

    return jsonify(result)


if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
    #httpd = simple_server.make_server(host, port, app)
    #print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()