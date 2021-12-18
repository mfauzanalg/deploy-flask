from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
import subprocess
import sys
from flask_cors import CORS, cross_origin
import pandas as pd
import joblib
from helper import data_input_preprocess

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

api = Api(app)

classifier = joblib.load('classifier.pkl')

class Churn(Resource):
  def get(self):
    data = pd.read_csv('users.csv')
    data = data.to_dict() 
    return {'data': data}, 200

  def post(self):
    parser = reqparse.RequestParser()  # initialize
    
    parser.add_argument('gender', required=True)
    parser.add_argument('seniorCitizen', required=True)
    parser.add_argument('partner', required=True)
    parser.add_argument('dependents', required=True)
    parser.add_argument('tenureMonths', required=True)
    parser.add_argument('phoneService', required=True)
    parser.add_argument('multipleLines', required=True)
    parser.add_argument('internetService', required=True)
    parser.add_argument('onlineSecurity', required=True)
    parser.add_argument('onlineBackup', required=True)
    parser.add_argument('deviceProtection', required=True)
    parser.add_argument('techSupport', required=True)
    parser.add_argument('streamingTV', required=True)
    parser.add_argument('streamingMovies', required=True)
    parser.add_argument('contract', required=True)
    parser.add_argument('paperlessBilling', required=True)
    parser.add_argument('paymentMethod', required=True)
    parser.add_argument('monthlyCharges', required=True)
    
    args = parser.parse_args()  # parse arguments to dictionary
    
    # create new dataframe containing new values
    dt_inp = [{
      'gender': args['gender'],
      'senior_citizen': args['seniorCitizen'],
      'partner': args['partner'],
      'dependents': args['dependents'],
      'tenure_months': int(args['tenureMonths']),
      'phone_service': args['phoneService'],
      'multiple_lines': args['multipleLines'],
      'internet_service': args['internetService'],
      'online_security': args['onlineSecurity'],
      'online_backup': args['onlineBackup'],
      'device_protection': args['deviceProtection'],
      'tech_support': args['techSupport'],
      'streaming_tv': args['streamingTV'],
      'streaming_movies': args['streamingMovies'],
      'contract': args['contract'],
      'paperless_billing': args['paperlessBilling'],
      'payment_method': args['paymentMethod'],
      'monthly_charges': int(args['monthlyCharges']),
      'total_charges': int(args['monthlyCharges']) * int(args['tenureMonths']),
    }]
    
    print(dt_inp)
    dt_inp = pd.DataFrame(dt_inp)
    dt_inp = data_input_preprocess(dt_inp)
    dt_inp = dt_inp.head(1).drop(columns=['churn_value'], axis=1)
    y_pred_input = classifier.predict(dt_inp)
    print('result:')
    print(y_pred_input[0])
    return {'data': int(y_pred_input[0])}, 200  # return data with 200 OK
    
api.add_resource(Churn, '/churn')

if __name__ == '__main__':
    app.run() 