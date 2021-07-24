from flask import Flask, render_template, jsonify, request, make_response
import numpy as np
import pickle
from joblib import load
import os
import time

app = Flask(__name__) 

# load .joblib Models
models = {'knn' : load(os.path.join(os.getcwd(), 'models/knn_model.joblib')),
		  'lr' : load(os.path.join(os.getcwd(), 'models/lr_model.joblib')),
		  'cb' : load(os.path.join(os.getcwd(), 'models/cb_model.joblib')),
		  'dt' : load(os.path.join(os.getcwd(), 'models/dt_model.joblib')),
		  'lgb' : load(os.path.join(os.getcwd(), 'models/lgb_model.joblib')),
		  'sgd' : load(os.path.join(os.getcwd(), 'models/sgd_model.joblib')),
		  'svc' : load(os.path.join(os.getcwd(), 'models/svc_model.joblib')),
		  'rf' : load(os.path.join(os.getcwd(), 'models/rf_model.joblib')),
		 }



@app.route("/") 
def home_view(): 
	params = {'normal' : 0,	'diabetes' : 0}
	return render_template('index.html', params = params)


@app.route('/upload', methods=['POST'])
def upload():
	params = dict()

	if (request.method == 'POST'):
		try:
			# Get data from user
			Pregnancies		= float(request.form['Pregnancies'])
			Glucose			= float(request.form['Glucose'])
			BloodPressure	= float(request.form['BloodPressure'])
			SkinThickness	= float(request.form['SkinThickness'])
			Insulin			= float(request.form['Insulin'])
			BMI				= float(request.form['BMI'])
			DPFunction		= float(request.form['DPFunction'])
			Age				= float(request.form['Age'])

			data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPFunction, Age]])
			
			for name, model in models.items():
				prediction = model.predict(data) # get predictions on dat
				params[name] = int(prediction[0]) # add predictions on params dict
				# print(name, prediction)			
 			
			
			params['status'] = True
		except:
			params['status'] = False
	return jsonify(params)

