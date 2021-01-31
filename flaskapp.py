#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 20:35:41 2021

@author: riteshtripathi
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "Hello The answer is"+str(prediction)

@app.route('/predict_file', methods = ["POST"])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    #print(df_test.head())
    prediction=classifier.predict(df_test)
    return "The predicted value for  the csv file is" + str(list(prediction))

if __name__=='__main__':
    #app.run(host='0.0.0.0',port=8000)
    app.run()