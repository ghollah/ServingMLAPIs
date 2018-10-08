#!/usr/bin/env python3.7.0
# coding: utf-8
from flask import Flask, request, render_template, jsonify
import pandas as pd
from sqlalchemy import *
from datetime import datetime
import sys
import os, io
import configparser
import json
from flask_cors import CORS
from flask_restful import Resource, Api
from flask_httpauth import HTTPBasicAuth
import re
import requests, random
import pandas as pd
import numpy as np
from sklearn.model_selection import*
from sklearn.metrics import*
# Load scikit's random forest classifier library
from sklearn.ensemble.forest import RandomForestClassifier
#module for class balancing
from imblearn.over_sampling import RandomOverSampler
#classifier
from sklearn.neural_network import MLPClassifier

#function to deal with nulls
def deal_with_nulls(dealing_with_nulls,dataset):
	if dealing_with_nulls=="mean":
		dataset=dataset.fillna(dataset.mean())
	else:
		if dealing_with_nulls=="median":
			dataset=dataset.fillna(dataset.median())
		else:
			if dealing_with_nulls=="interpolate":
				dataset=dataset.interpolate()
			else:
				dataset=dataset.dropna()
	return dataset

predictmodelrf = Flask(__name__)
CORS(predictmodelrf)

def runns(resp_var, size_of_test_data,dataset,positive_class,n_estimators,important_features,dealing_with_nulls):
	dataset = pd.read_csv('raw_data.csv', low_memory=False) # For testing purposes
	#----DATA PREPROCESSING
	#-------dealing with NULL values in the data
	#----------remove the rows in which the response is null
	dataset=dataset.dropna(subset=[resp_var])
	#----------dealing with nulls
	dataset=deal_with_nulls(dealing_with_nulls,dataset)
	#----FEATURE SELECTION
	#-------get predictors important in predicting the response
	#-----------transform categorical predictors to dummy variables
	predictors=dataset.drop(resp_var,axis=1,inplace=False)
	predictors=pd.get_dummies(predictors)
	#-----------balance the classes in the response var
	ros = RandomOverSampler(random_state=0)
	resp=dataset[resp_var]
	prds, resp = ros.fit_sample(predictors, resp)
	#-----------fit the random forest classifier to give us the important predictors
	rf_clf = RandomForestClassifier(n_estimators=n_estimators)
	rf_clf.fit(prds,resp)
	#-------get the important predictors
	feature_imp = pd.Series(rf_clf.feature_importances_,
                    index=list(predictors.iloc[:,0:])).sort_values(ascending=False)
	#-------names of the important predictors
	important_predictor_names = feature_imp.index[0:important_features]
	#-------subset the data to get only the important predictors and the response
	resp=pd.DataFrame(data=resp,columns=[resp_var])
	predictors=pd.DataFrame(prds,columns=list(predictors))
	dataset=pd.concat([resp,predictors],axis=1)
	#---------------------------------------------------------
	#----MODEL TRAINING
	#--------Remove the response variables from the features variables - axis 1 refers to the columns
	m_data= dataset.drop(resp_var, axis = 1,inplace=False) 
	# Response variables are the values we want to predict
	resp_var = np.array(dataset[resp_var])

	dataset = pd.get_dummies(m_data)
    
	# Saving feature names for later use
	feature_list = list(m_data.columns)
	# Convert to numpy array
	dataset = np.array(dataset)

	# Split the data into training and testing sets
	train_features, test_features, train_labels, test_labels = train_test_split(dataset, resp_var, test_size = size_of_test_data, random_state = 402)

	# Instantiate model with n_estimators decision trees
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=13563)

	# Train the model on training data
	clf.fit(train_features, train_labels)
    # evaluation
	predicted = clf.predict(test_features)
	pred_prob = clf.predict_proba(test_features)
    
	accuracy = accuracy_score(test_labels, predicted)
	#confusion matrix
	cnf = (confusion_matrix(test_labels,predicted))
	#precision score
	precision = precision_score(test_labels,predicted,pos_label=positive_class)
	#avg pres
	avg_precision = average_precision_score(test_labels,pred_prob[:,[1]])
	#recall score
	rec = recall_score(test_labels,predicted,pos_label=positive_class)
	#f1 scorea
	fscore = f1_score(test_labels,predicted,pos_label=positive_class)
	#fbeta score
	fbeta = fbeta_score(test_labels,predicted,beta=0.5)
	#hamming_loss
	hamming = hamming_loss(test_labels,predicted)
	#jaccard similarity score
	jaccard = jaccard_similarity_score(test_labels,predicted)
	#logloss
	logloss = log_loss(test_labels,predicted)
	#zero-oneloss
	zero_one = zero_one_loss(test_labels,predicted)
	#auc roc 
	area_under_roc = roc_auc_score(test_labels,pred_prob[:,[1]])
	#cohen_score
	cohen = cohen_kappa_score(test_labels,predicted)
	#mathews corr
	mathews = matthews_corrcoef(test_labels,predicted)
	# Variable importances from the important features selection stage
	variable_importance_list = list(zip(prds, feature_imp))
	output={"accuracy":accuracy,"precision":precision,"average precision":avg_precision,"recall":rec,"fscore":fscore,"fbeta":fbeta,"hamming":hamming,"jaccard":jaccard,"logloss":logloss,"zero_one":zero_one,"area_under_roc":area_under_roc,"cohen":cohen,"mathews":mathews}
	output=json.dumps(output)
	return jsonify({"Prediction": output})


#prediction end point
@predictmodelrf.route("/runModel", methods=['POST'])
def runModel(): 
	dataset = request.get_json().get('dataset')
	resp_var = request.get_json().get('resp_var')
	size_of_test_data = request.get_json().get('ratio_id')
	positive_class = request.get_json().get('positive_class')   
	n_estimators= request.get_json().get('n_estimators')
	important_features= request.get_json().get('important_features')
	dealing_with_nulls = request.get_json().get('dealing_with_nulls')
	result = runns(resp_var, size_of_test_data,dataset,positive_class,n_estimators,important_features, dealing_with_nulls)
	return result

if __name__ == '__main__':
	predictmodelrf.run(port=8082, debug=True)