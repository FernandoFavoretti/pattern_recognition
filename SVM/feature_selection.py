# import das libs necessárias
import pandas as pd # trabalhar com dataframes
import numpy as np # realizacao de algumas operacoes com matrizes

#imagens
import cv2 # transformacoes faceis em imagens
from PIL import Image # trabalhar com imagens

# ferramentas
import glob # exploracao de diretorios
from pylab import *
import random
from importlib import reload

# plot 
import matplotlib.pyplot as plt # plotagem
get_ipython().run_line_magic('matplotlib', 'inline')

# Machine Learning
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

#importando ferramentas já implementadas anteriormente
import feature_selection_GA as GA
reload(GA)

def get_MIC_features(X, Y, n_features=None):

	#O cálculo da informação mútua é realizado ao chamarmos o método implementado no pacote SKLearn
	MIC = feature_selection.mutual_info_classif(X=X, y=Y)

	#A resposta é convertida para um dataframe para facilitar a visualização e manipulação
	mic_df = pd.DataFrame(MIC)

	# Ao ordenarmos todos os elementos temos aqueles que nos trazem a maior informação sobre a variável resposta.
	sorted_mic_df = mic_df.sort_values(by=0, ascending=False)

	if n_features:
		return list(sorted_mic_df.index[:n_features])
	else:
		return sorted_mic_df

def get_GA_features(X, Y, verbose=True, test_size = 0.25, random_state = 42, pop_size = 10, cols_per_pop = 1600,
					num_generations = 1000, num_parents_mating = 4):
	# Precisamos criar um classificador simples para poder otimizar com o GA, vamos iniciar ocm um simples random forest como baseline

	# Divide em treino e teste
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
	if verbose:
		print('Training Features Shape:', X_train.shape)
		print('Training Labels Shape:', y_train.shape)
		print('Testing Features Shape:', X_test.shape)
		print('Testing Labels Shape:', y_test.shape)

	# Criando classificador baseline
	rf = RandomForestClassifier(n_estimators = 7, random_state = 42)
	rf.fit(X_train, np.ravel(y_train));

	# Fazendo predicoes baseline
	preds = rf.predict(X_test)
	score = roc_auc_score(y_test, preds)
	if verbose:
		print ("Baseline AUC {}".format(score))


	# Algoritmo Genetico
	# Vamos utilizar algoritmo genetico para trabalhar as features de X tentando otimizar a AUC tentando seleciona-las

	# Setamos parametros para o algoritmo genetico
	pop_size = 10
	cols_per_pop = 1600
	num_generations = 1000
	num_parents_mating = 4

	# rodamos o algoritmo
	final_cols = GA.optimize_ga(X_train, y_train,
	                            X_test, y_test,
	                            pop_size,
	                            cols_per_pop,
	                            num_generations,
	                            num_parents_mating,
	                            verbose)

	# predicao final com as colunas escolhidas pelo algoritmo
	GA_X_train = X_train[final_cols]
	GA_X_test = X_test[final_cols]


	# Criando classificador baseado no algoritmo
	rf = RandomForestClassifier(n_estimators = 7, random_state = 42)
	rf.fit(GA_X_train, np.ravel(y_train));

	# Fazendo novas predicoes
	preds = rf.predict(GA_X_test)
	score = roc_auc_score(y_test, preds)
	if verbose:
		print ("Final AUC {}".format(score))

	return final_cols