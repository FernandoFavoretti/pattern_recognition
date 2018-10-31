# import das libs necessárias
import pandas as pd # trabalhar com dataframes
import numpy as np # realizacao de algumas operacoes com matrizes

#imagens
import cv2 # transformacoes faceis em imagens
from PIL import Image # trabalhar com imagens

# ferramentas
import glob # exploracao de diretorios
from pylab import *
import tqdm

def get_sample(sample_size = 10000, IMG_SIZE = 400, random_state=42):
	#cria uma amostra estratificada de todas as imagens na pasta train 
	catPaths = pd.Series(glob.glob(r"../data/train/cat*")).sample(n=int(sample_size/2), random_state=42)
	dogPaths = pd.Series(glob.glob(r"../data/train/dog*")).sample(n=int(sample_size/2), random_state=42)

	samplePaths = list(catPaths) + list(dogPaths)

	# itera por cada imagem  adicionando a classe de acordo com o nome da img
	X = []
	Y = []
	for img in tqdm.tqdm(samplePaths):
	    img_data = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
	    img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
	    X.append([np.array(img_data)])
	    Y.append(0 if "cat" in img else 1)

	# prepara dataframe com img e classe
	X_train = np.array([i[0] for i in X]).reshape(-1, IMG_SIZE, IMG_SIZE)
	X_train = np.array([i.flatten() for i in X_train])

	# Prepara Dataframes
	X_train = pd.DataFrame(X_train)
	Y_train = pd.DataFrame(Y)

	return X_train,Y_train