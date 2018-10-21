# import das libs necessarias
import pandas as pd # trabalhar com dataframes
import numpy as np # realizacao de algumas operacoes com matrizes

#imagens
import cv2 # transformacoes faceis em imagens
from PIL import Image # trabalhar com imagens

# ferramentas
import glob # exploracao de diretorios
from pylab import *

# plot 
import matplotlib.pyplot as plt # plotagem
get_ipython().run_line_magic('matplotlib', 'inline')

# ML
from sklearn.decomposition import PCA #PCA sklearn

class PCA_and_Inverse():
    """docstring for PCA_and_inverse"""
    def __init__(self):
        self.pca_transformation = None
        self.pca_components = None

    def get_PCA(self, X, n_components=64):
        # Definimos quantas componentes gostariamos de utilizar 
        self.pca_img = PCA(n_components)

        # Fitamos o PCA e aplicamos a matriz X_train
        self.gray_img_pca = self.pca_img.fit_transform(X)

        return self.gray_img_pca
        

    def get_inverse_pca():
        # Vamos tentar agora reconstruir uma imagem reduzida
        # uma vez que reduzimos o tamanho de cada imagem para o numero de componetes principais
        # vamos analisar o que acontece se expandirmos de volta, apenas dando um reshape sem reconstruir de fato
        # a imagem
        X_inv_proj = self.pca_img.inverse_transform(self.gray_img_pca)
        X_proj_img = np.reshape(X_inv_proj,(1001,400,400))

        # plotando a imagem reconstruida com o pca
        fig = plt.figure(figsize=(8,8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for i in range(10):
            ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
            ax.imshow(X_proj_img[i], cmap=plt.cm.bone, interpolation='nearest')

