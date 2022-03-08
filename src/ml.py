import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def linear_regression(X, y, x_train, x_test, y_train, y_test, dataset):
    print("------\nLinearRegression model :\n")
    # Entrainement du modèle par Régression linéaire
    linear_reg = LinearRegression().fit(x_train,y_train)
    print("Training score: {}".format(linear_reg.score(x_train,y_train)))
    print("Test score: {}".format(linear_reg.score(x_test,y_test)))

    k_fold_cross(linear_reg, X, y)
    evaluation(linear_reg,"Linear regression", dataset)


def svr(dataset):
    ## Comme ce modele a besoin de beaucoup de puissance pour etre mis en oeuvre, nous allons réduire l'apprentissage aux 10000 éléments les plus récents du dataset
    ## On recupere toutes les données du dataset sauf la colonne "HIGH"
    # Entrées
    X = np.array(dataset)
    X = X[-9999:]

    ## On recupere la colonne high
    # Labels
    y = np.array(dataset['High'])
    y = y[-10000:]
    y = np.delete(y,0)
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    print("------\nSupport Vector Regression model :\n")

    model = make_pipeline(StandardScaler(), SVR(kernel='poly', C=1.0, epsilon=0.2)).fit(x_train, y_train)

    print("Training score: {}".format(model.score(x_train,y_train)))
    print("Test score: {}".format(model.score(x_test,y_test)))

    k_fold_cross(model,X,y)

    evaluation(model,"Support Vector Regression",dataset, full=False)


def sgd_regressor(X, y, x_train, x_test, y_train, y_test, dataset):
    print("------\nSGDRegressor model :\n")
    model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=100)).fit(x_train,y_train) 
    print("Training score: {}".format(model.score(x_train,y_train)))
    print("Test score: {}".format(model.score(x_test,y_test)))

    k_fold_cross(model, X, y)
    evaluation(model,"SGDRegressor", dataset)

def gaussian_process_regressor(dataset):
    ## Comme ce modele a besoin de beaucoup de puissance pour etre mis en oeuvre, nous allons réduire l'apprentissage aux 10000 éléments les plus récents du dataset
    ## On recupere toutes les données du dataset sauf la colonne "HIGH"
    # Entrées
    X = np.array(dataset)
    X = X[-9999:]

    ## On recupere la colonne high
    # Labels
    y = np.array(dataset['High'])
    y = y[-10000:]
    y = np.delete(y,0)
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    print("------\nGaussianProcessRegressor model :\n")
    kernel = DotProduct() + WhiteKernel()

    model = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel)).fit(x_train,y_train) 
    print("Training score: {}".format(model.score(x_train,y_train)))
    print("Test score: {}".format(model.score(x_test,y_test)))

    k_fold_cross(model, X, y)
    evaluation(model,"GaussianProcessRegressor", dataset, False)

def k_fold_cross(model, X, y): 
    nb_iter = 5
    # Mise en place du K-Fold Cross
    print("k-kold cross with {} sections :".format(str(nb_iter)))
    scores = cross_val_score(model, X, y, cv=nb_iter)
    print ("\nK-Fold cross scores : \n", scores)

def evaluation(model, model_name, dataset, full=True):
    if full:
        ## On essaye de tout prédire
        prediction_data = np.array(dataset)
    else:
        prediction_data = np.array(dataset)[-10000:]
        dataset = dataset[-10000:]
        
    prediction = np.array(model.predict(prediction_data))
    # Remplace toutes les valeurs inferieurs par 0
    prediction[prediction < 0] = 0
    
    # Erreur moyenne
    print("Mean Absolute Error : {}".format(mean_absolute_error(dataset['High'],prediction)))
    # Erreur maximale
    print("Maximum error : {}".format(max_error(dataset['High'], prediction)))

    plt.plot(prediction,'r', label="Predicted values")
    plt.plot(np.array(dataset['High']),'b',label="Targeted values")
    plt.suptitle("{} model evaluation".format(model_name))
    plt.legend(loc="upper left")
    plt.xlabel("Timestamp")
    plt.ylabel("Bitcoin High Value ($)")
    plt.show()


data = pd.read_csv("../bitstamp.csv",delimiter=',')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
print("Dataset description : \n\n"+str(data.describe().round(3)))

## On recupere toutes les données du dataset sauf la colonne "HIGH"
# Entrées
X = np.array(data)
X = X[:len(data)-31]

## On recupere la colonne high
# Labels
y = np.array(data['High'])
y = y[:-30]
y = np.delete(y,0)


## Split en données d'entrainement / test
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


linear_regression(X, y, x_train, x_test, y_train, y_test, data)
svr(data)
sgd_regressor(X, y, x_train, x_test, y_train, y_test, data)
gaussian_process_regressor(data)