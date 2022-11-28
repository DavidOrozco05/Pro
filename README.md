# Proyecto-Final-IA
{
 "cells": [
  {
   "cell_type": "markdown",
   "source": [
    "### David Orozco\r\n",
    "# Proyecto IA"
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Clasificadores: Logistic Regression, SVM, KNN.\r\n"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "source": [
    "from sklearn.model_selection import train_test_split\r\n",
    "import pandas as pd\r\n",
    "import numpy as np\r\n",
    "import matplotlib.pyplot as plt\r\n",
    "import csv\r\n",
    "from sklearn import metrics\r\n",
    "from sklearn.decomposition import PCA\r\n",
    "\r\n",
    "from sklearn.neighbors import KNeighborsClassifier\r\n",
    "from sklearn.svm import SVC\r\n",
    "from sklearn.linear_model import LogisticRegression\r\n",
    "\r\n",
    "from sklearn.preprocessing import minmax_scale\r\n",
    "from sklearn.preprocessing import StandardScaler\r\n",
    "\r\n",
    "from sklearn.metrics import classification_report\r\n",
    "from sklearn.metrics import confusion_matrix\r\n",
    "from sklearn.metrics import roc_auc_score #***AUC-ROC***\r\n",
    "from sklearn.metrics import matthews_corrcoef #***MCC***\r\n",
    "from sklearn.metrics import f1_score #***F1***\r\n",
    "from sklearn.metrics import plot_roc_curve"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "source": [
    "dataset = pd.read_csv('train.csv') #Se extraen los datos \r\n",
    "\r\n",
    "#Procesamiento de los datos\r\n",
    "dataset = dataset.dropna() #Quita las filas con valores NaN, que segun Kaggle es de un procentaje de 2%\r\n",
    "dataset = dataset.drop(['Artist Name'], axis =1) #Quita la columna del nombre del artista\r\n",
    "dataset = dataset.drop(['Track Name'], axis=1) #Quita la columna del nombre de la cancion\r\n",
    "\r\n",
    "x = dataset.drop([\"Class\"], axis = 1).values\r\n",
    "\r\n",
    "y = dataset[\"Class\"].values\r\n",
    "\r\n",
    "scaler = StandardScaler()\r\n",
    "scaler.fit(x)\r\n",
    "x= scaler.fit_transform(x)\r\n",
    "\r\n",
    "#PCA\r\n",
    "pca = PCA(n_components=14, svd_solver='full')\r\n",
    "pca.fit(x)\r\n",
    "#print(pca.explained_variance_ratio_)\r\n",
    "\r\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)\r\n",
    "\r\n",
    "x_train = pca.fit_transform(x_train)\r\n",
    "x_test = pca.fit_transform(x_test)"
   ],
   "outputs": [],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "## Primer metodo: Logistic Regression"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "source": [
    "logistic_regression = LogisticRegression(solver='newton-cg', max_iter=1000, multi_class='multinomial')  #Se crea el objeto de logistic regression\r\n",
    "#solver{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, \r\n",
    "logistic_regression.fit(x_train,y_train) #Se entrena el clasificador \r\n",
    "\r\n",
    "y_pred_lr= logistic_regression.predict(x_test) #Se predice con las caracteristias de la parte test\r\n",
    "y_predproba_lr= logistic_regression.predict_proba(x_test) #Se predice las probabilidades con las caracteristias de la parte test\r\n",
    "\r\n",
    "mcc_lr = metrics.matthews_corrcoef(y_test, y_pred_lr) #Coeficiente de correlacion de matthews\r\n",
    "#roc_auc_lr = metrics.roc_auc_score(y_test, y_pred_lr, multi_class='ovr') #Roc auc\r\n",
    "f1_lr = f1_score(y_test, y_pred_lr, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn') #F1 score\r\n",
    "\r\n",
    "\"\"\"\r\n",
    "roc = {label: [] for label in multi_class_series.unique()}\r\n",
    "for label in multi_class_series.unique():\r\n",
    "    logistic_regression.fit(x_train, y_train == label)\r\n",
    "    predictions_proba = logistic_regression.predict_proba(x_test)\r\n",
    "    roc[label] += roc_auc_score(y_test, predictions_proba[:,1]) \"\"\"\r\n",
    "\r\n",
    "print(\"MCC Logistic Regression: \", mcc_lr)\r\n",
    "#print(\"Roc_auc score: \", roc_auc_lr)\r\n",
    "print(\"F1 score Logistic Regression: \", f1_lr)"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "MCC Logistic Regression:  0.2700127837497189\n",
      "F1 score Logistic Regression:  0.4028776978417266\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "## Segundo metodo: SVM"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "source": [
    "c_svc=10\r\n",
    "kernel= 'rbf' #Kernel = linear, poly, rbf(default), sigmoid\r\n",
    "svc = SVC(C=c_svc, kernel=kernel, gamma=0.01) \r\n",
    "svc.fit(x_train, y_train) #Se entrena el clasificador \r\n",
    "y_pred_svc = svc.predict(x_test) #Se predice con las caracteristias de la parte test\r\n",
    "\r\n",
    "mcc_svc = metrics.matthews_corrcoef(y_test, y_pred_svc) #Coeficiente de correlacion de matthews\r\n",
    "#roc_auc_knn = metrics.roc_auc_score(y_test, y_pred_knn, multi_class='ovr') #Roc auc\r\n",
    "f1_svc = f1_score(y_test, y_pred_svc, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn') #F1 score\r\n",
    "\r\n",
    "\r\n",
    "print(\"MCC Logistic Regression: \", mcc_svc)\r\n",
    "#print(\"Roc_auc score: \", roc_auc_knn)\r\n",
    "print(\"F1 score Logistic Regression: \", f1_svc)\r\n"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "MCC Logistic Regression:  0.2505451525138137\n",
      "F1 score Logistic Regression:  0.39272111722386804\n"
     ]
    }
   ],
   "metadata": {}
  },
  {
   "cell_type": "markdown",
   "source": [
    "## Tercer metodo: KNN"
   ],
   "metadata": {}
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "source": [
    "n_neighbors = 75\r\n",
    "weights_knn = 'uniform' #weights{uniform, distance} \r\n",
    "distance_knn ='euclidean' #Tipos de distancias euclidean. manhattan. chebyshev. minkowski. wminkowski. seuclidean. mahalanobis. hamming\r\n",
    "knn = KNeighborsClassifier(n_neighbors,weights=weights_knn,metric=distance_knn, metric_params=None,algorithm='brute') #Se crea el objeto de knn\r\n",
    "knn.fit(x_train, y_train) #Se entrena el clasificador \r\n",
    "\r\n",
    "y_pred_knn = knn.predict(x_test)  #Se predice con las caracteristias de la parte test\r\n",
    "\r\n",
    "\"\"\"\r\n",
    "cfsion_mtx_knn = metrics.confusion_matrix(y_test, y_pred_knn) #Matriz de confusion\r\n",
    "TP_knn = cfsion_mtx_knn[1, 1]\r\n",
    "TN_knn = cfsion_mtx_knn[0, 0]\r\n",
    "FP_knn = cfsion_mtx_knn[0, 1]\r\n",
    "FN_knn = cfsion_mtx_knn[1, 0] \"\"\"\r\n",
    "\r\n",
    "mcc_knn = metrics.matthews_corrcoef(y_test, y_pred_knn) #Coeficiente de correlacion de matthews\r\n",
    "#roc_auc_knn = metrics.roc_auc_score(y_test, y_pred_knn, multi_class='ovr') #Roc auc\r\n",
    "f1_knn = f1_score(y_test, y_pred_knn, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn') #F1 score\r\n",
    "\r\n",
    "\r\n",
    "print(\"MCC Logistic Regression: \", mcc_knn)\r\n",
    "#print(\"Roc_auc score: \", roc_auc_knn)\r\n",
    "print(\"F1 score Logistic Regression: \", f1_knn)"
   ],
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "MCC Logistic Regression:  0.2827439179658535\n",
      "F1 score Logistic Regression:  0.4168429961912823\n"
      
     ]
    }
   ],
   "metadata": {}
  }
 ],
 "metadata": {
  "orig_nbformat": 4,
  "language_info": {
   "name": "python",
   "version": "3.8.8",
   "mimetype": "text/x-python",
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "pygments_lexer": "ipython3",
   "nbconvert_exporter": "python",
   "file_extension": ".py"
  },
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3.8.8 64-bit (conda)"
  },
  "interpreter": {
   "hash": "b3ba2566441a7c06988d0923437866b63cedc61552a5af99d1f4fb67d367b25f"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
