#Author: Muhammed yassin Ahmed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn import datasets
np.set_printoptions(precision=4)

dataset =datasets.load_iris()

X = dataset.data
y = dataset.target
target_names = dataset.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features)) #(4,4)
        SB = np.zeros((n_features, n_features)) #(4,4)
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))
            #scatter between classes
            n_c = X_c.shape[0] #getting number of samples

            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        A = np.linalg.inv(SW).dot(SB)
        #print(np.linalg.inv(SW))
        #print(SB)
        #print(A)
        #print('')
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T  #trasposing the eignvestors for easier calculations
        idxs = np.argsort(abs(eigenvalues))[::-1] #sorting eignvalues from high to low
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0:self.n_components] # store first n eigenvectors

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)

def display(z,z1):
  lda=LDA(2)
  lda.fit(z,z1)
  X_projected=lda.transform(z)
  print('Shape of X:',z.shape)
  print('Shape of projected X:',X_projected.shape)
  plt.figure()
  colors = ['red', 'green', 'blue']
  for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_projected[z1 == i, 0], X_projected[z1 == i, 1], alpha=.8, color=color, label=target_name)
  print('')
  plt.legend(loc='lower center', shadow=False, scatterpoints=1)
  plt.colorbar()
  plt.show()

display(X,y)

display(X_train,y_train)

display(X_test,y_test)

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy : ' + str(accuracy_score(y_test, y_pred)*100))

Model1=LinearDiscriminantAnalysis()
Model1.fit(X,y)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#evaluate model
scores = cross_val_score(Model1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores)*100)
