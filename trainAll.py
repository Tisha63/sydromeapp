import pandas as pd
df=pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
diseases=['drug reaction','allergy','common cold', 'chickenpox', 
          'neonatal jaundice', 'pneumonia', 'infectious gastroenteritis']
df = df[df['diseases'].isin(diseases)]
colums=df.columns
X=df.loc[:,colums[1:]]
y=df.loc[:,colums[0]]
#array Conver
X=X.to_numpy()

#spilit data

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
estimator=RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=25)
selector = selector.fit(X, y)
selected_features = df.loc[:,colums[1:]].columns[selector.support_]
X=df.loc[:,selected_features]
X=X.to_numpy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

import warnings
warnings.filterwarnings("ignore")
names = ["K-Nearest Neighbors","Decision Tree", "Random Forest",
         "Naive Bayes","ExtraTreesClassifier"]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    ExtraTreesClassifier(),
    ]

clfF=[]
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(name)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('--------------------------------------------------------------')
    clfF.append(clf)

import pickle
import bz2
sfile = bz2.BZ2File("model.pkl", 'wb')
pickle.dump(clfF, sfile)  
sfile1 = bz2.BZ2File("features.pkl", 'wb')
pickle.dump(selected_features, sfile1)  