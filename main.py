import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("dataset.csv")
df.info()
df.drop(['date','id'],axis=1,inplace=True)
x=df.drop(['flag'],axis=1)
y=df['flag']
from sklearn.preprocessing import StandardScaler as scaler
x_scaled=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of KNN classifier:', accuracy)
from sklearn.svm import SVC
svm=SVC(kernel='linear')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
accuracy1=accuracy_score(y_test,y_pred)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred1 = rf.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred1)
print('Accuracy:', accuracy2*100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred2 = dt.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred2)
print('Accuracy:', accuracy3)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred3 = lr.predict(X_test)
accuracy4 = accuracy_score(y_test, y_pred3)
print('Accuracy:', accuracy4)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train, y_train)
y_pred=NB.predict(X_test)
accuracy5=accuracy_score(y_test,y_pred)
print(accuracy5)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_train,y_train)
y_pred=kmeans.predict(X_test)
accuracy6=accuracy_score(y_test,y_pred)
accuracy6
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,random_state=100,max_features=5 )
gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)
accuracy7=accuracy_score(y_test,y_pred)
accuracy7
u=['KNN','SVM','RForest','DTree','Logistic','NaiveBias','kmeans','Gradientboster']
y=[accuracy,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,accuracy7]
plt.figure(figsize =(8.5, 5))
plt.xlabel('Algorithms')
plt.ylabel('accuracy')
plt.plot(u,y)
plt.figure(figsize=(9,5))
plt.bar(u,y)
lr.predict([[0.0995,0.137396,0.572,48,0.094495,6.595,0.065]])
lr.predict([[0.7575,0.778458,1.991,700,0.497389,51.366,0.201]])
y=gbc.predict([[0.485,0.432045,0.868,22,0.239146,9.505,0.072]])
if (y==0):
    print('Faithfull')
else:
    print('Unfaithfull')
import pickle
with open('svm.pkl', 'wb') as file:
    pickle.dump(svm, file)
from IPython.display import FileLink
# Create a download link for the model.pkl file
FileLink(r'svm.pkl')