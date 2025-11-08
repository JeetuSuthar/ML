
pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("diabetes.csv")
print(data.head())

x=data.drop("Outcome",axis=1)
y=data["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)

cm=confusion_matrix(y_test,y_pred_knn)
print(cm)
acc=accuracy_score(y_test,y_pred_knn)
print(acc)
error_rate=1-acc
print(error_rate)
print(precision_score(y_test,y_pred_knn))
print(recall_score(y_test,y_pred_knn))

tn,fp,fn,tp=cm.ravel()
specificity=tn/(tn+fp)
print(f"Specificity: {specificity:.4f}")
plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,
            xticklabels=['p no(0) ','p yes(1)'],
            yticklabels=['a no(0)','a yes(1)'])
plt.title("CM for knn diab dataset")
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

