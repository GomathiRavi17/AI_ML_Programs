import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv('iris-data.csv')
df.head()

df.tail()

df.shape

df['Species'].value_counts()

x = df.drop('Species',axis=1)
y = df['Species']

x.head()

y.head()

y = y.map({'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica': 2})
y.head()

y.tail()

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of the model is ' + str(round(accuracy, 2)) + ' %.')