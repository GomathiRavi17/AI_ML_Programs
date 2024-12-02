#df.head() and df.info are optional, it's just to visualize the data after changes

import pandas as pd

#for preprocessing and modelling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#reading the csv data
df = pd.read_csv('breast-cancer-data.csv')

#shows the first 5 entries of the dataframe
df.head()

#Gives the Information about the Dataframe (like Column names, Datatype of each column, etc..)
df.info()

#removing unnecessary columns directly in the dataframe
#axis = 1 means we are deleting the data column-wise
#passing inplace as true allows us to make the changes directly in the existing dataframe
#default inplace value is false which means it always returns the new dataframe with the specified changes
df.drop(['id','Unnamed: 32'],axis=1,inplace=True)

#viewing the data after dropping the columns
df.head()

#viewing the table info after dropping the columns
df.info()

#Storing the input data (independent features) in the variable x after dropping the output variable (diagnosis)
#here we have not used inplace attribute, so it returns a new dataframe after dropping diagnosis column
x = df.drop('diagnosis',axis=1)
x.head()

x.columns

#Storing the target/output variable in y
y = df['diagnosis']
y.head()

# converting categorical values into numerical values
y = y.map({'M': 1, 'B': 0})
y.head()

#checking if there is any nan or missing values
#if present replace those values with mean of that columns
if x.isnull().sum().any():
  print("Replacing missing values with mean: ", x.mean())
  x.fillna(x.mean(),inplace=True)

#performing standardization on the data
#this process is done inorder to convert the values in different units(like radius in meter, perimeter in cm,etc) to a standard value
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#viewing the input data after performing standardization
x_scaled

#Principal component analysis is done to reduce the number of features
#if there is more features, it will lead to less accuracy
#So, we are using pca to select the required features for training the model
pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x_scaled)

#Splitting the original dataset for training and testing
#x_pca - input data after performing pca
#y - output data
#random state - tells how much randomly we need to split the original dataset into training and testing
#test_size - used to specify how much percentage of the original dataset is allocated for testing. here it is 25%
#stratify - used to distribute the target classes among the training and testing dataset equally inorder to have a balanced dataset
X_train, X_test, y_train, y_test = train_test_split(x_pca,y,random_state=42,stratify=y,test_size=0.25)

#printing the shape of each dataset
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#creating svc model object with linear kernel
svc = SVC(kernel="linear")
#training the model with input dataset
svc.fit(X_train,y_train)

#predicting the output by providing the test data
y_pred = svc.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred,target_names=['Malignant','Benign']))

print(round(accuracy_score(y_test,y_pred),2))