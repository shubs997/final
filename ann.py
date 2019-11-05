#importig libraries
from keras.models import Sequential 
from keras.layers import Dense 

import pandas as p

#importing the dataset
dataset = p.read_csv("diabetes.csv")

#separating the features and the outcomes
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,-1]

#spliting the dataset into training and testing 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#buliding the network architecture
model = Sequential()
model.add(Dense(12,input_dim =8 , activation ='relu'))
model.add(Dense(12,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

#compiling the model
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam', metrics=['accuracy'])

# training the model
model.fit(X_train,y_train,epochs = 200, validation_split = 0.1, batch_size = 10)

# evaluating on the test set
accuracy = model.evaluate(X_test,y_test)

print(accuracy)

predictions = model.predict_classes(X_test,verbose = 0)

cm = confusion_matrix(X_test,predictions)
