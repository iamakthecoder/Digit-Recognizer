import csv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#read training data
train = pd.read_csv('train.csv')

#dataframe containing the training data features
train_features = train.drop(columns=['label'])

#dataframe containing the training data targets
train_target = train['label'].values

#KNN classifier/ create model
knn = KNeighborsClassifier(n_neighbors=3)
#fit the classifier to the data/ train the model
knn.fit(train_features,train_target)

#read testing data (features only)
test = pd.read_csv('test.csv')

#predictions by the model corresponding to the test data
prediction = knn.predict(test)


#store the predicitons in a csv file
with open('predictions_mnist.csv','w',newline="") as f:
    writer = csv.writer(f)

    header = ['ImageId','Label']
    writer.writerow(header)

    for i in range(len(prediction)):
        writer.writerow([i+1,prediction[i]])