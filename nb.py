
import sys
import csv
import pickle
import time
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


def case_1():
    print("hello")

    global file_str
    file_str = input("Enter the name of the data file\n")

    global data
    data = pd.read_csv(file_str, sep=',')

    print("Length: ", len(data))

    global header
    header = data.columns.tolist()
    #header.pop(0)

    X = data
    #X = data.drop('color', axis=1)
    y = data['color']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train, X_test = train_test_split(X, test_size=0.5, random_state=int(time.time()))

    global gnb
    gnb = GaussianNB()
    #gnb.fit(X_train["size", "act","age","inflated"].values, X_train["color"])
    gnb.fit(X,y)

    y_pred = gnb.predict(X_test)

    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["color"] != y_pred).sum(),
          100*(1-(X_test["color"] != y_pred).sum()/X_test.shape[0])
))

    print("Number of elements: ", len(y_pred))
    print("Number of 0: ", (len(y_pred)) - np.count_nonzero(y_pred))
    print("Number of 1: ", np.count_nonzero(y_pred))

    #global classifier
    #classifier = DecisionTreeClassifier()
    #classifier.fit(X_train, y_train)

    #y_pred = classifier.predict(X_test)
    #print("X_test")
    #print(X_test)

    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred



    main()



def case_2():

    nb_file = open(file_str + ".model", "wb")
    pickle.dump(gnb, nb_file)
    nb_file.close()
    print("Classifier saved")
    main()



def case_3():

    print("1: Enter new case\n")
    print("2: Quit\n")
    choice = input("Enter choice\n")

    if (choice == '2'):
        return

    elif (choice == '1'):

        item_values = list()
        for item in header:
            print("What is the value for ", str(item))
            value = input(" ")
            item_values.extend(value)


        my_array = np.asarray(item_values)
        #my_array.reshape(1,-1)
        #my_array = my_array.astype(np.float32)
        #my_array = pd.to_numeric(1)
        guess = gnb.predict([[int(my_array[0]),int(my_array[1]),int(my_array[2]),int(my_array[3]),int(my_array[4])]]);

        if (guess == 0):
            print("The balloon is yellow\n")
        if (guess == 1):
            print("The balloon is purple\n")


    main()


def case_4():

    model_name = input("Enter name of model file\n")
    model = open(model_name, "rb")
    print("Model opened\n")
    case_3()

def main():

    print("\n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Choose an option\n")
    print("1: Learn a naive bayesian classifier from training data\n")
    print("2: Save the classifier\n")
    print("3: Apply the classifier to new cases\n")
    print("4: Load a model saved previously and apply the model to new cases\n")
    print("5: Quit")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


    choice = input("Enter choice\n")

    if (choice == '1'):
        case_1()

    elif (choice == '2'):
        case_2()

    elif (choice == '3'):
        case_3()

    elif (choice == '4'):
        case_4()

    elif (choice == '5'):
        #sys.exit()
        return


main()
