import sys
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib as mpl
import os
import pickle
from flask import Flask, app, request, jsonify, render_template
from flask import Flask, request, render_template,jsonify

app = Flask(__name__)

def do_something(text1):
   url=text1
   '''print('Python: {}'.format(sys.version))
   print('Numpy: {}'.format(np.__version__))
   print('Pandas: {}'.format(pd.__version__))
   print('Sklearn: {}'.format(skl.__version__))
   print('Matplotlib: {}'.format(mpl.__version__))'''

   # Import the dataset
   # url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'
   names = ['Class', 'id', 'Sequence']
   df = pd.read_csv(url, names=names)
   print(df)
   # print("**************************************************************************")
   print(df.describe())
   print(
       "----------------------------------------------------------------------------------------------------------------")
   print(df.iloc[0])  # print the information of [0]
   # print("**************************************************************************")

   print(df['Sequence'].iloc[
             0])  # print the sequence[0] in this template 'tactagcaatacgcttgcgttcggtggttaagtatgtataat...', because usage of fun '.iloc'
   print(
       "----------------------------------------------------------------------------------------------------------------")

   # Preprocessing the dataset

   classes = df.loc[:, 'Class']
   print(classes)  # output is classes
   # print("**************************************************************************")

   # generate list of DNA sequences
   sequences = df.loc[:, 'Sequence']
   print(
       sequences)  # output is sequence in this template ' \t\ttactagcaatacgcttgcgttcggtggttaagtatgtataat...', because usage of fun '.loc'
   print(
       "----------------------------------------------------------------------------------------------------------------")

   dataset = {}
   i = 0

   # loop through sequences and split into individual nucleotides
   for seq in sequences:
       # split into nucleotides, remove tab characters
       nucleotides = list(seq)
       nucleotides = [x for x in seq if x != '\t']

       # append class assignment
       nucleotides.append(classes[i])

       # add to dataset
       dataset[i] = (nucleotides)

       # increment i
       i += 1
   # print("**************************************************************************")
   # print dataset lists without make lines between them .. split sequence nucleotides as elements of list and class as the last element of each list
   print("dataset[0]")
   print(dataset[0])
   # print("----------------------------------------------------------------------------------------------------------------")
   df = pd.DataFrame(dataset)
   # print(df)
   # print("**************************************************************************")
   # make transpose as matrices "make column rows and vice versa" +/- of classes type be as last column instade last row
   df = df.transpose()
   # print(df)
   # print("----------------------------------------------------------------------------------------------------------------")

   # for clarity, lets rename the last dataframe column to class
   df.rename(columns={57: 'Class'}, inplace=True)

   print(df)
   print("**************************************************************************")

   print(df.describe())
   print(
       "----------------------------------------------------------------------------------------------------------------")
   print(df)
   print("**************************************************************************")

   # Record value counts for each sequence
   series = []
   for name in df.columns:
       # take t,g,c,a,+,- and make number of freq as evry index for all sequences
       series.append(
           df[name].value_counts())  # class will be only in index 57 which is "class type" in last raw of the series

   info = pd.DataFrame(series)
   print("info")
   print(info)
   print(
       "----------------------------------------------------------------------------------------------------------------")
   details = info.transpose()
   print("details")
   print(details)
   print("**************************************************************************")

   # We can't run machine learning algorithms on the data in 'String' formats. We need to switch
   # it to numerical data.
   numerical_df = pd.get_dummies(
       df)  # mainplate all sequences by Convert categorical variable [t,c,g,a,+,-] into indicator variables[a0-a56 ,t0-t56 ,...., class+,class-].

   print("numerical_df")
   print(numerical_df)
   # We don't need both class columns.  Lets drop one then rename the other to simply 'Class'.
   df = numerical_df.drop(columns=['Class_-'])  # delete class -
   print("df")
   print(df)
   df.rename(columns={'Class_+': 'Class'},
             inplace=True)  # make class + is the base by which if data =+ then it belongs to + class else it for - immbededly
   print("df")
   print(df)
   print(df.iloc[:5])  # 0_a  0_c  0_g  0_t  1_a  1_c  1_g  ...  55_g  55_t  56_a  56_c  56_g  56_t  Class
   print(
       "----------------------------------------------------------------------------------------------------------------")

   # Use the model_selection module to separate training and testing datasets
   from sklearn import model_selection

   # Create X and Y datasets for training
   # for use as x train and x test for model_selection consist of N list for N column nucliotoid
   X = np.array(df.drop(['Class'], 1))
   print("X")
   print(X)
   print("**************************************************************************")

   # for use as y train and y test for model_selection consist of one list for one column class
   y = np.array(df['Class'])
   print("y")
   print(y)
   print(
       "----------------------------------------------------------------------------------------------------------------")
   # define seed for reproducibility
   seed = 1

   # split data into training and testing datasets
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20,
                                                                       random_state=seed)  # /// random seed
   print(X_train)
   print("**************************************************************************")
   print(X_test)
   print("**************************************************************************")
   print(y_train)
   print("**************************************************************************")
   print(y_test)
   print("**************************************************************************")

   # Now that we have our dataset, we can start building algorithms! We'll need to import each algorithm we plan on using
   # from sklearn.  We also need to import some performance metrics, such as accuracy_score and classification_report.

   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.neural_network import MLPClassifier
   from sklearn.gaussian_process import GaussianProcessClassifier
   from sklearn.gaussian_process.kernels import RBF
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
   from sklearn.naive_bayes import GaussianNB
   from sklearn.svm import SVC
   from sklearn.metrics import classification_report, accuracy_score

   # define scoring method
   scoring = 'accuracy'

   # Define models to train
   models = []

   models.append(('Nearest Neighbors', KNeighborsClassifier(n_neighbors=2)))
   models.append(('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0))))
   models.append(('Decision Tree', DecisionTreeClassifier(max_depth=5)))
   '''models.append(('Random Forest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
   models.append(('Neural Net', MLPClassifier(alpha=1, max_iter=400, warm_start=True, verbose=0)))
   models.append(('AdaBoost', AdaBoostClassifier()))
   models.append(('Naive Bayes', GaussianNB()))
   models.append(('SVM Linear', SVC(kernel='linear')))
   models.append(('SVM RBF', SVC(kernel='rbf')))
   models.append(('SVM Sigmoid', SVC(kernel='sigmoid')))
   models.append(('SVM Polynomial', SVC(kernel='poly')))'''

   # evaluate each model in turn
   results = []
   names = []

   for name, model in models:
       # split into 10 smaller sets generally follow the same principles.
       kfold = model_selection.KFold(n_splits=10)
       # scoring = 'accuracy'
       # compare different machine learning models and get sence how it will work with data
       # to make sence and concept and how it will match the prediction of the test curve
       # to know how to get y-test from x-test by the knowledge taken from this method with x-y train
       cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
       results.append(cv_results)
       names.append(name)
       # calculate the mean of data of the cross value ,, standerd deviation to know how dispersed the data is in relation to the mean
       msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
       print(msg)
       #return msg
       print(
           "----------------------------------------------------------------------------------------------------------------")
   print("****************************************************************************************************")

   from sklearn.metrics import classification_report, accuracy_score
   text = " "
   for name, model in models:
       model.fit(X_train, y_train)
       # guessed y by the machine learning techniques "models" after feeded by train data x,y
       predictions = model.predict(X_test)
       print(name)
       # prediction of y and real y compare and get accuracy
       print(accuracy_score(y_test, predictions))
       # accuracy : number of time in which the model predict the y test correctly  true prediction
       # ratio of the number of correct predictions to the total number of predictions (the number of test data points
       # True Positives + True Negaitve/(Total Data)     1 is the optimal
       # precision : How much the accuracy of correct decision taken -- It is a ratio of true positives(words classified as false, and which are actually false) to all positives(all words classified as false
       # [True Positives/(True Positives + False Positives)]
       # Recall(sensitivity): words that actually were spam ,were classified by us as spam.-- ratio of true positives(words classified as false, and which are actually false) to all the words that were actually false
       # [True Positives/(True Positives + False Negatives)]
       # f1-score:weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0 and lower than accuracy
       # F1 Scode = 2 * ( (*Precision* * *Recall*) / (*Precision* + *Recall*) )
       # support number of sample in each class
       print(classification_report(y_test, predictions))
       report=classification_report(y_test, predictions)
       image = Image.open(r'E:\pythonProject1\Color-white.jpg')
       draw = ImageDraw.Draw(image)
       # specified font size
       # font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20)
       #report.__add__(report+"\n")
       text = text.__add__("                                     "+ name + "\n\n\n")
       text = text.__add__(report + "\n\n\n")
       # drawing text size
       #draw.text((5, 5), text, fill="black", align="center")

       #image.show()
      # print((report))
       #return image.show()
   print( "----------------------------------------------------------------------------------------------------------------")
   font = ImageFont.truetype(r'G:\downloads\khayal Font New\Myriad Bold_1.ttf', 75)
   draw.text((250,250), text, fill="black",font=font, align="left")
   # image.show()
   # print((report))
   return image.show()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/join', methods=['GET','POST'])
def my_form_post():
    text1 = request.form['text1']
    word = request.args.get('text1')
    combine = do_something(text1)
    result = {
        "output": combine
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
