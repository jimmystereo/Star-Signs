import pandas as pd

df3 = pd.read_csv('dataset/okcupid_profiles.csv')
df3

df3['sign'].value_counts()
df3['sign_clean'] = df3['sign'].apply(lambda x: str(x).split(' ')[0])
df3[df3['sign_clean'] != 'nan']['sign_clean'].dropna().value_counts().plot(kind='bar')

df3[df3['sign_clean'] == 'nan']

import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df3[['sign', 'essay0']]


def load_data(file_name, col_name):
    """
    Read in input file and load data

    root_dir: a path for data directory
    datafile: a text file for saving output

    return: X and y dataframe
    """
    output_file = open('output.txt', 'a')

    df = pd.read_csv(file_name)
    print("\n********** Data Summary **********\n")
    print(df.shape, "\n")
    print(df.head(3), "\n")
    print(df.info(), "\n")

    print("\n********** Data Summary **********\n", file=output_file)
    print(df.shape, "\n", file=output_file)
    print(df.head(3), "\n", file=output_file)
    print(df.info(), "\n", file=output_file)
    ## Remove duplicates if any and keep first occurrence
    # df.drop_duplicates(subset=['pmid'], keep='first', inplace=True)

    print("\n********** Data Shape after Removing Duplicates **********\n")
    print(df.shape, "\n")

    print("\n********** Data Shape after Removing Duplicates **********\n", file=output_file)
    print(df.shape, "\n", file=output_file)

    # if col_name == 'mix':
    #     df['mix'] = df['title'] + df['abstract']
    ## clean the sign column
    df['sign'] = df['sign'].apply(lambda x: str(x).split(' ')[0])
    df = df[df['sign'] != 'nan']

    df = df[['sign', col_name]]
    ## Check if any columns contain null values
    print("\n********** Count of Null Values for Each Column **********\n")
    print(df.isnull().sum(), "\n")

    print("\n********** Count of Null Values for Each Column **********\n", file=output_file)
    print(df.isnull().sum(), "\n", file=output_file)

    ## Drop instances including null values
    df = df.dropna()

    print("\n********** Data Shape after Removing Null Values **********\n")
    print(df.shape, "\n")

    print("\n********** Data Shape after Removing Null Values **********\n", file=output_file)
    print(df.shape, "\n", file=output_file)

    print("\n********** Class Label Distribution **********\n")
    print(df["sign"].value_counts())

    print("\n********** Class Label Distribution **********\n", file=output_file)
    print(df["sign"].value_counts(), file=output_file)
    ## Trim unnecessary spaces for strings
    df[col_name] = df[col_name].apply(lambda x: str(x).strip())

    ## Split into X and y (target)
    X, y = df.loc[:, col_name], df.loc[:, 'sign']
    output_file.close()
    return X, y


# Insert your code here

def preprocess_data(data):
    """
       Preprocess data with lowercase conversion, punctuation removal, tokenization, stemming

       X_data: X data in dataframe
       return: transformed dataframe

    """
    output_file = open('output.txt', 'a')

    print("\n\n********** Pre-processing Data **********\n")
    print("\n\n********** Pre-processing Data **********\n", file=output_file)

    ## Make sure that data type is string
    X_data = data.astype(str)

    ## 1. convert all characters to lowercase
    X_data = X_data.map(lambda x: x.lower())

    ## 2. remove punctuation
    X_data = X_data.str.replace('[^\w\s]', '')

    ## 3. tokenize sentence
    X_data = X_data.apply(nltk.word_tokenize)

    ## 4. remove stopwords
    stopword_list = stopwords.words("english")
    X_data = X_data.apply(lambda x: [word for word in x if word not in stopword_list])

    ## 5. stemming
    stemmer = PorterStemmer()
    X_data = X_data.apply(lambda x: [stemmer.stem(y) for y in x])

    ## 6. removing unnecessary space
    X_data = X_data.apply(lambda x: " ".join(x))
    output_file.close()

    return X_data


# Insert your code here

def split_data(X_data, y_data):
    output_file = open('output.txt', 'a')

    ## 1. Split data

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=5, stratify=y_data)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=5, stratify=y_test)

    ## 2. Reset index

    # Train Data
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Validation Data
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # Test Data
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\n********** Data Shape after Splitting **********\n")
    print("\nX_train: ", X_train.shape)
    print("\nX_val: ", X_val.shape)
    print("\nX_test: ", X_test.shape)

    print("\n********** Data View after Splitting **********\n")
    print("\nX_train:\n", X_train.head(3))
    print("\nX_val:\n", X_val.head(3))
    print("\nX_test:\n", X_test.head(3))

    print("\n********** Data Shape after Splitting **********\n", file=output_file)
    print("\nX_train: ", X_train.shape, file=output_file)
    print("\nX_val: ", X_val.shape, file=output_file)
    print("\nX_test: ", X_test.shape, file=output_file)

    print("\n********** Data View after Splitting **********\n", file=output_file)
    print("\nX_train:\n", X_train.head(3), file=output_file)
    print("\nX_val:\n", X_val.head(3), file=output_file)
    print("\nX_test:\n", X_test.head(3), file=output_file)
    output_file.close()

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def evaluate_model(y_true, y_pred):
    output_file = open('output.txt', 'a')

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['RCT', 'Non-RCT']))

    print(confusion_matrix(y_true, y_pred), file=output_file)
    print(classification_report(y_true, y_pred, target_names=['RCT', 'Non-RCT']), file=output_file)
    output_file.close()


def fit_model(X, y, modelname):
    """
    This function return a fitted model based on the provided X and y data, and the desired model type
    Model options:

    Decision Tree
    Logisitic regression
    Support Vector Machines
    Random Forest
    """
    if modelname == "DecisionTree":

        model = DecisionTreeClassifier(random_state=0)
    elif modelname == "SVM":

        model = SVC(random_state=0)
    elif modelname == "Logisitic":

        model = LogisticRegression(random_state=0)
    elif modelname == "RandomForest":

        model = RandomForestClassifier(random_state=0)
    model.fit(X, y)
    return model


import sys
from contextlib import redirect_stdout
from sklearn.feature_extraction.text import TfidfVectorizer

X, y = load_data('dataset/okcupid_profiles.csv', 'essay0')

Xp = preprocess_data(X)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(Xp, y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = fit_model(X_train, y_train, 'SVM')
y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred)
