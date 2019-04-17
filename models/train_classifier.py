# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix,f1_score
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import pickle
import sys

def load_data(database_filepath):

    '''
    Arg: database filepath where the data is stored

    Output: features, labels and categories names

    '''
    # define features and label arrays and load the data from the database

    df = pd.read_sql_table('InsertTableName',con=engine,)
    X = df['message'].values
    y = df.drop(['original','message','id','genre'], axis=1).values
    target_names = list(df.drop(['original','message','id','genre'],axis=1))

    return X, y, target_names


def tokenize(text):

    '''
    Arg:text to be tokenized

    Output: tokens
    '''

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words("english")]

    return tokens

def build_model():

    """
    build a model pipeline

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    return pipeline



def evaluate_model(model, X_test, y_test, category_names):

    """
    print a summary test score (F1, recall and precision ) for the 36 categories

    arg: model already fitted, X_test, y_test, and category_names

    """

    # output model test results
    y_pred=model.predict(X_test)
    print(classification_report(y_test, y_pred,target_names=category_names))


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
