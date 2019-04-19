import json
import plotly
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    print(genre_names)

    #data for the third bar chart
    df_new= df.drop(columns=['id','message','original','genre'])
    type_names = df_new.sum().sort_values(ascending=False).index
    type_counts = df_new.sum().sort_values(ascending=False).values


    #data for the third bar chart
    word_serie= pd.Series(np.concatenate([x.split() for x in df['message']])).str.lower()
    word_serie=word_serie[~word_serie.isin(stopwords.words("english"))].value_counts()

    word_count= word_serie.sort_values(ascending=False)[:9]
    word_name = word_serie.sort_values(ascending=False)[:9].index


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker ={
                        'color':'rgba(50, 171, 96, 0.6)'
                    }
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=type_counts,
                    y=type_names,
                    orientation="h",
                    marker ={
                        'color':'rgba(50, 171, 96, 0.6)'
                    }
                )
            ],

            'layout': {
                'title': 'Distribution of categories',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': "Frecuency of categories"
                },
                'margin': {
                            'l':200,
                            'r':20,
                            't':70,
                            'b':70,

                },
            }
        },

        {
            'data': [
                Bar(
                    x=word_name,
                    y=word_count,
                    marker ={
                        'color':'rgba(50, 171, 96, 0.6)'
                    }

                )
            ],

            'layout': {
                'title': 'TOP 9 Words Distribution',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Frecuency of words"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
