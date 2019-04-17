import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    load the messages and categories files and merge both on 'id'
    Args:
        messages_filepath: Filepath to the messages dataset
        categories_filepath: Filepath to the categories dataset
    Returns:
        df: Merged Pandas dataframe
    """

    # read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #merge messages with categories
    df =  messages.merge(categories, on='id')

    return df

def clean_data(df):

    """
    it cleans the data frame to be saved and used for ML modeling
    Args:
        df: df to be cleaned

    Returns:
        df: cleaned dataframe
    """


    #create a dataframe with 36 categories
    categories = df.categories.str.split(pat=';',expand=True)

    #select the first row of the categories dataframe
    #and extract the column names
    row = categories.loc[0]
    category_colnames = list(row.apply(lambda x :x[:-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] =categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        #make sure there is onle 1 or 0
        categories[column] = categories[column].replace(2, 1)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):

    """
    it saves the data frame in the crated SQLite database
    Args:
        df: df to be save
        database_filename: name of the database

    """


    # load to database

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('clean_data', engine,if_exists="replace" ,index=False)




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
