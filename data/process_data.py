
# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    categories1 = df.categories
    categories2 = categories1.str.split(pat=';', n=-1, expand=True)
     
    # select the first row of the categories dataframe
    row = categories2.loc[0,:]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories2.columns = category_colnames
    
    for column in categories2:
        # set each value to be the last character of the string
        categories2[column] = categories2[column].str[-1]
        # convert column from string to numeric
        
        #categories2[column] = categories2[column].astype("Int64")
        
    categories2['id'] = df['id']
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df2 = df.merge(categories2)
    
    return df2



def save_data(df, database_filename):
    db_file_name = ['sqlite:///', database_filename]
    engine = create_engine(''.join(db_file_name))
    df.to_sql('TableToUse', engine, index=False, if_exists='replace') 
    


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