# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Function to load the sql database and return the variables needed for the model.
    
    Args: 
    The sql database consisting of the messages and the corresponding categories.
    		
    Returns: 
    The varibles used for model fitting as well as the category names.
    	
    """
    
    # load data from database
    db_file_name = ['sqlite:///', database_filepath]
    engine = create_engine(''.join(db_file_name))
    print(''.join(db_file_name))
    df = pd.read_sql("SELECT * FROM TableToUse", engine)
    X = df['message']
    category_names = ['id','message', 'original','genre']
    Y = df.drop(category_names, axis=1)
    
    return X,Y, category_names


def tokenize(text):
    """
    Function to convert input text into tokens.
    
    Args: 
    Input text.
    		
    Returns: 
    The corresponding tokens.
    	
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]

    return tokens

def build_model():
    """
    Function to build the ML pipeline and perform grid search to obtain optimized parameters for the pipeline.
    

    Returns: 
    The ML pipeline.
    	
    """
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ]))
    ])),

    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Function that predicts the categories on the test messages.
    
    Args: 
    The ML pipeline, the test messages, the actual categories and the category names.
    	
    """
    
    Y_pred = pd.DataFrame(model.predict(X_test), columns = Y_test.columns)

    labels = category_names
    for col_no in range(len(labels)):
        print(classification_report(Y_test.iloc[:,col_no], Y_pred.iloc[:, col_no]))

def save_model(model, model_filepath):
    """
    Function to save the ML model.
    
    Args: 
    The ML pipeline and the path to save the model as a pickle file.
    		
    	
    """
    
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


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