# Disaster_Prediction

Udacity project on analyzing disaster data and predicting the disaster category from the input messages


Project Overview
In this project, we analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.

In the data folder, you'll find a data set containing real messages that were sent during disaster events. We create a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. 

Below are a few screenshots of the web app:

![Screenshot 1](data/images/disaster-response-project1.png?raw=true "Title")

The above image is the homepage of the app. When you enter a message and click on "classify message", you get a screen that looks like the below image.

![Screenshot 2](data/images/disaster-response-project2.png?raw=true "Title")

### Project Components:

There are three components in this project.

1. ETL Pipeline
In a Python script under the data folder, process_data.py is a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. ML Pipeline
In a Python script under the models folder, train_classifier.py is a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App
Under the app folder, there is a run.py python scipt that launches the app where you can input messages to be classified.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Open the webpage and type in your message to predict the disaster category.