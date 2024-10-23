# Disaster Prediction

## Project Overview

This  project focuses on analyzing disaster data to predict disaster categories based on input messages. By leveraging real messages sent during disaster events, we develop a model that classifies these messages for timely assistance. This project was done as a part of the [Udacity Data Scientist Nanodegree program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 

## Table of Contents

1. [Project Components](#project-components)
   - [ETL Pipeline](#1-etl-pipeline)
   - [ML Pipeline](#2-ml-pipeline)
   - [Flask Web App](#3-flask-web-app)
2. [Web App Preview](#web-app-preview)
3. [Instructions](#instructions)
4. [License](#license)

## Project Components

The project consists of three main components:

### 1. ETL Pipeline

The **ETL (Extract, Transform, Load)** process is implemented in the `process_data.py` script located in the `data` folder. This pipeline performs the following tasks:

- Loads datasets containing messages and categories
- Merges the datasets
- Cleans the data
- Stores the cleaned data in a SQLite database

### 2. ML Pipeline

The **Machine Learning** pipeline is implemented in the `train_classifier.py` script located in the `models` folder. This pipeline includes:

- Loading data from the SQLite database
- Splitting the dataset into training and testing sets
- Building a text processing and machine learning pipeline
- Training and tuning the model using `GridSearchCV`
- Outputting results on the test set
- Exporting the final model as a pickle file

### 3. Flask Web App

The Flask web application is launched using the `run.py` script located in the `app` folder. This app allows emergency workers to input messages and receive classification results across several disaster categories.

## Web App Preview

Here are a few screenshots of the web app:


Below are a few screenshots of the web app:

##### Home page of the app

- **Bar Chart of Messages per Genre:**

![Screenshot 1](data/images/disaster-response-project1.png?raw=true "Title")

- **Pie Chart of Message Proportions by Genre:**

![Screenshot 3](data/images/disaster-response-project3.png?raw=true "Title")


### Message Classification

When you enter a message and click "Classify Message," you receive results displayed in a user-friendly format:

![Screenshot 2](data/images/disaster-response-project2.png?raw=true "Title")


## Instructions

To set up the database and model, follow these steps:

1. Run the ETL pipeline to clean the data and store it in the database:
   ```bash
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```

2. Train the machine learning classifier and save the model:
   ```bash
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

3. Navigate to the `app` directory:
   ```bash
   cd app
   ```

4. Run the web app:
   ```bash
   python run.py
   ```

5. Open your web browser and type in a message to predict the disaster category.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
