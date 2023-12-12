# BraniacProject

## Table of Contents

1. [Introduction](##introduction)
2. [Project Overview](#project-overview)
   - [File Structure](##file-Structure)
   - [Data Preprocessing](#data-preprocessing)
   - [Models](#models)
   - [Docker Setup](#docker-setup)
   - [FastAPI Application](#fastapi-application)
   - [Testing](#testing)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Usage](#usage)

## Introduction

Welcome to the Movie Rating Prediction project! This project is dedicated to predicting movie ratings based on various movie descriptors using advanced machine learning models. Our comprehensive approach encompasses data preprocessing, the development of two distinct models (Support Vector Regression and Neural Network), and the creation of a real-time prediction interface using FastAPI.

## File Structure

```plaintext
.github
├── workflows
│   └── data_preprocessor_workflow.yml - GitHub Actions workflow for automating the data preprocessing process.
│   └── preprocessor_extractor_workflow.yml - GitHub Actions workflow for extracting features using the preprocessor.
.docker_files
├── workflows
│   └── Dockerfile.fastapi - Dockerfile for building the FastAPI application container.
│   └── Dockerfile.mlmodel - Dockerfile for building the container for machine learning models.
│   └── Dockerfile.nnmodel - Dockerfile for building the container for the neural network model.
│   └── Dockerfile.pgadmin - Dockerfile for building the container for pgAdmin.
│   └── Dockerfile.postgres - Dockerfile for building the container for PostgreSQL.
│   └── Dockerfile.preprocessing - Dockerfile for building the container for the data preprocessing script.
│   └── init-db.sql - SQL script for initializing the PostgreSQL database.
src
├── data
│   └── data_preprocessor.py - Class for data preprocessing.
│   └── feature_extractor.py - Class for feature extraction.
│   └── helper.py - Helper functon for spliting data and scaling data.
├── dataset
│   └── data.csv - CSV file containing the dataset.
├── db
│   ├── base.py - Base SQLAlchemy model.
│   ├── dataloader.py - Data loader script.
│   ├── db_engine.py - Class for connecting and communicating with database.
│   ├── keyword.py - Model for handling keywords.
│   ├── production_company.py - Model for handling production companies.
│   └── production_country.py - Model for handling production countries.
├── models
│   ├── ml_model.py - Class for the machine learning model.
│   ├── model.py - Abstract model class.
│   ├── models_dict.py - Dictionary of available models.
│   └── nn_model.py - Class for the neural network model.
├── saved_models
│   └── nn_model.h5 - Saved weights of the neural network model.
│   └── svr_model.sav - Saved Support Vector Regression model.
├── main.py - Main script for running the FastAPI application.
├── ml_model_main.py - Main script for training and running the machine learning model.
├── nn_model_main.py - Main script for training and running the neural network model.
├── prepare_save_data.py - Script for preparing and saving data.
└── router.py - FastAPI router definition.
tests
├── test_data_preprocessor.py - Unit tests for the data preprocessing script.
└── test_preprocessor_extractor.py - Integrated tests for the feature extraction and data_preprocessor classes.
.env - Environment variables configuration file.
.env.docker - Environment variables for Docker.
dataPreprocessing.ipynb - Jupyter Notebook for data preprocessing with complete analysis.
docker-compose.yml - Docker Compose configuration file.
requirements.txt - List of project dependencies.
Zadatak.pdf - Project assignment document.
```

## Data Preprocessing

In this project, the data preprocessing pipeline plays a crucial role in preparing the dataset for training machine learning models to predict movie ratings. The focus is on three key columns: `keywords`, `production_companies`, and `production_countries`. Here's a detailed overview of the data preprocessing steps:

### Columns Selection

We concentrate on predicting movie ratings using three specific columns: `keywords`, `production_companies`, and `production_countries`. Thorough analysis has revealed that these columns significantly impact movie ratings, while other columns have been determined to have negligible influence.

### Transformation from JSON to Lists

The selected columns originally contain JSON-formatted data. To facilitate analysis and model training, each JSON entry is transformed into a list of strings. This list represents the various descriptors associated with each movie.

### Creating Lists of Unique Words

For each of the three columns, we compile a list of all unique words present across the dataset. These lists are sorted based on the frequency of appearance, with the most common words at the beginning.

### Binarization of Arrays

For each row in the dataset, binarized arrays are created for the three selected columns. This process involves mapping the presence of each word in the respective lists to a binary value (1 if present, 0 if not).

### Concentration Points Calculation

Concentration points are calculated for each column, representing the strength of influence of specific words on movie ratings. This step identifies key descriptors that contribute significantly to the overall prediction.

### Weighting Process

Concentration points are then weighted to assign varying degrees of importance to different descriptors. This ensures that the significance of each word is appropriately considered in the prediction process.

### Aggregation into Float Numbers

The final step involves aggregating the processed information into float numbers for each column. The weighted concentration points are combined to form a single float value that represents the contribution of the descriptors in predicting the movie rating.

Complete analysis can be found in dataPreprocessing.ipynb file.

## Models

### Support Vector Regression (SVR) Model

The SVR model is employed for the machine learning (ML) component of this project. SVR is a powerful regression technique that excels in capturing complex relationships between features and target variables.

### Neural Network Model

The neural network model is constructed using four dense layers. The architecture includes the usage of the Adam optimizer and ReLU activation functions. Mean Squared Error (MSE) is chosen as the loss function for training.

### Metrics

For the evaluation of the models, the following metrics are employed:

- **R2 Score (Coefficient of Determination):** This metric quantifies the proportion of the variance in the dependent variable that is predictable from the independent variables. An R2 score of 1 indicates perfect predictions.

- **Mean Squared Error (MSE):** This metric calculates the average of the squared differences between predicted and actual values. Lower MSE values indicate better model performance.

- **Mean Absolute Error (MAE):** This metric computes the average absolute differences between predicted and actual values. It provides a measure of the average prediction error.

These metrics collectively offer insights into the model's accuracy, precision, and ability to generalize to unseen data, providing a comprehensive evaluation of the regression performance.

## Docker Setup

Our project is containerized using Docker for efficient deployment and reproducibility. The Docker setup comprises six Dockerfiles, each representing a specific component of our system.

### Dockerfiles

1. **Dockerfile.fastapi:**
   - This Dockerfile is responsible for building the container for the FastAPI application.
   - Dependencies: All other Docker containers must be built and running.

2. **Dockerfile.mlmodel:**
   - This Dockerfile builds the container for the machine learning model (SVR).
   - Dependencies: PostgreSQL must be running, and the data preprocessing script and ML model script must be successfully executed.

3. **Dockerfile.nnmodel:**
   - This Dockerfile constructs the container for the neural network model.
   - Dependencies: PostgreSQL must be running, and the data preprocessing script and neural network model script must be successfully executed.

4. **Dockerfile.pgadmin:**
   - Builds the container for pgAdmin, a web-based PostgreSQL administration tool.
   - Dependencies: PostgreSQL must be running.

5. **Dockerfile.postgres:**
   - Creates the container for PostgreSQL, a relational database used to store preprocessed data.
   - No specific dependencies.

6. **Dockerfile.preprocessing:**
   - This Dockerfile is responsible for building the container that executes the data preprocessing script.
   - Dependencies: PostgreSQL must be running.

This orchestrated build process ensures that each component is built in the correct order, with dependencies satisfied before progressing to the next step.

## FastAPI Application

Our FastAPI application serves as the interface for interacting with the machine learning models and exploring the dataset. The following API endpoints provide various functionalities:

### API Endpoints

#### Get All Possible Words

#### 1. Get All Possible Words of Production Companies Column
   - **Endpoint:** `/production_companies/`
   - **HTTP Method:** GET
   - **Description:** Retrieves all unique words present in the `production_companies` column of the dataset.

#### 2. Get All Possible Words of Production Countries Column
   - **Endpoint:** `/production_countries/`
   - **HTTP Method:** GET
   - **Description:** Retrieves all unique words present in the `production_countries` column of the dataset.

#### 3. Get All Possible Words of Keywords Column
   - **Endpoint:** `/keywords/`
   - **HTTP Method:** GET
   - **Description:** Retrieves all unique words present in the `keywords` column of the dataset.

#### Check If Word Exists in a Specific Column

#### 4. Check If Word Exists in Production Companies Column
   - **Endpoint:** `/production_companies/{name}`
   - **HTTP Method:** GET
   - **Parameters:**
     - `{name}`: The word to check for in the `production_companies` column.
   - **Description:** Checks whether the specified word exists in the `production_companies` column of the dataset.

#### 5. Check If Word Exists in Production Countries Column
   - **Endpoint:** `/production_countries/{name}`
   - **HTTP Method:** GET
   - **Parameters:**
     - `{name}`: The word to check for in the `production_countries` column.
   - **Description:** Checks whether the specified word exists in the `production_countries` column of the dataset.

#### 6. Check If Word Exists in Keywords Column
   - **Endpoint:** `/keywords/{name}`
   - **HTTP Method:** GET
   - **Parameters:**
     - `{name}`: The word to check for in the `keywords` column.
   - **Description:** Checks whether the specified word exists in the `keywords` column of the dataset.

#### Prediction

#### 7. Predict Movie Rating
   - **Endpoint:** `/model/{model_name}/predict`
   - **HTTP Method:** POST
   - **Parameters:**
     - `{model_name}`: Specify the model for prediction (`nn` for neural network or `ml` for machine learning).
     - JSON payload with lists of strings for each column (keywords, production_companies, production_countries).
   - **Description:** Predicts the movie rating based on the selected model. The prediction will only work if all input strings are found in the existing words for each respective column.

These API endpoints provide a comprehensive set of functionalities for exploring the dataset, checking word existence in specific columns, and making movie rating predictions using either the neural network or machine learning model.

## Testing

Testing is a crucial aspect of ensuring the reliability and functionality of our project. We have implemented a suite of tests, including both unit tests and integrated tests.

### Unit Test

#### Data Preprocessor Class
- **Test Script:** `test_data_preprocessor.py`
- **Description:** This unit test focuses on evaluating the functions of the `DataPreprocessor` class. It ensures that individual functions within this class perform as expected, validating the correctness of data preprocessing steps.

### Integrated Tests

#### Data Preprocessor Class and Feature Extractor Class
- **Test Script:** `test_preprocessor_extractor.py`
- **Description:** Integrated tests are designed to evaluate the collaboration and interaction between the `DataPreprocessor` class and the `FeatureExtractor` class. These tests validate the end-to-end functionality of the data preprocessing and feature extraction processes.

### Automation with GitHub Actions

Both unit and integrated tests are automatically run whenever changes are pushed or pulled in the repository. This is achieved through GitHub Actions, ensuring that the testing suite is executed seamlessly as part of our continuous integration process. This automated testing approach helps maintain the integrity of our codebase and ensures that new changes do not introduce unexpected issues.

To view the test results and ensure the stability of our project, refer to the GitHub Actions workflow defined in the `.github/workflows/` directory.

Feel free to explore the test scripts for more detailed information on individual test cases and assertions.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following prerequisites installed on your machine:

- Docker: [Download and Install Docker](https://www.docker.com/get-started)
- Python: We recommend using Python 3.11.2 for running the project.

   If you don't have Python installed, you can download it from [Python Official Website](https://www.python.org/downloads/)

### Installation

#### Docker Setup

1. **Ensure Docker is Running:**
   Make sure Docker is installed on your machine and is running.

2. **Build and Run Everything:**
   Use Docker Compose to build and run all necessary containers for the project.
   ```bash
   docker-compose up --build

#### Local Development

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt

### Usage

#### Docker Setup

**Run Specific Containers:**
   If everything is already built, you can run specific containers using:.
   ```bash
   docker-compose up postgres pgadmin fastapi_app
```

#### Accessing pgAdmin

To access pgAdmin, follow these steps:

1. Open your web browser and go to [http://localhost:5050/](http://localhost:5050/).

2. Log in with the following credentials:
   - Email: `braniac@example.com`
   - Password: `braniac`

3. After logging in, follow the steps below to register a new server:

   - Right-click on "Servers" in the left sidebar.
   - Choose "Register" and then click "Server..."

4. In the "General" tab of the open window:

   - Set the Name to `Braniac`.

5. In the "Connection" tab:

   - Set the Host name to `postgres`.
   - Set the Port to `5432`.
   - Enter `braniac` for the Maintenance database, Username, and Password.

6. Click "Save" to register the new server.

That's it! You have now successfully accessed pgAdmin and registered a new server for your project. You can manage and interact with your PostgreSQL database through the registered server in pgAdmin.

### Accessing FastAPI

To interact with the FastAPI application, follow these steps:

1. Open your web browser and go to [http://localhost:8000/docs](http://localhost:8000/docs).

2. You will be directed to the FastAPI Swagger Documentation, which provides a user-friendly interface for exploring and testing the available APIs.

3. Explore the following APIs:

   - **Get All Possible Words of Production Companies Column:**
     - Endpoint: `/production_companies/`

   - **Get All Possible Words of Production Countries Column:**
     - Endpoint: `/production_countries/`

   - **Get All Possible Words of Keywords Column:**
     - Endpoint: `/keywords/`

   - **Check If Word Exists in Production Companies Column:**
     - Endpoint: `/production_companies/{name}`

   - **Check If Word Exists in Production Countries Column:**
     - Endpoint: `/production_countries/{name}`

   - **Check If Word Exists in Keywords Column:**
     - Endpoint: `/keywords/{name}`

   - **Predict Movie Rating:**
     - Endpoint: `/model/{model_name}/predict`

4. Click on each API endpoint to view detailed documentation, input parameters, and sample requests. You can also test the APIs directly from the Swagger Documentation.

That's it! You are now ready to explore and interact with the FastAPI application and its various endpoints.

### Running Main Scripts for Local Development

If you prefer local development, you can run each main script individually to interact with different components of the project. Ensure that you have installed the required Python dependencies using the `pip install -r requirements.txt` command.

To execute the data preprocessing script locally, use the following command:

```bash
python prepare_save_data.py
```

To run the machine learning model script locally, execute the following command:

```bash
python ml_model_main.py
```

For the neural network model, run the following command:

```bash
python nn_model_main.py
```

To run the FastAPI application locally, use the following command:

```bash
uvicorn main:app --reload
```




