-- Connect to the braniac database
\c braniac;

-- Create table for train_features
CREATE TABLE train_features (
    id SERIAL PRIMARY KEY,
    production_companies FLOAT,
    production_countries FLOAT,
    keywords FLOAT
);

-- Create table for val_features
CREATE TABLE val_features (
    id SERIAL PRIMARY KEY,
    production_companies FLOAT,
    production_countries FLOAT,
    keywords FLOAT
);

-- Create table for test_features
CREATE TABLE test_features (
    id SERIAL PRIMARY KEY,
    production_companies FLOAT,
    production_countries FLOAT,
    keywords FLOAT
);

-- Create table for train_target
CREATE TABLE train_target (
    id SERIAL PRIMARY KEY,
    rating FLOAT
);

-- Create table for val_target
CREATE TABLE val_target (
    id SERIAL PRIMARY KEY,
    rating FLOAT
);

-- Create table for test_target
CREATE TABLE test_target (
    id SERIAL PRIMARY KEY,
    rating FLOAT
);

-- Create table for production_companies
CREATE TABLE production_companies (
    id SERIAL PRIMARY KEY,
    company TEXT
);

-- Create table for production_countries
CREATE TABLE production_countries (
    id SERIAL PRIMARY KEY,
    country TEXT
);

-- Create table for keywords
CREATE TABLE keywords (
    id SERIAL PRIMARY KEY,
    keyword TEXT
);

