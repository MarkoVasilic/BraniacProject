from data.data_preprocessor import DataPreprocessor
import data.helper as helper

def main():
    data_preprocessor = DataPreprocessor('dataset/data.csv')
    features = data_preprocessor.preprocess()
    helper.save_features_to_postgres(features)
    scaled_features, target_ratings = data_preprocessor.process_features_and_target()
    helper.save_dataframes_to_postgres(scaled_features, target_ratings)
    
if __name__ == "__main__":
    main()