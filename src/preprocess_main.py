from data_preprocessor import DataPreprocessor

def main():
    data_preprocessor = DataPreprocessor('data.csv')
    data_preprocessor.preprocess()
    
if __name__ == "__main__":
    main()