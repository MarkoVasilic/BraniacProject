import warnings
warnings.filterwarnings('ignore')
from db.dataloader import DataLoader
from models.ml_model import SVRModel

def main():
    data_loader = DataLoader()
    features, labels = data_loader.load_data()

    svr_model = SVRModel()
    svr_model.train(features['train_features'], labels['train_labels'])
    svr_model.evaluate(features['test_features'], labels['test_labels'])
    svr_model.save()

if __name__ == "__main__":
    main()
