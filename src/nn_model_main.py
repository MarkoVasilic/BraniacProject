import warnings
warnings.filterwarnings('ignore')
from db.dataloader import DataLoader
from models.nn_model import NNModel

def main():
    data_loader = DataLoader()
    features, labels = data_loader.load_data()

    nn_model = NNModel()
    nn_model.build_neural_network(input_dim=features['train_features'].shape[1])
    nn_model.train(features, labels)
    nn_model.evaluate(features['test_features'], labels['test_labels'])
    nn_model.save()

if __name__ == "__main__":
    main()
