from config import get_args
from dataset import Dataset
from trainer import Trainer
from evaluator import Evaluator
from models.linreg import LinReg
from models.svm import SVM
from models.knn_classifier import KNNClassifier
from models.logreg_classifier import LogisticRegression
from models.mlp_classifier import MLPClassifier
from models.mlp_regressor import MLPRegressor
import pickle


def get_model(config, dataset) :
    '''
    ADD BRANCH HERE, IF YOU HAVE ADDED A MODEL INTO models FOLDER
    '''
    if config.model_type == 'linreg' :
        model = LinReg(config, dataset)
    elif config.model_type == 'knnclass' :
        model = KNNClassifier(config, dataset)
    elif config.model_type=='svm':
        model = SVM(config, dataset)
    elif config.model_type == 'logreg':
        model = LogisticRegression(config, dataset)
    elif config.model_type == 'mlpclass' :
        model = MLPClassifier(config, dataset)
    elif config.model_type == 'mlpreg' :
        model = MLPRegressor(config, dataset)


    return model

def main():
    config = get_args()

    # saves the dataset object as pickle file for quick use
    if not config.use_preprocessed :
        dataset = Dataset(config)
        with open(config.datadir+'prepro_v1.pkl', 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    else :
        with open(config.datadir+'prepro_v1.pkl', 'rb') as f:
            dataset = pickle.load(f)
    dataset.get_data_info()
    model = get_model(config, dataset)

    # Training Step
    trainer = Trainer(config, model)
    trained_model = trainer.train()

    # Evaluation Step
    evaluator = Evaluator(config, trained_model)
    evaluator.evaluate()



if __name__ == '__main__' :
    main()