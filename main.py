from config import get_args
from dataset import Dataset
from trainer import Trainer
from evaluator import Evaluator
from models.linreg import LinReg
import pickle


def get_model(config, dataset) :
    if config.model_type == 'linreg' :
        model = LinReg(config, dataset)

    return model

def main():
    config = get_args()

    # saves the dataset object as pickle file for quick use
    if not config.use_prepro_v1 :
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