from config import get_args
from dataset import Dataset
from evaluator import Evaluator
from models.linreg import LinReg


def get_model(config, dataset) :
    if config.model_type == 'linreg' :
        model = LinReg(config, dataset)

    return model

def main():
    config = get_args()
    dataset = Dataset(config)
    dataset.get_data_info()
    model = get_model(config, dataset)

    # Training Step
    model.train()

    evaluator = Evaluator(config, dataset, model)
    evaluator.evaluate()



if __name__ == '__main__' :
    main()