from config import get_args
from dataset import Dataset
from trainer import Trainer
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
    trainer = Trainer(config, model)
    trained_model = trainer.train()

    # Evaluation Step
    evaluator = Evaluator(config, trained_model)
    evaluator.evaluate()



if __name__ == '__main__' :
    main()