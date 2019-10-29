import numpy as np

class Evaluator():
    def __init__(self, config, model):
        self.config = config
        self.trainX = model.dataset.trainX
        self.trainY = model.dataset.trainY
        self.testX = model.dataset.testX
        self.testY = model.dataset.testY
        self.model = model

    def evaluate(self):
        print("\n------------EVALUATION------------")
        self.metrics()
        self.profitability()

    def metrics(self):
        train_out, test_out = self.model.predict()

        if 'mse' in self.config.eval_metrics :
            train_error = (train_out - self.trainY)**2
            train_MSE = 1/train_out.shape[0] * np.sum(train_error, axis=0)

            test_error = (test_out - self.testY)**2
            test_MSE = 1/test_out.shape[0] * np.sum(test_error, axis=0)

            print('train set : HOME MSE - {}  /  AWAY MSE - {}'.format(train_MSE[0], train_MSE[1]))
            print('test set : HOME MSE - {}  /  AWAY MSE - {}'.format(test_MSE[0], test_MSE[1]))
            print('\n')

        if 'mae' in self.config.eval_metrics :
            train_MAE = 1/train_out.shape[0] * np.sum(np.abs(train_out - self.trainY), axis=0)
            test_MAE = 1/test_out.shape[0] * np.sum(np.abs(test_out - self.testY), axis=0)

            print('train set : HOME MAE - {}  /  AWAY MAE - {}'.format(train_MAE[0], train_MAE[1]))
            print('test set : HOME MAE - {}  /  AWAY MAE - {}'.format(test_MAE[0], test_MAE[1]))
            print('\n')

    def profitability(self):
        train_out, test_out = self.model.predict()


        '''
        back testing algo
        '''