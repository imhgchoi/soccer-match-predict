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
        self.accuracy()
        self.profitability()

    def accuracy(self):
        train_out, test_out = self.model.predict()

        train_error = (train_out - self.trainY)**2
        train_MSE = 1/(2*train_out.shape[0]) * np.sum(train_error, axis=0)

        test_error = (test_out - self.testY)**2
        test_MSE = 1/(2*test_out.shape[0]) * np.sum(test_error, axis=0)

        print(train_MSE)
        print(test_MSE)

    def profitability(self):
        prediction = self.model.predict()


        '''
        back testing algo
        '''