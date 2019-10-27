

class Evaluator():
    def __init__(self, config, model):
        self.config = config
        self.trainset = model.dataset.train_set
        self.testset = model.dataset.test_set
        self.model = model

    def evaluate(self):
        self.accuracy()
        self.profitability()

    def accuracy(self):
        prediction = self.model.predict()

    def profitability(self):
        prediction = self.model.predict()


        '''
        back testing algo
        '''