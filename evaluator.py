

class Evaluator():
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
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