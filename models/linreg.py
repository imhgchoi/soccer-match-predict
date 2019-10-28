import numpy as np
from models.base_model import BaseModel

class LinReg(BaseModel):
    def __init__(self, config, dataset):
        super(LinReg, self).__init__(config, dataset)


    def set_params(self):
        self.w = self.w_init()
        self.alpha = self.config.alpha

    def w_init(self) :
        if self.config.w_init == 'uniform' :
            weight = np.random.uniform(-1,1,size=[100,1])   # example

        return weight

    def preprocess(self):
        use_cols = ['Date','FTHG','FTAG','home_wins', 'home_draws', 'home_losses', 'home_goals', 'home_oppos_goals',
                    'home_shots', 'home_oppos_shots', 'home_shotontarget', 'home_oppos_shotontarget',
                    'away_wins', 'away_draws', 'away_losses', 'away_goals', 'away_oppos_goals',
                    'away_shots', 'away_oppos_shots', 'away_shotontarget', 'away_oppos_shotontarget','Hodds','Dodds','Aodds']

        train = self.dataset.train_set[use_cols]
        test = self.dataset.test_set[use_cols]
        exit()

    def train(self):
        '''
        Just a raw example ; this code will not properly work yet
        '''
        print(self.dataset.train_set)

        def cost():
            return 1 / (2 * len(self.dataset.train_set)) * \
                   np.sum(np.power((np.matmul(self.dataset.train_set, self.w) - self.dataset.train_set.iloc[:,1]), 2))

        iter = 1
        iter_cost = []
        while True:
            temp = [0] * self.w.shape[0]
            for i in range(self.w.shape[0]):
                t = self.w[i][0] - self.alpha * (1 / len(self.dataset.train_set)) * \
                               np.sum((np.matmul(self.dataset.train_set, self.w) - self.dataset.train_set.iloc[:,1]) * \
                                      self.dataset.train_set[:, i])
                temp[i] = t
            param = temp.copy()
            print('iter', str(iter), ':', cost())
            iter_cost.append([iter, cost()])
            iter += 1

            if cost() < .001:
                print(param)
                break


    def predict(self,):
        '''
        Refers to parameters and produce predictions
        :return: predicted value for input data
        '''
        pass
