import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler


class MLPRegressor(BaseModel):
    def __init__(self, config, dataset):
        super(MLPRegressor, self).__init__(config, dataset)

    def set_params(self):
        self.hidden = np.random.uniform(-2, 2, size=[41, 64])
        self.out = np.random.uniform(-2, 2, size=[65, 2])

        self.alpha = self.config.mlpclass_alpha
        self.maxiter = self.config.mlpclass_maxiter
        self.tol = self.config.mlpclass_tolerance
        self.printstep = self.config.mlpclass_printstep

    def preprocess(self):
        use_cols = ['FTR', 'home_wins', 'home_draws', 'home_losses', 'home_goals', 'home_oppos_goals',
                    'home_shots', 'home_oppos_shots', 'home_shotontarget', 'home_oppos_shotontarget',
                    'away_wins', 'away_draws', 'away_losses', 'away_goals', 'away_oppos_goals', 'away_shots',
                    'away_oppos_shots', 'away_shotontarget', 'away_oppos_shotontarget',
                    'home_oppos_wins', 'home_oppos_draws', 'home_oppos_losses', 'home_fouls', 'home_yellowcards',
                    'home_redcards', 'home_cornerkicks', 'home_oppos_cornerkicks', 'home_oppos_fouls',
                    'home_oppos_yellowcards', 'home_oppos_redcards', 'away_fouls', 'away_yellowcards', 'away_redcards',
                    'away_cornerkicks', 'away_oppos_cornerkicks', 'away_oppos_fouls', 'away_oppos_yellowcards',
                    'away_oppos_redcards', 'Hodds', 'Dodds', 'Aodds']

        train = self.dataset.train_set[use_cols]
        test = self.dataset.test_set[use_cols]

        # encode label
        train['FTR'] = train['FTR'].replace('H', 0).replace('D', 1).replace('A', 2)
        test['FTR'] = test['FTR'].replace('H', 0).replace('D', 1).replace('A', 2)

        # separate X, Y Dataset
        trainY = np.array(train.iloc[:, :1]).flatten()
        trainX = np.array(train.iloc[:, 1:])
        testY = np.array(test.iloc[:, :1]).flatten()
        testX = np.array(test.iloc[:, 1:])

        # apply min max scaling
        scaler = MinMaxScaler()
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)

        # save it in the dataset object
        self.dataset.trainX = trainX
        self.dataset.trainY = trainY
        self.dataset.testX = testX
        self.dataset.testY = testY

    def forward(self, X):
        # add 1 for bias input --> multiply weights --> activation function
        X = np.concatenate((X, np.ones([X.shape[0],1])), axis=1)
        h = np.matmul(X, self.hidden)
        h = 1 / (1 + np.exp(-h))

        h_ = np.concatenate((h, np.ones([h.shape[0],1])), axis=1)
        out = np.matmul(h_, self.out)
        return h_, out


    def train(self):
        trainX_ = np.concatenate((self.dataset.trainX, np.ones([self.dataset.trainX.shape[0],1])), axis=1)
        trainY = np.array(pd.get_dummies(self.dataset.trainY))
        losses = []

        for e in range(self.maxiter) :
            # forward propagation
            h, out = self.forward(self.dataset.trainX)

            loss = np.mean(np.sum(-(trainY * np.log(out)), axis =1))
            losses.append(loss)
            if e % self.printstep == 0 :
                print('epoch {0} loss : {1:.5f}'.format(e+1, loss))

            # back propagation
            out_grad = np.transpose(np.matmul(np.transpose(out-trainY), h))
            hidden_grad =  np.transpose(np.matmul(np.transpose(np.matmul(out-trainY, np.transpose(self.out[:-1,:])) * (h[:,:-1]*(1-h[:,:-1]))), trainX_))

            # weight update
            self.hidden = self.hidden - self.alpha * hidden_grad
            self.out = self.out - self.alpha * out_grad

            # exit condition
            if e > 100 and (losses[-2] - losses[-1] < self.tol and losses[-2] - losses[-1] > 0):
                break

        plt.plot(losses)
        plt.title('mlp classifier gradient descent loss')
        plt.ylabel('NLL')
        plt.xlabel('epoch')
        plt.savefig('./out/mlpclass_loss.png')
        plt.close()

    def predict(self):
        _, train_out = self.forward(self.dataset.trainX)
        _, test_out = self.forward(self.dataset.testX)

        train_out = np.argmax(train_out, axis=1)
        test_out = np.argmax(test_out, axis=1)
        return train_out, test_out