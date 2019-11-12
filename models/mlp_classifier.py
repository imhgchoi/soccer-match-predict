import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler


class MLPClassifier(BaseModel):
    def __init__(self, config, dataset):
        super(MLPClassifier, self).__init__(config, dataset)

    def set_params(self):
        if self.config.mlpclass_w_init == 'uniform' :
            self.hidden = np.random.uniform(-2, 2, size=[41, 64])
            self.out = np.random.uniform(-2, 2, size=[65, 3])
        elif self.config.mlpclass_w_init == 'xavier' :
            i = 41
            h = 64
            o = 3
            self.hidden = np.random.uniform(-np.sqrt(6/(i+h)), np.sqrt(6/(i+h)), size=[i, h])
            self.out = np.random.uniform(-np.sqrt(6/(h+o+1)), np.sqrt(6/(h+o+1)), size=[h+1, o])


        self.alpha = self.config.mlpclass_alpha
        self.maxiter = self.config.mlpclass_maxiter
        self.dropout = self.config.mlpclass_dropout
        self.tol = self.config.mlpclass_tolerance
        self.printstep = self.config.mlpclass_printstep

        self.phase = 'train'
        self.decision_threshold = self.config.mlpclass_decision_threshold

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
        # dropout
        dropout_num = int(np.round(self.hidden.shape[1] * self.dropout))
        rand_idx = np.random.randint(0,self.hidden.shape[1],dropout_num)

        hidden_save = self.hidden.copy()
        out_save = self.out.copy()
        self.hidden[:, rand_idx] = 0
        self.out[rand_idx, :] = 0

        # add 1 for bias input --> multiply weights --> activation function
        X = np.concatenate((X, np.ones([X.shape[0],1])), axis=1)
        h = np.matmul(X, self.hidden)
        h = 1 / (1 + np.exp(-h))

        h_ = np.concatenate((h, np.ones([h.shape[0],1])), axis=1)
        out = np.exp(np.matmul(h_, self.out))
        out = out / np.expand_dims(np.sum(out, axis=1), axis=1)

        # restore dropout
        self.hidden = hidden_save
        self.out = out_save

        return rand_idx, h_, out


    def train(self):
        trainX_ = np.concatenate((self.dataset.trainX, np.ones([self.dataset.trainX.shape[0],1])), axis=1)
        trainY = np.array(pd.get_dummies(self.dataset.trainY))
        losses = []

        prev_hidden = self.hidden.copy()
        prev_out = self.out.copy()
        for e in range(self.maxiter) :
            if self.config.mlpclass_optimizer == 'gd' :
                # forward propagation
                dropout_idx, h, out = self.forward(self.dataset.trainX)

                loss = np.mean(np.sum(-(trainY * np.log(out)), axis=1))
                losses.append(loss)
                if e % self.printstep == 0:
                    print('epoch {0} loss : {1:.5f}'.format(e + 1, loss))

                # back propagation
                out_grad = np.transpose(np.matmul(np.transpose(out - trainY), h))
                hidden_grad = np.transpose(
                    np.matmul(np.transpose(np.matmul(out - trainY, np.transpose(self.out[:-1, :]))
                                           * (h[:, :-1] * (1 - h[:, :-1]))), trainX_))

                # finalize Delta
                hidden_delta = self.alpha * hidden_grad
                out_delta = self.alpha * out_grad

            elif self.config.mlpclass_optimizer == 'nag' :
                hidden_momentum = self.hidden - prev_hidden
                out_momentum = self.out - prev_out

                # update momentum effect
                current_hidden = self.hidden.copy()
                current_out = self.out.copy()
                self.hidden = self.hidden - self.config.mlpclass_momentum * hidden_momentum
                self.out = self.out - self.config.mlpclass_momentum * out_momentum

                # forward propagation
                dropout_idx, h, out = self.forward(self.dataset.trainX)

                # restore momenutm effect and save as previous weights for next iter
                self.hidden = current_hidden.copy()
                self.out = current_out.copy()
                prev_hidden = current_hidden.copy()
                prev_out = current_out.copy()

                loss = np.mean(np.sum(-(trainY * np.log(out)), axis =1))
                losses.append(loss)
                if e % self.printstep == 0 :
                    print('epoch {0} loss : {1:.5f}'.format(e+1, loss))

                # back propagation with momentum
                out_grad = np.transpose(np.matmul(np.transpose(out-trainY), h))
                hidden_grad =  np.transpose(np.matmul(np.transpose(np.matmul(out-trainY, np.transpose(self.out[:-1,:]))
                                                                   * (h[:,:-1]*(1-h[:,:-1]))), trainX_))

                # finalize Delta
                hidden_delta = self.alpha * hidden_grad + self.config.mlpclass_momentum * hidden_momentum
                out_delta = self.alpha * out_grad + self.config.mlpclass_momentum * out_momentum

            # apply dropout
            hidden_delta[:, dropout_idx] = 0
            out_delta[dropout_idx, :] = 0

            # weight update
            self.hidden = self.hidden - hidden_delta
            self.out = self.out - out_delta

            # exit condition
            if e > 10000 and (losses[-2] - losses[-1] < self.tol and losses[-2] - losses[-1] > 0):
                break

        plt.plot(losses)
        plt.title('mlp classifier gradient descent loss')
        plt.ylabel('NLL')
        plt.xlabel('epoch')
        plt.savefig('./out/mlpclass_loss.png')
        plt.close()

    def predict(self):
        dropout_tmp = self.dropout
        self.dropout = 0
        _, _, train_out = self.forward(self.dataset.trainX)
        _, _, test_out = self.forward(self.dataset.testX)
        self.dropout = dropout_tmp  # just in case

        train_mask = np.max(train_out, axis=1) >= self.decision_threshold
        test_mask = np.max(test_out, axis=1) >= self.decision_threshold

        train_odds = np.argmin(np.array(self.dataset.train_set[['Hodds','Dodds','Aodds']]),axis=1)
        test_odds = np.argmin(np.array(self.dataset.test_set[['Hodds','Dodds','Aodds']]),axis=1)

        train_out = train_mask * np.argmax(train_out, axis=1) + np.invert(train_mask) * train_odds
        test_out = test_mask * np.argmax(test_out, axis=1)+ np.invert(test_mask) * test_odds
        return train_out, test_out