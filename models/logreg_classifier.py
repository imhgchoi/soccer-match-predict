import numpy as np
from models.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LogisticRegression(BaseModel):
    def __init__(self, config, dataset):
        super(LogisticRegression, self).__init__(config, dataset)


    def set_params(self):
        self.log_w = self.log_w_init()
        self.log_alpha = self.config.logreg_alpha
        #self.max_iter=self.config.logreg_max_iter

    def log_w_init(self) :
        if self.config.logreg_w_init == 'uniform' :
            weight = np.random.uniform(-2,2,size=[41, 1])
        return weight

    def preprocess(self):
        use_cols = ['FTR','home_wins', 'home_draws', 'home_losses', 'home_goals', 'home_oppos_goals',
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
        train['FTR'] = train['FTR'].replace('H',0).replace('D',1).replace('A',2)
        test['FTR'] = test['FTR'].replace('H',0).replace('D',1).replace('A',2)

        # separate X, Y Dataset
        trainY = np.array(train.iloc[:,:1])
        trainX = np.array(train.iloc[:,1:])
        testY = np.array(test.iloc[:,:1])
        testX = np.array(test.iloc[:,1:])

        # apply min max scaling
        scaler = MinMaxScaler()
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)

        # add ones for bias term
        trainX = np.concatenate((trainX, np.ones([trainX.shape[0], 1])), axis=1)
        testX = np.concatenate((testX, np.ones([testX.shape[0], 1])), axis=1)

        # save it in the dataset object
        self.dataset.trainX = trainX
        self.dataset.trainY = trainY
        self.dataset.testX = testX
        self.dataset.testY = testY



    def train(self):
        iter = [0,0,0]
        past_NLL = [9999999, 9999999,9999999]
        classes=[0,1,2]
        self.weights=[[],[],[]]
        for c in classes:
            binary_y=np.where(self.dataset.trainY==c,1,0)
            self.weights[c]=self.log_w_init()
            while True:
                iter[c] += 1
                z = np.matmul(self.dataset.trainX, self.weights[c])
                h=1 / (1 + np.exp(-z))
                error=h-binary_y
                NLL = -1 / h.shape[0] * (np.matmul(np.transpose(np.log(h)),binary_y)+np.matmul(np.transpose(np.log(1-h)),1-binary_y))
                gradient = 1 / h.shape[0] * np.matmul(np.transpose(error), self.dataset.trainX)

                self.weights[c]=self.weights[c] - self.log_alpha * np.transpose(gradient)
                print('iter {} : NLL {}'.format(iter, NLL))

                #if ((past_NLL[c] - NLL) < self.config.logreg_tolerance) or (iter[c]>=self.max_iter):
                if (past_NLL[c] - NLL) < self.config.logreg_tolerance:
                    break
                past_NLL[c] = NLL.copy()


    def predict(self):
        classes=[0,1,2]
        train_out=[]
        for row in self.dataset.trainX:
            train_out_idx=np.argmax(np.matmul(row,np.transpose(self.weights)))
            train_out.append(classes[train_out_idx])
        test_out=[]
        for row in self.dataset.testX:
            test_out_idx=np.argmax(np.matmul(row,np.transpose(self.weights)))
            test_out.append(classes[test_out_idx])
        return train_out,test_out