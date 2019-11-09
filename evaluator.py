import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class Evaluator():
    def __init__(self, config, model):
        self.config = config
        self.trainX = model.dataset.trainX
        self.trainY = model.dataset.trainY
        self.testX = model.dataset.testX
        self.testY = model.dataset.testY
        self.model = model


    def reg_to_class(self, arr) :
        boolean = np.array([(arr[:,0] > arr[:,1]),
                            (arr[:,0] == arr[:,1]),
                            (arr[:,0] < arr[:,1])])
        return np.argmax(boolean, axis=0)


    def evaluate(self):
        print("\n\n###################################### EVALUATION ######################################\n")
        if self.config.model_type in ['linreg','kernelreg','nnreg','knnreg','ridge','lasso'] :
            self.regression_metrics()
        else :
            self.classification_metrics()
        self.profitability()


    def regression_metrics(self):
        train_out, test_out = self.model.predict()

        # Regression Evaluation Metrics
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

        # Regression to Classification
        train_pred = self.reg_to_class(train_out)
        train_actual = self.reg_to_class(self.trainY)
        test_pred = self.reg_to_class(test_out)
        test_actual = self.reg_to_class(self.testY)

        # Classification Reports
        print('---Train set confusion matrix and classification report---')
        print(confusion_matrix(train_pred, train_actual))
        print('accuracy : {0:.4f}\n'.format(accuracy_score(train_pred, train_actual)))
        print(classification_report(train_pred, train_actual))

        print('---test set confusion matrix and classification report---')
        print(confusion_matrix(test_pred, test_actual))
        print('accuracy : {0:.4f}\n'.format(accuracy_score(test_pred, test_actual)))
        print(classification_report(test_pred, test_actual))


    def classification_metrics(self):
        train_out, test_out = self.model.predict()
        # Classification Reports
        print('---Train set confusion matrix and classification report---')
        print(confusion_matrix(train_out, self.trainY))
        print('accuracy : {0:.4f}\n'.format(accuracy_score(train_out, self.trainY)))
        print(classification_report(train_out, self.trainY))
        print('---test set confusion matrix and classification report---')
        print(confusion_matrix(test_out, self.testY))
        print('accuracy : {0:.4f}\n'.format(accuracy_score(test_out, self.testY)))
        print(classification_report(test_out, self.testY))


    def profitability(self):
        print('---Profitability Testing : Backtest---')
        train_out, test_out = self.model.predict()

        # If model is regression type, alter prediction to classification
        if self.config.model_type in ['linreg','kernelreg','nnreg','knnreg','ridge','lasso'] :
            train_pred = self.reg_to_class(train_out)
            train_actual = self.reg_to_class(self.trainY)
            test_pred = self.reg_to_class(test_out)
            test_actual = self.reg_to_class(self.testY)
        elif self.config.model_type == 'logreg':
            train_pred = train_out
            train_actual = self.trainY.flatten()
            test_pred = test_out
            test_actual = self.testY.flatten()
        else :
            train_pred = train_out
            train_actual = self.trainY
            test_pred = test_out
            test_actual = self.testY

        train_res = pd.DataFrame({'pred': train_pred, 'actual': train_actual})
        test_res = pd.DataFrame({'pred': test_pred, 'actual': test_actual})

        # see if prediction is correct
        train_res['correct'] = train_res['pred'] == train_res['actual']
        test_res['correct'] = test_res['pred'] == test_res['actual']

        # concat with bet odds
        train_res = pd.concat([train_res, self.model.dataset.train_set[['Hodds','Dodds','Aodds']].reset_index(drop=True)], axis=1)
        test_res = pd.concat([test_res, self.model.dataset.test_set[['Hodds','Dodds','Aodds']].reset_index(drop=True)], axis=1)

        # generate random benchmark
        train_res['rand_pred'] = np.random.randint(0,3,size=train_res.shape[0])
        test_res['rand_pred'] = np.random.randint(0,3,size=test_res.shape[0])
        train_res['rand_correct'] = train_res['rand_pred'] == train_res['actual']
        test_res['rand_correct'] = test_res['rand_pred'] == test_res['actual']

        # compute profit for train
        train_profit = 0
        train_profit_list = [0]
        rand_profit = 0
        rand_profit_list = [0]
        for idx, row in enumerate(train_res.iterrows()) :
            train_profit -= self.config.bet_money
            rand_profit -= self.config.bet_money
            if row[1]['correct'] :
                train_profit += (row[1][3+row[1]['pred']] * self.config.bet_money) * (1-self.config.commission) * (1-self.config.tax)
            if row[1]['rand_correct'] :
                rand_profit += (row[1][3+row[1]['rand_correct']] * self.config.bet_money) * (1-self.config.commission) * (1-self.config.tax)
            train_profit_list.append(train_profit / (self.config.bet_money * (idx+1)))
            rand_profit_list.append(rand_profit / (self.config.bet_money * (idx+1)))
        print('Train set Net Profit : ', train_profit)
        print('Train set Return on Investment [ Net Profit / Total Money Invested ] : ', train_profit / (self.config.bet_money * train_res.shape[0]))
        print('Train Random Net Profit : ', rand_profit)
        print('Train Random Return on Investment [ Net Profit / Total Money Invested ] : ', rand_profit / (self.config.bet_money * train_res.shape[0]))
        print('\n')
        plt.plot(train_profit_list[int(len(train_profit_list)*0.1):])
        plt.plot(rand_profit_list[int(len(rand_profit_list)*0.1):])
        plt.title('Train Set Bet Backtest')
        plt.xlabel('Soccer Match Sequence')
        plt.ylabel('Return on Investment [ Net Profit / Total Money Invested ]')
        plt.legend(['train','random'])
        plt.savefig('./out/{}_train_profit.png'.format(self.config.model_type))
        plt.close()

        # compute profit for test
        test_profit = 0
        test_profit_list = [0]
        rand_profit = 0
        rand_profit_list = [0]
        for idx, row in enumerate(test_res.iterrows()) :
            test_profit -= self.config.bet_money
            rand_profit -= self.config.bet_money
            if row[1]['correct'] :
                test_profit += (row[1][3+row[1]['pred']] * self.config.bet_money) * (1-self.config.commission) * (1-self.config.tax)
            if row[1]['rand_correct'] :
                rand_profit += (row[1][3+row[1]['rand_correct']] * self.config.bet_money) * (1-self.config.commission) * (1-self.config.tax)
            test_profit_list.append(test_profit / (self.config.bet_money * (idx+1)))
            rand_profit_list.append(rand_profit / (self.config.bet_money * (idx+1)))
        print('Test set Net Profit : ', test_profit)
        print('Test set Return on Investment [ Net Profit / Total Money Invested ] : ', test_profit / (self.config.bet_money * test_res.shape[0]))
        print('Test Random Net Profit : ', rand_profit)
        print('Test Random Return on Investment [ Net Profit / Total Money Invested ] : ', rand_profit / (self.config.bet_money * test_res.shape[0]))
        plt.plot(test_profit_list[int(len(test_profit_list)*0.1):])
        plt.plot(rand_profit_list[int(len(rand_profit_list)*0.1):])
        plt.title('Test Set Bet Backtest')
        plt.xlabel('Soccer Match Sequence')
        plt.ylabel('Return on Investment [ Net Profit / Total Money Invested ]')
        plt.legend(['test','random'])
        plt.savefig('./out/{}_test_profit.png'.format(self.config.model_type))
        plt.close()