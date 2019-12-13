import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier


class AutoML():

    def __init__(self, path, target, type_ML='regression'):
        self.df = pd.read_csv(path)
        self.target = target
        self.type_ML = type_ML
        self.fitted = False
        self.models = {}
        self.results = {}
        if self.type_ML == 'regression':
            self.models['Linear Regression'] = LinearRegression()
        elif self.type_ML == 'classification':
            self.models['Log Reg'] = LogisticRegression(random_state=0)
            self.models['SVM'] = SVC(random_state=0, kernel='rbf')
            self.models['Random Forest'] = RandomForestClassifier(random_state=0)
            self.models['Ada Boost'] = AdaBoostClassifier(random_state=0)
            self.models['Gradient Boost'] = GradientBoostingClassifier(random_state=0)
            self.models['X Gradient Boost'] = XGBClassifier(random_state=0)

    def preprocess(self):
        '''
        process the dataframe
        :return:
        '''
        # Nan
        dimension = self.df.shape
        # delete columns with more than 10% NaN
        for column in self.df.columns:
            if self.df[column].isnull().sum() / dimension[0] > 0.1:
                self.df.drop([column], inplace=True, axis=1)

        # drop rows with Nans
        self.df.dropna(axis=0, inplace=True)

        # Separate X et y:
        y = self.df[self.target]
        X = self.df.drop([self.target], axis=1)

        # categories : getdummies
        for column in X.columns:
            try:
                pd.to_numeric(X[column])
            except ValueError:
                dummies = pd.get_dummies(X[column], drop_first=True)
                new_columns = dummies.columns
                X[column + '_' + new_columns] = dummies[new_columns]
                X.drop([column], inplace=True, axis=1)

        return train_test_split(X, y, test_size=0.25, random_state=0)

    def auto_fit(self, X, y, params=None):
        '''
        Fit X to y using every kind of model
        :param param:
        :param X:
        :param y:
        :return:
        '''

        if params is None:
            for model in self.models:
                self.models[model].fit(X, y)
        else:
            self.models['Log Reg'] = self.grid_search_log_reg(X, y, params['Log Reg'], 'accuracy')
            self.models['SVM'] = self.grid_search_svm(X, y, params['SVM'], 'accuracy')
            self.models['Random Forest'] = self.grid_search_random_forest(X, y, params['Random Forest'], 'accuracy')
            self.models['Ada Boost'] = self.grid_search_adaboost(X, y, params['Ada Boost'], 'accuracy')
            self.models['Gradient Boost'] = self.grid_search_gboost(X, y, params['Gradient Boost'], 'accuracy')
            self.models['X Gradient Boost'] = self.grid_search_xgboost(X, y, params['X Gradient Boost'], 'accuracy')
        self.fitted = True

    def auto_predict(self, X):
        '''
        predict the target
        :param X:
        :return:
        '''
        if self.fitted:
            predict = {}
            for model in self.models:
                predict[model] = self.models[model].predict(X)
            return predict
        else:
            print("fit the models before trying to predict")

    def grid_search(self, X, y, model, params, scoring):
        '''
        Search the best parameters according to the scoring metric
        for the machine learning model
        :param model: machine learning object
        :param params: parameters to test
        :param scoring: metric
        :return: gridsearch object
        '''
        clf = GridSearchCV(model, params, scoring=scoring, n_jobs=-1)
        clf.fit(X, y)
        return clf.best_estimator_

    def grid_search_log_reg(self, X, y, params, scoring):
        return self.grid_search(X, y, self.models['Log Reg'], params, scoring)

    def grid_search_svm(self, X, y, params, scoring):
        return self.grid_search(X, y, self.models['SVM'], params, scoring)

    def grid_search_random_forest(self, X, y, params, scoring):
        return self.grid_search(X, y, self.models['Random Forest'], params, scoring)

    def grid_search_adaboost(self, X, y, params, scoring):
        return self.grid_search(X, y, self.models['Ada Boost'], params, scoring)

    def grid_search_gboost(self, X, y, params, scoring):
        return self.grid_search(X, y, self.models['Gradient Boost'], params, scoring)

    def grid_search_xgboost(self, X, y, params, scoring):
        return self.grid_search(X, y, self.models['X Gradient Boost'], params, scoring)

    def accuracy(self, X_train, y_train, X_test, y_test):
        json = {}
        for model in self.models:
            model_accu = {
                'test_accuracy': accuracy_score(self.models[model].predict(X_test), y_test),
                'train_accuracy': accuracy_score(self.models[model].predict(X_train), y_train)}
            json[model] = model_accu
        self.results = json

    def resultats(self, params=None):
        X_train, X_test, y_train, y_test = self.preprocess()
        self.auto_fit(X_train, y_train, params=params)
        self.accuracy(X_train, y_train, X_test, y_test)
        return self.results




######  TEST  #####
auto_data = AutoML('Social_Network_Ads.csv', 'Purchased', type_ML='classification')
params = {'Log Reg': {'C': [1, 10]},
          'SVM': {'kernel': ['rbf'], 'C': [1, 10, 50, 100], 'gamma': [0.1, 1, 5, 10, 50]},
          'Random Forest': {'n_estimators': [10, 50, 100], 'criterion': ['entropy', 'gini'],
                            'max_depth': [2, 3, 5], 'max_features': ['auto', 2, 3]},
          'Ada Boost': {'n_estimators': [10, 50, 100], 'learning_rate': [0.1, 1, 10]},
          'Gradient Boost': {'n_estimators': [10, 20, 30], 'learning_rate': [0.01, 0.1, 1],
                             'max_depth': [3, 5]},
          'X Gradient Boost': {'n_estimators': [10, 20, 30]}}
results = auto_data.resultats(params=params)
print(results)
