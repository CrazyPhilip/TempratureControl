# !usr/bin/env python

# encoding:utf-8

from __future__ import division

'''

__Author__:沂水寒城

功能： gcForest 实践 

'''

import numpy as np

# from GCForest import gcForest

from sklearn.externals import joblib

from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris, load_digits

from sklearn.model_selection import train_test_split


def irisFunc():
    '''

    对鸢尾花数据集进行测试

    '''

    iris = load_iris()

    X, y = iris.data, iris.target

    print('==========================Data Shape======================')

    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    model = gcForest(shape_1X=4, window=2, tolerance=0.0)

    model.fit(X_train, y_train)

    # 持久化存储

    joblib.dump(model, 'irisModel.sav')

    model = joblib.load('irisModel.sav')

    y_predict = model.predict(X_test)

    print('===========================y_predict======================')

    print(y_predict)

    accuarcy = accuracy_score(y_true=y_test, y_pred=y_predict)

    print('gcForest accuarcy : {}'.format(accuarcy))


if __name__ == '__main__':
    irisFunc()