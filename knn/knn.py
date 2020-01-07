import csv
import math

from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

def load_data(CSV_FILE_PATH):
    X = pd.read_csv(CSV_FILE_PATH)
    target_var = 'set_temp'  # 目标变量
    # 数据集的特征
    features = list(X.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = X[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target, 将目标变量进行编码
    X['target'] = X[target_var].apply(lambda x: Class_dict[x])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(X['target'])
    y_bin_labels = []  # 对多分类进行0-1编码的变量
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        X['y' + str(i)] = transformed_labels[:, i]
    # 将数据集分为训练集和测试集

    return X, features, y_bin_labels, Class_dict


def train(s):
    csv_file_path = './room_1/subfile_%d.csv' % s

    X, features, y_bin_labels, Class_dict = load_data(csv_file_path)
    kf = KFold(n_splits=10, shuffle=True)
    Y = X[y_bin_labels].values.flatten()
    X = X[features].values
    round = 1

    loss_array = []
    acc_array = []
    result_file = open('./room_1_result.csv', mode='a+', encoding='utf8', newline='')
    csv_writer = csv.writer(result_file)

    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = Y[train_index], Y[test_index]

        knn = KNeighborsClassifier()

        knn.fit(train_x, train_y)

        y_predict = knn.predict(test_x)

        # print('-----predict value is ------')

        # print(y_predict)

        # print('-----actual value is -------')

        # print(test_y)

        count = 0

        for j in range(len(y_predict)):
            if y_predict[j] == test_y[j]:
                count += 1

        acc_array.append(100 * count / len(y_predict))
        # print('accuracy is %0.2f%%' % (100 * count / len(y_predict)))
        round += 1

    if s == 1:
        headers = ['room', 'subfile', 'acc']
        csv_writer.writerow(headers)

    csv_writer.writerow([1, s, math.fsum(acc_array)/10])
    result_file.close()


if __name__ == '__main__':
    for s in range(1, 101):
        train(s)
