import numpy as np
from sklearn.model_selection import KFold
import csv
import math
from svmutil import *


# 第二步，训练和测试
def train(i):
    csv_file_path = './room_1/subfile_%d_train.csv' % i
    y, x = svm_read_problem(csv_file_path)  # y是标签

    kf = KFold(n_splits=10, shuffle=True)
    f1_array = []
    acc_array = []
    round = 1
    for train_index, test_index in kf.split(y):
        x_train, y_train = np.array(x)[train_index], np.array(y)[train_index]
        x_test, y_test = np.array(x)[test_index], np.array(y)[test_index]
        print('subfile %d, round %d' % (i, round))
        model = svm_train(y_train, x_train, '-c 4 -h 0')
        svm_save_model('./models/subfile_%d.model' % i, model)

        p_label, p_acc, p_val = svm_predict(y_test, x_test, model)  # 预测
        acc_array.append(p_acc[0])

        # 接下来计算F1
        tp = 0  # 正类被预测为正类的数
        fp = 0  # 负类被预测为正类的数
        fn = 0  # 正类被预测为负类的数
        for (y_te, label) in zip(y_test, p_label):
            if y_te == 1.0 and label == 1.0:
                tp += 1.0
            if y_te == 0 and label == 1.0:
                fp += 1.0
            if y_te == 1.0 and label == 0:
                fn += 1.0

        p = tp / (tp + fp + 1)
        r = tp / (tp + fn + 1)
        f = (2 * p * r) / (p + r + 1)
        f1_array.append(f)

        round += 1

    with open('./room_1_result.csv', mode='a+', encoding='utf8', newline='') as f_save_result:
        csv_writer = csv.writer(f_save_result)

        if i == 100:
            headers = ['room', 'subfile', 'f1', 'acc']
            csv_writer.writerow(headers)

        csv_writer.writerow([1, i, math.fsum(f1_array)/10, math.fsum(acc_array)/10])


def test(x):
    model = svm_load_model('./subfile_1.model')
    y_test, x_test = svm_read_problem('./subfile_1_test.csv')  # 读取测试数据，y是标签
    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)    # 预测
    # print('ACC:' + str(p_acc[0]) + ', MSE:' + str(p_acc[1]) + ', SCC:' + str(p_acc[2]))
    # ACC, MSE, SCC = evaluations(y_test, p_label)    # 评估
    # print(y_test)
    # print(p_label)
    # print(p_acc)

    with open('./test_result.txt', 'w') as f_save_result:
        f_save_result.write('ACC:' + str(p_acc[0]) + ', MSE:' + str(p_acc[1]) + ', SCC:' + str(p_acc[2]) + '\n')


if __name__ == '__main__':

    for i in range(1, 101):
        train(i)
        # test(str(x))
