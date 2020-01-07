# iris_keras_dnn.py
# Python 3.5.1, TensorFlow 1.6.0, Keras 2.1.5
# ========================================================
# 导入模块
import os
import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
from keras.engine.saving import load_model
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import LabelBinarizer
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# 读取CSV数据集，并拆分为训练集和测试集
# 该函数的传入参数为CSV_FILE_PATH: csv文件路径
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
    # train_x, test_x, train_y, test_y = train_test_split(X[features], X[y_bin_labels], train_size=0.7, test_size=0.3, random_state=0)
    # return train_x, test_x, train_y, test_y, Class_dict

    return X, features, y_bin_labels, Class_dict


def main():

    # 0. 开始
    print("\nIris dataset using Keras/TensorFlow ")
    np.random.seed(4)
    tf.compat.v1.set_random_seed(13)

    CSV_FILE_PATH = './subfile_1_train.csv'

    # 2. 定义模型
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=5, input_dim=13, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=3, kernel_initializer=init, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])

    # 1. 读取CSV数据集
    print("Loading data into memory")
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
    X, features, y_bin_labels, Class_dict = load_data(CSV_FILE_PATH)
    Y = X[y_bin_labels].values
    X = X[features].values
    round = 1
    for train_index, test_index in rkf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = Y[train_index], Y[test_index]

        # 3. 训练模型
        b_size = 1
        max_epochs = 100
        print("Starting training %d" % round)
        h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=2)
        print("Training  %d finished \n" % round)

        model.save('train_2_rooms_model_' + str(round) + '.m')

        # 4. 评估模型
        eval = model.evaluate(test_x, test_y, verbose=0)
        print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100) )
        with open('./train_2_rooms_result.txt', mode='a+', encoding='utf8') as result_file:
            result_file.write('Round %d: loss = %0.6f accuracy = %0.2f%% \n' % (round, eval[0], eval[1] * 100) )
        round += 1


def test():
    # 5. 使用模型进行预测
    CSV_FILE_PATH = './subfile_1_train.csv'
    train_x, test_x, train_y, test_y, Class_dict = load_data(CSV_FILE_PATH)

    model = load_model('model.m')
    np.set_printoptions(precision=4)
    unknown = np.array([[5,7,95.00,1,15.00,1,200.00,29.2,72.0,0.0,127.0,3.0,24.97789967]], dtype=np.float32)
    predicted = model.predict(unknown)
    print("Using model to predict species for features: ")
    print(unknown)
    print("\nPredicted softmax vector is: ")
    print(predicted)
    species_dict = {v:k for k,v in Class_dict.items()}
    print("\nPredicted species is: ")
    print(species_dict[np.argmax(predicted)])


if __name__ == '__main__':
    main()
    # test()
