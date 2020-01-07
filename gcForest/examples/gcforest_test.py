
# 导入必要的模块

from gcforest.gcforest import GCForest


def init():

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for x in range(1, 3):
        with open('./subfile_' + str(x) + '_train.csv', mode='r', encoding='utf8') as fpto_train:
            lines = fpto_train.readlines()

            for row in lines:
                line = row.split(',')
                y_train.append(line[0])
                for col in line[1:]:
                    x_train.append(col)

        with open('./subfile_' + str(x) + '_test.csv', mode='r', encoding='utf8') as fpto_test:
            lines = fpto_test.readlines()

            for row in lines:
                line = row.split(',')
                y_test.append(line[0])
                for col in line[1:]:
                    x_test.append(col)

    return x_train, y_train, x_test, y_test


def get_toy_config():

    config = {}

    ca_config = {}

    ca_config["random_state"] = 0  # 0 or 1

    ca_config["max_layers"] = 100  # 最大的层数，layer对应论文中的level

    ca_config["early_stopping_rounds"] = 3  # 如果出现某层的三层以内的准确率都没有提升，层中止

    ca_config["n_classes"] = 3      # 判别的类别数量

    ca_config["estimators"] = []

    ca_config["estimators"].append(

            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,

             "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )

    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})

    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})

    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})

    config["cascade"] = ca_config    # 共使用了四个基学习器

    return config


def train():
    # 初始化一个gcForest对象

    gc = GCForest(get_toy_config())  # config是一个字典结构

    # gcForest模型最后一层每个估计器预测的概率concatenated的结果

    x_train, y_train, x_test, y_test = init()

    X_train_enc = gc.fit_transform(x_train, y_train)

    # 测试集的预测
    y_pred = gc.predict(x_test)


if __name__ == '__main__':
    init()
    train()
