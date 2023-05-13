from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def main():

    # 下载数据集
    iris = datasets.load_iris()

    # 取出样本
    x = iris.data

    # 取出标签
    y = iris.target

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 构建决策树

    # 信息熵
    model = RandomForestClassifier(criterion="gini", n_estimators=10)
    model.fit(x_train, y_train)

    # 预测标签
    y_predict = model.predict(x_test)

    print("==========真实值=============")
    print(y_test)
    print("==========预测值=============")
    print(y_predict)

    print(classification_report(y_test, y_predict))

    for index, model_ in enumerate(model.estimators_):
        plt.figure()
        plot_tree(model_)
        # plt.show()
        plt.savefig('tree' + str(index) + '.png')


if __name__ == '__main__':

    main()