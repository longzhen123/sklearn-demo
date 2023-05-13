from sklearn.tree import DecisionTreeClassifier, plot_tree
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
    model = DecisionTreeClassifier(criterion="gini")
    model.fit(x_train, y_train)

    # 预测标签
    y_predict = model.predict(x_test)

    print("==========真实值=============")
    print(y_test)
    print("==========预测值=============")
    print(y_predict)

    print(classification_report(y_test, y_predict))

    plt.figure()
    plot_tree(model)
    plt.show()
    # plt.savefig('tree.png')


if __name__ == '__main__':

    main()