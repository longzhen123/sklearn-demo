from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def main():

    # 下载数据集
    iris = datasets.load_iris()

    # 取出数据和标签
    iris_x = iris.data
    iris_y = iris.target

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.4)

    model = KNeighborsClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("============真实值=============")
    print(y_test)
    print("============预测值=============")
    print(y_pred)

    print(classification_report(y_test, y_pred))


if __name__ == '__main__':

    main()

