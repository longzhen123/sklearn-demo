from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def main():

    n_cluster = 3

    # 生成数据
    x, y = datasets.make_blobs(n_samples=500, centers=n_cluster)

    # 全体可视化
    plt.figure()
    colors = ['red', 'green', 'blue', 'gray', 'pink', 'orange', 'yellow']

    for i in range(n_cluster):

        plt.scatter(x[y == i, 0],
                    x[y == i, 1],
                    color=colors[i])

    # plt.show()
    plt.savefig('all.png')

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 训练模型
    K = n_cluster
    model = KMeans(n_clusters=K)
    model.fit(x_train)

    y_pred = model.fit_predict(x_test)

    print(y_pred)

    # 测试集可视化-真实
    plt.figure()
    colors = ['red', 'green', 'blue', 'gray', 'pink', 'orange', 'yellow']

    for i in range(n_cluster):
        plt.scatter(x_test[y_test == i, 0],
                    x_test[y_test == i, 1],
                    color=colors[i])

    # plt.show()
    plt.savefig('true.png')

    # 测试集可视化-预测
    plt.figure()
    colors = ['red', 'green', 'blue', 'gray', 'pink', 'orange', 'yellow']

    for i in range(K):
        plt.scatter(x_test[y_pred == i, 0],
                    x_test[y_pred == i, 1],
                    color=colors[i])

    # plt.show()
    plt.savefig('pred.png')


if __name__ == '__main__':

    main()