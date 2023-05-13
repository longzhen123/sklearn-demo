from sklearn import datasets


def main():

    # 波士顿数据集
    # boston = datasets.load_boston()
    # print(boston)

    # 鸢尾花数据集
    # iris = datasets.load_iris()
    # print(iris)

    # 糖尿病数据集
    # diabetes = datasets.load_diabetes()
    # print(diabetes)

    # 手写数据集
    # digits = datasets.load_digits()
    # print(digits)

    # Olivetti脸部图像数据集
    # olivetti_faces = datasets.fetch_olivetti_faces()
    # print(olivetti_faces)

    # 新闻分类数据集
    # newsgroups = datasets.fetch_20newsgroups()
    # print(newsgroups)

    # 带标签的人脸数据集
    lfw_people = datasets.fetch_lfw_people()
    print(lfw_people)


if __name__ == '__main__':

    main()