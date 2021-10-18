# coding=utf-8
# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt

import data_process
import logistic_regression
if __name__ == '__main__':

    #获取处理后的数据
    train_label, train_data, test_label, test_data=data_process.process()


    #迭代次数
    epochs = 100
    #mini_batch大小
    batch_size = 4
    #步长
    step = 0.05

    #调用逻辑回归模型，获得结果和参数变化的矩阵
    w, b, accuracys, w_s, b_s = logistic_regression.mini_batch(batch_size, step, epochs, train_data, train_label,test_data,test_label)


    # 输出最终结束
    result = accuracys[len(accuracys)-1]
    #result 0.883333
    print("the last accuracy {}".format(result))

    #以下为绘制图像的过程
    iterations = len(w_s)

    dis_w = []
    dis_b = []

    for i in range(iterations):
        dis_w.append(np.linalg.norm(w_s[i] - w))
        dis_b.append(np.linalg.norm(b_s[i] - b))

    print(
        "the parameters is: step length alpah:{}; batch size:{}; Epoches:{}".format(step, batch_size, epochs))
    # print("Result: accuracy:{:.2%},time cost:{:.2f}".format(accuracys[-1], time_cost))

    plt.title('The Model accuracy variation chart ')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.plot(accuracys, 'm')
    plt.grid()
    plt.show()

    plt.title('The distance from the optimal solution')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.plot(dis_w, 'r', label='distance between W and W*')
    plt.plot(dis_b, 'g', label='distance between b and b*')
    plt.legend()
    plt.grid()
    plt.show()





