#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Qiuxin Du
# time: 2022/6/28

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取数据
data = np.loadtxt('ex1data1.txt' , delimiter=',')
x = data[:,0][:,np.newaxis]
y = data[:,1][:,np.newaxis]

#数据可视化
# plt.figure()
# fg1 = plt.scatter(x, y, marker='x', c='r')
# plt.xlabel('populations')
# plt.ylabel('profits')
# plt.xticks(np.linspace(4,24,11))
# plt.yticks(np.linspace(-5,25,7))
# plt.title('Training data')

m = len(y)
a = np.ones((m, 1))
X = np.column_stack((a, x))
# print(X)
#计算代价
def computeCost(X,y,theta):
    J = 0
    J = np.sum(((X.dot(theta)-y)**2))/(2*m)
    return J

inerations = 1500
alpha = 0.01
# 梯度下降
def gradientDescent(X,y,theta,alpha,num_iters):
    J_history = np.zeros((num_iters,1))
    for iter in range(0,num_iters):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / m
        # theta = theta - alpha * X.T @ (X @ theta - y) / m
        J_history[iter] = computeCost(X,y,theta)
        print(iter,"++++",theta,"++++",J_history[iter])
    return theta,J_history



theta = np.zeros((2,1))
print('with theta = [0,0] ',computeCost(X,y,theta))
theta,J_history = gradientDescent(X,y,theta,alpha,inerations)
print('Theta found by gradient descent:\n',theta)

#代价函数可视化
plt.figure()
x_cost = np.arange(inerations)
fg2 = plt.plot(x_cost, J_history,c='b')
plt.title('Cost')


#拟合回归曲线
plt.figure()
fg1 = plt.scatter(x, y, marker='x', c='r')
x_regression = np.linspace(4,24)
y_regression = theta[0,0] + theta[1,0]*x_regression
fg1 = plt.plot(x_regression,y_regression,c='b')
plt.xlabel('populations')
plt.ylabel('profits')
plt.xticks(np.linspace(4,24,11))
plt.yticks(np.linspace(-5,25,7))
plt.title('Training data with regression function')
plt.show()