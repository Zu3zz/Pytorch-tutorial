# -*- coding: utf-8 -*-
# @Author : 3zz
# @Time   : 2019-08-07 12:08
# @File   : 03_Simple_Regression.py
import numpy as np


# y = wx + b
def compute_error_for_lin_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        # loss = (WX + b - y) ** 2 compute gradient
        w_gradient += -(2 / N) * x * (y - (w_current * x) + b_current)
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_lin_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}"
          .format(num_iterations, b, w,
                  compute_error_for_lin_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()
