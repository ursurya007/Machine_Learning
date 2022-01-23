print("y = 2x +3, Gradient Function, Cost function (Mean Squared error)")
print("Cost Function: (1/n)* sum of [( y-y')^2]")
print("m derivative: -(2/n)* sum of [x * ( y-y')]")
print("b derivative: -(2/n)* sum of [( y-y')]")
import numpy as np

def gradient_Descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        Y_Predicted = (m_curr * x) + b_curr
        cost = 1/n * sum([val**2 for val in (y - Y_Predicted)])
        md = -(2/n)*sum(x*(y-Y_Predicted))
        bd = -(2/n)*sum((y-Y_Predicted))
        m_curr = m_curr - (learning_rate)*md
        b_curr = b_curr - (learning_rate)*bd
        print("m: {}, b: {}, cost: {} iteration: {}".format(m_curr, b_curr, cost, i))


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_Descent(x, y)
