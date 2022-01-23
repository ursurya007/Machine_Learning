import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def estimate_coef(x, y):
    #num of observation points
    n = np.size(x)
    print("size of x:", x)

    #mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    print("mean of x:", m_x)
    print("mean of y:", m_y)
    print("np.sum(x*y): ", np.sum(x*y))
    print("np.sum(x*x): ", np.sum(x*x))

    #Cross-deviation and deviation about x
    ss_xy = np.sum(x*y) - n*m_x*m_y
    ss_xx = np.sum(x*x) - n*m_x*m_x
    print("Cross Deviation ss_xy: ", ss_xy)
    print("Deviation ss_xx: ", ss_xx)

    #regression co-efficients
    b_1 = ss_xy/ss_xx
    b_0 = m_y-b_1*m_x

    print("b_1 (m)", b_1)
    print("b_0 (c)", b_0)

    return (b_0, b_1)

def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m", marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()

# def Linear_Method(z, y):
#     reg = linear_model.LinearRegression()
#     reg.fit(z, y)
#
#     m = reg.coef_
#     b = reg.intercept_
#
#     print("m: ", m)
#     print("b: ", b)

def main():
    x = np.arange(0, 10, 1)
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    # z = np.array([[1, 3, 2, 5, 10, 4, 12, 8, 20, 15], [4, 12, 8, 20, 15, 1, 3, 2, 5, 10]])
    # z = np.array([[1, 3, 2, 5, 10], [4, 12, 8, 20, 15]])
    print(x, type(x))
    print(y, type(y))

    #co-efficients
    b = estimate_coef(x, y)

    # Linear_Method(z, y)
    print("Estimated Co-efficients: \nb_0 = {} \nb_1 = {}".format(b[0], b[1]))

    #Plotting regression line
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()