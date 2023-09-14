import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Load data as arrays
X = np.array([[3.66206796],
              [3.60068353],
              [3.35561064],
              [3.13566507],
              [2.88363364],
              [2.6731554],
              [2.50183297],
              [2.34888979],
              [2.23473318],
              [2.35913087],
              [2.27122698],
              [2.38720761],
              [2.64280863],
              [2.88190681],
              [3.10203949],
              [3.35992634],
              [3.57304099],
              [3.70489007],
              [3.82332084],
              [3.73969921]])

Y = np.array([[3.66206796],
             [3.83371583],
             [3.76138194],
             [3.73451543],
             [3.54827928],
             [3.3268446],
             [3.10573058],
             [2.87973988],
             [2.6425749],
             [2.5382605],
             [2.27122698],
             [2.14947774],
             [2.23523361],
             [2.36062194],
             [2.51922415],
             [2.64007366],
             [2.87837823],
             [3.13019327],
             [3.38453978],
             [3.53294554]])

# Convert raw data to suitable ndarray for later use

X1, Y1 = [], []
for i in range(len(X)):
    X1.append(X[i, 0])
    Y1.append(Y[i, 0])
X, Y = np.array(X1), np.array(Y1)

# Creates empty matrix
M = np.empty((0, 6))

# Appends row vectors from data, assuming a quadratic form Ax^2+Bxy+Cx+Dy^2+Ey+F=0
for i in range(len((X))):
    row_vec = np.array([X[i]**2, X[i]*Y[i],
                       X[i], Y[i]**2, Y[i], 1])
    M = np.vstack((M, row_vec))

# Perform SVD on M
U, S, VT = np.linalg.svd(M)


# Obtain right singular vector corresponding to minimum singular value, which gives an initial guess
for i in range(len(S)):
    if S[i] == np.min(S):
        col_val = i
V = np.transpose(VT)

# Get initial parameters
param_list = V[:, 5]

# Construct the cost function, which is what we are trying to minimize.


def cost(params, X, Y):
    a, b, c, d, e, f = params[0], params[1], params[2], params[3], params[4], params[5]

    g11, g12, g13, g22, g23, g33 = a**2+d**2, a*d + \
        b*d, a*e+d*f, b**2+d**2, d*e+b*f, e**2+f**2

    w11, w12, w13, w22, w23, w33 = a**3+2*a*d**2+b*d**2, d * \
        (a**2+d**2+a*b+b**2), a**2*e+d**2*e+a*d*f+b*d*f, a*d**2+2 * \
        b*d**2+b**3, a*d*e+b*d*e+b**2*f+d**2*f, a*e**2+2*d*e*f+b*f**2

    # Construct the W, G, and C matrices for the sum
    C, G, W = np.array([[a, b, c], [b, d, e], [c, e, f]]), np.array([[g11, g12, g13], [g12, g22, g23], [
        g13, g23, g33]]), np.array([[w11, w12, w13], [w12, w22, w23], [w13, w23, w33]])

    cost = []
    # Construct the cost function
    for i in range(len(X)):
        m_i = np.array([X[i], Y[i], 1])
        if np.matmul(np.matmul(m_i, G), m_i)**2 > np.matmul(np.matmul(m_i, C), m_i)*np.matmul(np.matmul(m_i, W), m_i):
            cost.append(np.matmul(np.matmul(m_i, C), m_i)**2/((1+np.sqrt((np.matmul(np.matmul(
                m_i, G), m_i)**2-np.matmul(np.matmul(m_i, C), m_i)*np.matmul(np.matmul(m_i, W), m_i))
                / (np.matmul(np.matmul(m_i, G), m_i))**2))**2*np.matmul(np.matmul(m_i, G), m_i)))
        else:
            cost.append(1/4 * np.matmul(np.matmul(m_i, C), m_i)**2 /
                        (np.matmul(np.matmul(m_i, G), m_i)))
    cost = np.array(cost)
    return cost


# Least Levenberg-Marquadt minimization on cost function, yield parameters
ls = least_squares(cost, param_list, method='lm', args=(X, Y))
op_param = ls.x
op_param[1], op_param[2], op_param[4] = 2 * \
    op_param[1], 2*op_param[2], 2*op_param[4]


print(op_param)

plt.scatter(X, Y)
y, x = np.ogrid[0:5:1000j, 0:5:1000j]
plt.contour(x.ravel(), y.ravel(), op_param[0]*pow(x, 2)+op_param[1] *
            x*y+op_param[2]*x+op_param[3]*pow(y, 2)+op_param[4]*y+op_param[5], [0])
plt.grid()
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.ylabel('y')
plt.xlabel('x')
plt.show()
