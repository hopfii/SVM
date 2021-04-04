from succinctly.datasets import get_dataset, linearly_separable as ls
import cvxopt.solvers
import numpy as np


# calculate bias as average over all support vectors
def compute_b_avg(w, X, y):
    return np.sum([y[i] - np.dot(w, X[i]) for i in range(len(X))]) / len(X)

# calculate bias from one support vector
def compute_b(w, X, y):
    return ((1/y) - np.dot(w, np.transpose(X)))

# calculate w for svm
def compute_w(multipliers, X, y):
    return np.sum(multipliers[i] * y[i] * X[i] for i in range(len(y)))

# load dataset
# X, y = get_dataset(ls.get_training_examples)

X = np.array([[1, 2], [2, 1], [3, 3], [4, 1],  [6, 2], [4, 9], [2, 10], [3, 9], [4, 8], [5, 8], [6, 9], [7, 5]])
y = np.array([1., 1., 1., 1., 1., -1., -1., -1., -1., -1., -1., -1.])
m = X.shape[0]
print(X)
print(y)

# Gram matrix - The matrix of all possible inner products of X.
K = np.array([np.dot(X[i], X[j])
 for j in range(m)
 for i in range(m)]).reshape((m, m))
Q = cvxopt.matrix(np.outer(y, y) * K)
print("K")
print(K)
print("outer")
print(np.outer(y, y))
print("Q")
print(Q)
q = cvxopt.matrix(-1 * np.ones(m))
print("c")
print(q)

# equality constraints (form: Ax = b)
# for SVM problem: y^T alpha = 0
A = cvxopt.matrix(y, (1, m))
b = cvxopt.matrix(0.0)
print("Equality constraints")
print(A)
print(b)

# Inequality constraints: (Gx <= h)
# for SVM problem:: 0 <= alpha <= inf
# (-1 * alpha) <= 0 -> violated for alpha < 0
G = cvxopt.matrix(np.diag(-1 * np.ones(m)))
h = cvxopt.matrix(np.zeros(m))
print("Inequality constraints")
print(G)
print(h)

# solve problem
solution = cvxopt.solvers.qp(Q, q, G, h, A, b)

# get lagrange multipliers
multipliers = np.ravel(solution['x'])
print("alpha")
print(multipliers)

# support vectors have positive lagrange multiplier
has_positive_multiplier = multipliers > 1e-7
print("Pos multipliers")
print(has_positive_multiplier)
sv_multipliers = multipliers[has_positive_multiplier]
print("sv mults")
print(sv_multipliers)
support_vectors = X[has_positive_multiplier]
print("Support vectors")
print(support_vectors)
support_vectors_y = y[has_positive_multiplier]
print("Support vectors y")
print(support_vectors_y)

w = compute_w(multipliers, X, y)
w_from_sv = compute_w(sv_multipliers, support_vectors, support_vectors_y)
print("w")
print(w)
print("w from support vectors:")
print(w_from_sv)

b = compute_b(w, support_vectors[0], support_vectors_y[0]) 
print("b calculated from single support vector")
print(b)
print("b calculated as avg over all support vectors")
print(compute_b_avg(w, support_vectors, support_vectors_y))



