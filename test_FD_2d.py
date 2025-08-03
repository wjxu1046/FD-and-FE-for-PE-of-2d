import torch
import numpy as np


def u_exact(x, y):
    return torch.matmul(torch.sin(np.pi * x).unsqueeze(-1), torch.sin(np.pi * y).unsqueeze(0)).reshape((M+1)*(N+1))

def f(x, y):
    return 2 * np.pi ** 2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)

def matrix_PE(M, N, f):
    A = torch.zeros((M+1) * (N+1), (M+1) * (N+1))
    F = torch.zeros((M+1) * (N+1))
    hx = 1.0/M
    hy = 1.0/N
    c = hx / hy
    x = torch.linspace(0,1,M+1)
    y = torch.linspace(0,1,N+1)
    for i in range(M+1):
         for j in range(N+1):
            # if i == 0 :
            #     A[j, j] == 1
            # if j == 0 :
            #     A[i * (N+1)] == 1
            # if i == M :
            #     A[M * (N+1) + j, M * (N+1) + j]
            # if j == N :
            #     A[i * (N+1) + N, i * (N+1) + N]
            if i == 0 or j == 0 or i == M or j == N:
                A[i * (N+1) + j, i * (N+1) + j] = 1
                F[i * (N+1) + j] = 0
            else:
                A[i * (N+1) + j,i * (N+1) + j] = 2 * (c + 1/c)
                A[i * (N+1) + j,(i-1) * (N+1) + j] = -1/c
                A[i * (N+1) + j,(i+1) * (N+1) + j] = -1/c
                A[i * (N+1) + j,i * (N+1) + j-1] = -c
                A[i * (N+1) + j,i * (N+1) + j+1] = -c
                F[i * (N+1) + j] = f(x[i], y[j]) * hx * hy
                
    return A, F

def solve_PE(M,N,f):
    A, F = matrix_PE(M,N,f)
    u_sol = torch.linalg.solve(A,F)
    return u_sol

M = 20
N = 20
x = torch.linspace(0,1,M+1)
y = torch.linspace(0,1,N+1)
# A,F = matrix_PE(N,f)
# print(A)
# print(F)
u_pre = solve_PE(M,N,f)
u_true = u_exact(x,y)
# print(u_pre)
# print(u_true)
error = torch.max(torch.abs(u_true-u_pre))

print(error)


import matplotlib.pyplot as plt
x = x.numpy()
y = y.numpy()
u_pre = u_pre.numpy()
u_true = u_true.numpy()

from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(x, y)

u_pre_2d = u_pre.reshape((M+1, N+1))
u_true_2d = u_true.reshape((M+1, N+1))
error_2d = np.abs(u_pre_2d - u_true_2d)


# 预测解可视化
fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, u_pre_2d.T, cmap='viridis')
ax1.set_title("Predicted Solution $u_{pre}$")

# 精确解可视化
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, u_true_2d.T, cmap='viridis')
ax2.set_title("Exact Solution $u_{exact}$")

# 误差可视化
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, error_2d.T, cmap='hot')
ax3.set_title("Absolute Error $|u_{pre} - u_{exact}|$")

plt.tight_layout()
plt.show()

# plt.figure()
# plt.plot(x,u_pre,'r',label = 'pre solution')
# plt.plot(x,u_true,'b',label = 'true solution')
# plt.title('solution of PE for FD of 1d')
# plt.legend()
# plt.show()