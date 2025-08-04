import torch
import numpy as np


def u_exact(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)


def f(x, y):
    return 2 * np.pi ** 2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)


def generate_mesh(M, N):
    x = torch.linspace(0, 1, M + 1)
    y = torch.linspace(0, 1, N + 1)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    
    elements = []
    for i in range(M):
        for j in range(N):
            # 按行编号
            n0 = i * (N + 1) + j
            n1 = n0 + 1
            n2 = n0 + (N + 1)
            n3 = n2 + 1
            elements.append([n0, n1, n3])  # upper right triangle
            elements.append([n0, n3, n2])  # lower left triangle
            
            # # 按列编号
            # n0 = i  + j *(M + 1)
            # n1 = n0 + 1
            # n2 = n0 + (M + 1)
            # n3 = n2 + 1
            # elements.append([n0, n1, n3])  # upper right triangle
            # elements.append([n0, n3, n2])  # lower left triangle
            
            
            
    return points, torch.tensor(elements, dtype=torch.long)

# 每个三角形上的局部刚度矩阵与面积
def local_stiffness(p1, p2, p3):
    mat = torch.stack([p2 - p1, p3 - p1], dim=1)  # shape: [2,2]
    area = 0.5 * torch.abs(torch.linalg.det(mat))
    inv_mat_T = torch.linalg.inv(mat.T)
    grad_phi = torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    grads = grad_phi @ inv_mat_T
    A_local = area * (grads @ grads.T)
    return A_local, area

# 全局组装（稠密）
def assemble_global(M, N):
    points, elements = generate_mesh(M, N)
    n_pts = points.shape[0]
    A = torch.zeros((n_pts, n_pts), dtype=torch.float32)
    F = torch.zeros(n_pts, dtype=torch.float32)

    for elem in elements:
        p = points[elem]  # shape: [3,2]
        A_loc, area = local_stiffness(p[0], p[1], p[2])
        xc, yc = p.mean(dim=0)
        f_val = f(xc, yc)
        for i in range(3):
            F[elem[i]] += f_val * area / 3
            for j in range(3):
                A[elem[i], elem[j]] += A_loc[i, j]
    
    # Dirichlet 边界条件
    boundary_mask = (points[:, 0] == 0) | (points[:, 0] == 1) | \
                    (points[:, 1] == 0) | (points[:, 1] == 1)
    for idx in torch.where(boundary_mask)[0]:
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        F[idx] = 0.0

    return A, F, points

# 解线性系统
def solve_poisson_fem(M, N):
    A, F, points = assemble_global(M, N)
    u = torch.linalg.solve(A, F)
    return u, points

# 主调用
M = 20
N = 20
u_fem, pts = solve_poisson_fem(M, N)
u_true = u_exact(pts[:, 0], pts[:, 1])
error_max = torch.max(torch.abs(u_fem - u_true))
print("Max error:", error_max.item())

import matplotlib.pyplot as plt

# 可视化
u_fem_grid = u_fem.reshape((M+1, N+1))
u_true_grid = u_true.reshape((M+1, N+1))
err_grid = torch.abs(u_fem_grid - u_true_grid)

X = pts[:, 0].reshape(M+1, N+1).numpy()
Y = pts[:, 1].reshape(M+1, N+1).numpy()
u_fem_np = u_fem_grid.numpy()
u_true_np = u_true_grid.numpy()
err_np = err_grid.numpy()

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, u_fem_np, cmap='viridis')
ax1.set_title("FEM Solution (Torch)")

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, u_true_np, cmap='viridis')
ax2.set_title("Exact Solution")

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, err_np, cmap='hot')
ax3.set_title("Absolute Error")

plt.tight_layout()
plt.show()
