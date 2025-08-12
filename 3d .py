import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Quadratic bowl: f(x,y) = x^2 + y^2 (convex surface)
def f(X, Y):
    return X**2 + Y**2

# Analytic gradient for a descent step
def grad(x, y):
    return np.array([2*x, 2*y])

# Meshgrid for surface/contours
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Choose a starting point and one gradient step
p0 = np.array([6.0, 6.0])
g0 = grad(*p0)
alpha = 0.08                            # step size
p1 = p0 - alpha * g0                    # new point after one step

# Figure with 3D surface
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Surface with colormap and colorbar
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.95)
fig.colorbar(surf, ax=ax, shrink=0.6, label='f(x, y)')

# Plot the current point and descent arrow on the surface
z0 = f(p0[0], p0[1])
z1 = f(p1, p1[1])
ax.scatter(p0, p0[1], z0, c='red', s=60)           # red point
ax.quiver(p0, p0[1], z0,
          (p1-p0), (p1[1]-p0[1]), (z1 - z0),
          color='yellow', length=1.0, normalize=False)

# Optional: show contour projection on the "floor"
ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis, levels=20)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z = x^2 + y^2')
ax.set_title('3D Surface with Gradient Descent Step')

# Put the floor at z=0 so the projection is visible
ax.set_zlim(0, Z.max())
plt.tight_layout()
plt.show()

