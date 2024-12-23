import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize


class GradientFind():
  def __init__(self, f, grad_f=None, initial_params=np.array([2, 2]), treshold=0.1, learning_rate=0.1,
               animation_interval_ms=1000, grad_approximetely=False, metod='classic', lr_dynamic=False) -> None:
    self.x, self.y = sp.symbols('x y')
    if callable(f):
      self.function = f
    else:
      self.is_multivariable = True
      formula = sp.sympify(f)
      self.function = sp.lambdify((self.x, self.y), formula)

    if grad_f == None:
      # assert type(f) is str, "Variable is not of type string!"
      if type(f) is str:
        self.grad_f = self.find_grad(f=f)
      else:
        self.grad_f = None
    else:
      self.grad_f = grad_f
    self.treshold = treshold
    self.learning_rate = learning_rate
    self.animation_interval_ms = animation_interval_ms
    self.initial_params = initial_params
    self.grad_approximetely = grad_approximetely
    self.metod = metod
    self.lr_dynamic = lr_dynamic

  def grad(self,
           # point: np.ndarray,
           x, y,
           dt: float = 0.0001) -> np.ndarray:
    dx = (self.function(x - dt, y) - self.function(x, y)) / dt
    dy = (self.function(x, y - dt) - self.function(x, y)) / dt

    return np.array([-dx, -dy])

  def find_grad(self, f):
    gradient = [sp.diff(sp.sympify(f), var) for var in (self.x, self.y)]
    numgradfun = [sp.lambdify([self.x, self.y], g) for g in gradient]
    return numgradfun

  def func_to_minimize(self, vars):
    return self.function(vars[0], vars[1])

  def find(self):
    x0 = self.initial_params
    x_values = [x0]
    prev = x0
    i = 0

    beta = 0.5

    m = np.array([0, 0])
    v = np.array([0, 0])
    b1 = 0.6
    b2 = 0.999
    e = 10e-8
    d = 0.001

    while True:
      if self.grad_approximetely == True:
        grad = self.grad(x0[0], x0[1])
      else:
        grad = np.array([g(*x0) for g in self.grad_f])
      # print(grad)

      if self.metod == 'classic':
        x_new = x0 - self.learning_rate * grad
      elif self.metod == 'momentum':
        x_new = x0 - self.learning_rate * grad + beta * (x0 - prev)
        prev = x0
      elif self.metod == 'adaptive':
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2

        x_new = x0 - self.learning_rate * m / (np.sqrt(v) + e)

      x_values.append(x_new)
      if np.linalg.norm(x_new - x0) < self.treshold:
        break
      x0 = x_new
      i += 1

      if self.lr_dynamic:
        self.learning_rate = self.learning_rate / (1 + i * d)

    min_finded = x_values[-1]
    print(f'Кол-во шагов = {i}')
    print(f'min -> x = {round(min_finded[0], 5)}, y = {round(min_finded[1], 5)}')
    self.x_values = x_values
    result = minimize(self.func_to_minimize, x0)
    y_find = self.function(min_finded[0], min_finded[1])
    error = abs(result.fun - y_find)
    print(f'C scipy ~= Аналитическое решение -> {result.x}')
    print(f'Ошибка = {error}')
    self.result_analitics = result

  def plot_simple(self):
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = self.function(X, Y)

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50)
    plt.scatter([x[0] for x in self.x_values], [x[1] for x in self.x_values], c='red', label='Gradient Descent')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_animation(self):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = self.function(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    points, = ax.plot([], [], [], 'ro')  # Points of descent
    arrows = []

    def init():
      points.set_data([], [])
      points.set_3d_properties([])
      return points,

    def update(frame):
      if frame < len(self.x_values) - 1:
        x_start = self.x_values[frame][0]
        y_start = self.x_values[frame][1]
        z_start = self.function(x_start, y_start)

        x_end = self.x_values[frame + 1][0]
        y_end = self.x_values[frame + 1][1]
        z_end = self.function(x_end, y_end)

        points.set_data([x[0] for x in self.x_values[:frame + 1]], [x[1] for x in self.x_values[:frame + 1]])
        points.set_3d_properties([self.function(x[0], x[1]) for x in self.x_values[:frame + 1]])

        ax.quiver(x_start, y_start, z_start, x_end - x_start, y_end - y_start, z_end - z_start,
                  color='green', arrow_length_ratio=0.1)

      return points,

    ani = FuncAnimation(fig, update, frames=len(self.x_values), init_func=init,
                        blit=True, repeat=False, interval=self.animation_interval_ms)

    ax.scatter(self.x_values[-1][0], self.x_values[-1][1],
               self.function(self.x_values[-1][0], self.x_values[-1][1]),
               color='green', s=100, label='Optimum Fineded')
    ax.text(self.x_values[-1][0], self.x_values[-1][1], self.function(self.x_values[-1][0], self.x_values[-1][1]),
            'Optimum Fineded', fontsize=10, ha='right', color='green', fontweight='bold')
    x = self.result_analitics.x[0]
    y = self.result_analitics.x[1]
    minimum_func = self.function(x, y)
    ax.scatter(x, y, minimum_func, color='blue', s=100,
               label='Optimum')
    ax.text(x, y, minimum_func, 'Optimum', fontsize=10, ha='left', color='blue',
            fontweight='bold')

    plt.title('Gradient Descent Visualization')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis (Function value)')
    plt.show()



ex = GradientFind(f='sin(x + y) + (x-y)**2 -1.5*x + 2.5*y + 1', initial_params=np.array([-5,5]),
                  treshold=0.001, learning_rate = 0.1, animation_interval_ms=1000, grad_approximetely=False,
                  metod='adaptive', lr_dynamic=True)
ex.find()
ex.plot_animation()



ex = GradientFind(f='0.26*(x**2+y**2) - 0.48*x*y', initial_params=np.array([-5,5]), treshold=0.001, learning_rate = 0.01, animation_interval_ms=10)
ex.find()
ex.plot_animation()



ex = GradientFind(f='sin(x + y) + (x-y)**2 -1.5*x + 2.5*y + 1', initial_params=np.array([-5,5]),
                  treshold=0.001, learning_rate = 0.01, animation_interval_ms=200)
ex.find()
ex.plot_animation()


# ex = GradientFind(f='(x**2 + y - 11)**2 + (x + y**2 - 7)**2', initial_params=np.array([-5,5]), treshold=0.001, learning_rate = 0.01, animation_interval_ms=1000)
# ex.find()
# ex.plot_animation()





