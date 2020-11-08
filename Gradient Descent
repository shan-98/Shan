%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def f(x1, x2):
  '''
  Function definition
  '''
  return x1*x1 + (x2-2)*(x2-2)
  
#plotting function in wireframe visual

limit = 3.0
x1 = np.linspace(-limit, limit, 100)
x2 = np.linspace(-limit, limit, 100)

X1, X2 = np.meshgrid(x1, x2)
Y = np.square(X1) + np.square(X2-2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X1, X2, Y, rstride=10, cstride=10)

plt.show()

def gd(p_0, max_iter, rho, threshold=1.0e-4):
  '''
  Gradient descent algorithm
  Args:
    p_0: initial point
    max_iter: maximum iteration
    rho: learning rate
    threshold: threshold to check convergence on function value
  Returns:
    obtained_points: return the obtained points for the gradient descent
    f_value: function value for each iteration
    t: number of iteration
  '''
  obtained_points = []
  f_value = []
  
  x1_t, x2_t = p_0[0], p_0[1]
  
  obtained_points.append((x1_t, x2_t))
  f_value.append(f(x1_t, x2_t))

  for t in range(max_iter):
    x1_t_1 = x1_t - rho * 2 * x1_t
    x2_t_1 = x2_t - rho * 2 * (x2_t-2)
    
    obtained_points.append((x1_t_1, x2_t_1))
    
    f_value.append(f(x1_t_1, x2_t_1))
    
    if abs(f_value[t+1]-f_value[t]) < threshold:
      break
    
    x1_t = x1_t_1
    x2_t = x2_t_1
  
  if t == max_iter-1:
    print('Reached maximum iteration in Gradient Descent')
  
  return obtained_points, f_value, t+1
  
def f_plot(f_values, title):
  '''
  Plots function value at different iterations given the function values
  
  Args:
    f_value: function value at different iterations
  '''
  plt.plot(f_values)
  plt.title(title)
  plt.xlabel('iteration')
  plt.ylabel('function value')
  plt.show()

def gd_contour_plot(X1, X2, Y, op):
  
  '''
  Draws the function on 2d with contour and shows gradient convergence path
   
  Args:
    X1, X2: function argument value in meshgrid
    Y: function value for X1, X2 values
    op: obtained points in gradient descent
  '''
  contour_levels = [x*x for x in np.linspace(0, 10, 15)]
  plt.axhline(0, color='black', alpha=.5, dashes=[2, 4],linewidth=1)
  plt.axvline(0, color='black', alpha=0.5, dashes=[2, 4],linewidth=1)
  cp = plt.contour(X1, X2, Y, contour_levels, colors='black', linestyles='dashed', linewidths=1)
  plt.clabel(cp, inline=1, fontsize=10)
  cp = plt.contourf(X1, X2, Y, contour_levels, alpha=0.8)
  
  for i in range(len(op) - 1):
    plt.annotate('', xy=op[i+1], xytext=op[i], arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1}, va='center', ha='center')
  
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.title('Sequence of points')
  plt.show()
  
#Testing the model with learning rate p = 0.01, initial point (x1, x2) = (1, 1)
  
p_0 = (1,1)
rho = 0.01
max_iter = 10000
op, fv, t = gd(p_0, max_iter, rho)

title = 'Converging at iteration: '+str(t)
f_plot(fv, title)

gd_contour_plot(X1, X2, Y, op)

#testing the model with learning rate p = 0.1, initial point (x1, x2) = (1, 1)

p_0 = (1,1)
rho = 0.1
max_iter = 10000
op, fv, t = gd(p_0, max_iter, rho)

title = 'Converging at iteration: '+str(t)
f_plot(fv, title)

gd_contour_plot(X1, X2, Y, op)
