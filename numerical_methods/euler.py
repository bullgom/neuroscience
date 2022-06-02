import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

"""
https://pythonnumericalmethods.berkeley.edu/notebooks/chapter22.03-The-Euler-Method.html

S(t_i+1) = S(t_i) + h * S(t_i)

Approximate the solution to the initial value problem between 0 and 1 in increments of 0.1 
using Explicit Euler Method
Plot the difference between the approximated solution and the exact solution

Equation: df(t)/dt = e^(-t)
Initial Condition: f_0 = -1
Exact Solution: f(t) = -e^(-t)
"""

plt.style.use("seaborn-poster")

def euler(y: float, h: float, t:float, f: Callable[[float], float]) -> float:
    
    return y + h * f(t)

def dfdt(t: float) -> float:
    return np.exp(-t)

def exact(t: float) -> float:
    return - np.exp(-t)

f_0 = -1
step = 0.1

start = 0
end = 1

x = np.arange(start, end+step, step)

exact_y = [exact(t) for t in x]

approx_y = [f_0]

y_last = f_0
for i in range(len(x)-1):
    t = x[i]
    y_last = approx_y[i]
    y_t = euler(y_last, step, t, dfdt)
    approx_y.append(y_t)

plt.plot(x, exact_y, label="Exact")
plt.plot(x, approx_y, "bo--", label="Approximate")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.grid()
plt.show()
print((exact_y))    
print((approx_y))    
    

