import numpy as np
import matplotlib.pyplot as plt
import random

def divide_circle(radius, n):
    theta = 2 * np.pi / n  # Angle between points
    # points = [(radius * np.cos(i * theta), radius * np.sin(i * theta)) for i in range(n)]
    points = []
    r = 0 #random.random() * random.choice([1, -1])
    for i in range(n):
        x = radius * np.cos(i * theta + r)
        y = radius * np.sin(i * theta + r)
        points.append((x, y))
    points = [(round(x, 3), round(y, 3)) for x, y in points]
    return points

# Example usage
radius = 10
n = 90
division_points = divide_circle(1, 3)
division_points = divide_circle(1, 3)
division_points.extend(divide_circle(4, 6))
division_points.extend(divide_circle(7, 9))
division_points.extend(divide_circle(10, 12))
# print(division_points)

x, y = zip(*division_points)
# plt.figure()
plt.scatter(x[:n], y[:n], color='red')
plt.scatter(x[n:], y[n:], color='blue')
plt.show()