import numpy as np
import matplotlib.pyplot as plt

def divide_circle(radius, n):
    theta = 2 * np.pi / n  # Angle between points
    points = [(radius * np.cos(i * theta), radius * np.sin(i * theta)) for i in range(n)]
    points = [(round(x, 1), round(y, 1)) for x, y in points]
    return points

# Example usage
radius = 10
n = 360
division_points = divide_circle(1, 3)
division_points.extend(divide_circle(4, 9))
division_points.extend(divide_circle(7, 27))
division_points.extend(divide_circle(10, 81))
print(division_points)

x, y = zip(*division_points)
plt.figure(figsize=(1, 1))
plt.scatter(x, y)
plt.show()