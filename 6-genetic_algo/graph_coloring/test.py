import numpy as np
import matplotlib.pyplot as plt
import random

def divide_circle(radius, n):
    theta = 2 * np.pi / n  # Angle between points
    points = [(radius * np.cos(i * theta), radius * np.sin(i * theta)) for i in range(n)]
    #points = []
    r = random.random() * random.choice([1,1])
    for i in range(n):
        #r = random.random() * random.choice([1,-1])
        x = radius * np.cos(i * theta + r)
        y = radius * np.sin(i * theta + r)
        points.append((x, y))
    points = [(round(x, 2), round(y, 2)) for x, y in points]
    return points

# Example usage
radius = 10
n = 90
division_points = divide_circle(1, n)
#division_points.extend(divide_circle(4, 6))
#division_points.extend(divide_circle(7, 9))
#division_points.extend(divide_circle(10, 12))
print(division_points)

x, y = zip(*division_points)
plt.figure(figsize=(1, 1))
plt.scatter(x[:n], y[:n], color='red')
plt.scatter(x[n:], y[n:], color='blue')
plt.show()