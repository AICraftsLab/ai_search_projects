import numpy as np

def divide_circle(radius, n):
    theta = 2 * np.pi / n  # Angle between points
    points = [(radius * np.cos(i * theta), radius * np.sin(i * theta)) for i in range(n)]
    points = [(round(x), round(y)) for x, y in points]
    return points

# Example usage
radius = 10
n = 8
division_points = divide_circle(radius, n)
print(division_points)
