#import of nessecary packages
import numpy as np
from matplotlib import pyplot as plt

#defining number of trails and create empty lists
number_of_trials = 100000
n_points_in_circle = 0
n_points_outside_circle = 0

xs = []
ys = []

#filling in points
for _ in range(number_of_trials):
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    xs.append(x)
    ys.append(y)

#finding the distance from the origin of the circle
for i in range(number_of_trials):
    distance_from_origin = np.sqrt(xs[i]**2 + ys[i]**2)
    if distance_from_origin <= 1:
        n_points_in_circle += 1
    else:
        n_points_outside_circle += 1
        
#pi is the ratio of points in the circle and outside of the circle times four
pi_estimate = (n_points_in_circle / number_of_trials) * 4
print(f"Estimated value of pi: {pi_estimate}")

#plotting the points and the circle
plt.figure(figsize=(6, 6))
plt.scatter(xs, ys, c=['blue' if np.sqrt(xs[i]**2 + ys[i]**2) <= 1 else 'red' for i in range(number_of_trials)], s=1)
circle = plt.Circle((0, 0),1, color='black', fill=False)
plt.gca().add_artist(circle)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title('Monte Carlo Simulation Estimating the Value of Pi \n Value of Pi: {:.4f}'.format(pi_estimate) + '\n Number of Trials: {}'.format(number_of_trials))
plt.xlabel('x')
plt.ylabel('y')
plt.show()