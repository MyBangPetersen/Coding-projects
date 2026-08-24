#script to do different kinds of interpolations of data
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator


#two lists that we would like to have more points
x = np.array([2, 4, 5, 7, 8, 10, 11, 15, 16, 17, 20])
y = np.array([1, 2, 4, 5, 6, 7, 10, 14, 17, 18, 21])

#or a function
xf = np.linspace(0, 10, num = 15)
function = np.cos(-xf**2 / 9.0)

#beginning with linear interploation
#scipy has a package for this

def interpolate_lin(data_x, data_y, point):
    interpolation = interp1d(data_x, data_y)
    interpolation_val = interpolation(point)
    return interpolation, interpolation_val

print(interpolate_lin(x, y, 2.5))
#this works for a single point where you want to know the interpolation
#we want to make a solution that works for an entire data set and can do multiple interpolations for every step
#using numpys interpolation tool instead

def interpolate_lin_np(data_x, data_y, num_points):
    x_new = np.linspace(data_x[0], data_x[-1], num = num_points)
    interpolation = np.interp(x_new, data_x, data_y)
    return x_new, interpolation

interp = interpolate_lin_np(xf, function, 100)
interp0 = interpolate_lin_np(xf, function, 20)

plt.plot(xf, function, 'o', c = 'g', label = 'data, 15 points')
plt.plot(interp0[0], interp0[1], '-', c = 'r', label = 'interpolation, 20 points')
plt.plot(interp[0], interp[1], '-', c = 'b', label = 'interpolation, 50 points')
plt.legend(loc='best')
plt.title('An example of linear interpolation')
plt.grid()
plt.show()

#moving on to other non-linear interpolations
#cubic splines produces a more rounded curve instead of having sharp corners where the lines meet (piecewise polynomials)
#the spline has a nu-argument to see the derivative of the splines

def cubic_spline(data_x, data_y, num_points, nu):
    data_x_new = np.linspace(data_x[0], data_x[-1], num_points)
    spline = CubicSpline(data_x, data_y)
    splines = spline(data_x_new, nu)
    return data_x_new, splines


interp_cubic = cubic_spline(xf, function, 50, 0)

plt.plot(xf, function, 'o', c = 'g', label = 'data')
plt.plot(interp_cubic[0], interp_cubic[1], '-', c = 'r', label = 'interpolation')
plt.legend(loc='best')
plt.title('An example of spline interpolation')
plt.grid()
plt.show()

#plt.plot(xf, function, 'o', c = 'g', label = 'data')
plt.plot(cubic_spline(xf, function, 50, 1)[0], cubic_spline(xf, function, 50, 1)[1], '-', c = 'r', label = 'interpolation, nu = 1')
plt.plot(cubic_spline(xf, function, 50, 2)[0], cubic_spline(xf, function, 50, 2)[1], '-', c = 'g', label = 'interpolation, nu = 2')
plt.plot(cubic_spline(xf, function, 50, 3)[0], cubic_spline(xf, function, 50, 3)[1], '-', c = 'b', label = 'interpolation, nu = 3')
plt.legend(loc = 'best')
plt.grid()
plt.title('An example of spline interpolation \n with different values of nu')
plt.show()

#we now want to look at a dataset with an outlier different from all the other points
#this is handled by monotone cubic that are only once differentiable to preserve the shape of the data 
#scipt has two possibilities for handeling this: PchipInterpolator and Akima1DInterpolator

#data with an outlier
xs = np.array([1., 2., 3., 4., 4.5, 5., 6., 7., 8])
ys = xs**2
ys[4] += 101

def akima(data_x, data_y, num_points):
    data_x_new = np.linspace(data_x[0], data_x[-1], num_points)
    akima = Akima1DInterpolator(data_x, data_y)
    interpolation = akima(data_x_new)
    return data_x_new, interpolation

def pchip(data_x, data_y, num_points):
    data_x_new = np.linspace(data_x[0], data_x[-1], num_points)
    pchip = PchipInterpolator(data_x, data_y)
    interpolation = pchip(data_x_new)
    return data_x_new, interpolation

plt.plot(xs, ys, 'o', c = 'k', label = 'data with outlier')
plt.plot(cubic_spline(xs, ys, 50, 0)[0], cubic_spline(xs, ys, 50, 0)[1], '.-.', c = 'g', label = 'spline interpolation')
plt.plot(akima(xs, ys, 50)[0], akima(xs, ys, 50)[1], '-', c = 'r', label = 'akima interpolation')
plt.plot(pchip(xs, ys, 50)[0], pchip(xs, ys, 50)[1], '--', c = 'b', label = 'pchip interpolation')
plt.title('Example where akima and pchip works better than splines')
plt.legend(loc = 'best')
plt.grid()
plt.show()
