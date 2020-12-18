import numpy as np
import matplotlib.pyplot as plt

import splineTools

# hard code file path for now
filePath = "C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/303-POST/outsidePoints/combined_slice_2.txt"

# read points from file
points = np.loadtxt(filePath)

# perform angular reordering of points
orderedPoints = splineTools.reOrder(points)

# copy another point to the end for a closed spline curve
orderedPoints = np.append(orderedPoints, [orderedPoints[0]], 0)
numDataPoints = len(orderedPoints)

# set up parameters for spline fit
numControlPoints = 25
n = numControlPoints - 1
degree = 3
numCalcControlPoints = numControlPoints + degree

# generate the knots (numKnots = n + 2d + 2)
numKnots = n + 2*degree + 2
tau = np.zeros(32)
numOpenKnots = n + degree + 1
tau[0:numOpenKnots] += np.linspace(0, 1, numOpenKnots)
for i in range(0, degree + 1):
    diff = tau[i + 1] - tau[i]
    tau[numOpenKnots + i] = tau[numOpenKnots + i - 1] + diff

# set up parameterization
t = splineTools.parameterizeClosedCurve(orderedPoints, tau, degree)

p_mat = np.transpose(orderedPoints)
A_mat = np.zeros((numDataPoints, numCalcControlPoints))

for j in range(numDataPoints):
    for k in range(numCalcControlPoints):
        A_mat[j][k] = splineTools.NVal(tau, t[j], k-1, degree, 0)

# create a constrained A matrix
A_mat_con = A_mat
A_mat_con[:, 0:degree] = A_mat_con[:, 0:degree] + A_mat_con[:, (numCalcControlPoints-degree):numCalcControlPoints]
A_mat_con = A_mat_con[:, 0:numControlPoints]

# solve matrix equations for control points
b_mat = np.linalg.lstsq(A_mat_con, orderedPoints)[0]
b = np.transpose(b_mat)

# duplicate last 'degree' control points
new_b = np.zeros((2, numCalcControlPoints))
new_b[:, 0:numControlPoints] = b
new_b[:, numControlPoints:numControlPoints+degree] = b[:, 0:degree]
b = new_b

# calculate the spline
# TODO figure out why the first 4 values (and last 4?) returned by this are incorrect
interpCurve = splineTools.BSVal(b, tau, t, 0)

# plot everything
plt.plot(orderedPoints[:, 0], orderedPoints[:, 1], 'x', b[0, :], b[1, :], 'o', interpCurve[0, :], interpCurve[1, :])
plt.title('Outside points b-spline fit')
plt.legend(('data points', 'control points', 'B-spline curve'))
plt.xlim((min(b[0, :])-10, max(b[0, :])+10))
plt.ylim((min(b[1, :])-10, max(b[1, :])+10))
plt.show()

# TODO calculate error in the fit
errorVector = interpCurve - np.transpose(orderedPoints)
errorInFit = np.sum(errorVector[0, :]**2 + errorVector[1, :]**2)

