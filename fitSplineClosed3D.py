import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time

import splineTools

# define parameters for reading from file (hardcoded for now, but should be easy to integrate into PATS)
fileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/303-POST/outsidePoints/combined_slice_'
fatName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/303-POST/outsidePoints/fat_slice_'
startFrame = 2
stopFrame = 7
numSlices = (stopFrame - startFrame) + 1

# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(fileName, startFrame, stopFrame)

# resample the data so each slice has the same number of points
# do this by fitting each slice with B-spline curve
resampleNumControlPoints = 7
degree = 3
resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
    splineTools.reSampleAndSmoothPoints(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)

########################################################################################################################

# # create synthetic tube data for testing
# numPointsPerContour = 257
# numSlices = 10
# r = 6
# newX, newY, newZ = splineTools.generateTubeData(numPointsPerContour, numSlices, r)
#
# # copy beginning point to end so curve will be closed
# x = np.zeros((numSlices, numPointsPerContour+1))
# x[:, :-1] = newX
# x[:, numPointsPerContour] = newX[:, 0]
#
# y = np.zeros((numSlices, numPointsPerContour+1))
# y[:, :-1] = newY
# y[:, numPointsPerContour] = newY[:, 0]
#
# z = np.zeros((numSlices, numPointsPerContour+1))
# z[:, :-1] = newZ
# z[:, numPointsPerContour] = newZ[:, 0]
#
# numPointsPerContour += 1
#
# # rename to use instead of data that would have been read in
# resampX = x
# resampY = y
# resampZ = z

########################################################################################################################
#plot the data
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(numSlices):
#     ax.plot(resampX[i, :], resampY[i, :], resampZ[i, :])
# plt.show()

# set up parameters for spline fit
numControlPointsU = 9
numControlPointsV = 6
degree = 3
numCalcControlPointsU = numControlPointsU + degree
m = numControlPointsU - 1
n = numControlPointsV - 1
M = numPointsPerContour - 1
N = numSlices - 1

# Figure out what set of two knots will be for u parameter direction (around each contour)
numKnotsU = m + 2*degree + 2
tauU = np.zeros(numKnotsU)
numOpenKnotsU = m + degree + 1
tauU[0:numOpenKnotsU] = np.linspace(0, 1, numOpenKnotsU)
for i in range(degree + 1):
    diff = tauU[i + 1] - tauU[i]
    tauU[numOpenKnotsU + i] = tauU[numOpenKnotsU + i - 1] + diff

numKnotsV = n + degree + 2
numInteriorKnotsV = numKnotsV - 2*degree
tauV = np.zeros(numKnotsV)
tauV[degree:numInteriorKnotsV+degree] = np.linspace(0, 1, numInteriorKnotsV)
tauV[numInteriorKnotsV+degree:numKnotsV] = np.ones(degree)

# set up parameterization
U, V, firstKnotU, lastKnotU, firstKnotV, lastKnotV = splineTools.parameterizeTube(resampX, resampY, resampZ,
                                                                                  tauU, tauV, degree)

# now we need to set up matrices to solve for mesh of control points
# (B*V*T^T = P)

# set up B matrix
B = np.zeros((M+1, numCalcControlPointsU))
for r in range(M+1):
    for i in range(numCalcControlPointsU):
        uVal = U[0, r]
        B[r, i] = splineTools.NVal(tauU, uVal, i-1, degree, 0)

# set up C matrix
C = np.zeros((N+1, n+1))
for s in range(N + 1):
    for j in range(n + 1):
        vVal = V[s, 0]
        C[s, j] = splineTools.NVal(tauV, vVal, j-1, degree, 0)

# now set up Px, Py, and Pz matrices
Px = np.transpose(resampX)
Py = np.transpose(resampY)
Pz = np.transpose(resampZ)

# constrain the B matrix so last three control points of each slice
# equal the first three (for cubic)
B_con = B
B_con[:, 0:degree] = B_con[:, 0:degree] + B_con[:, numCalcControlPointsU-degree:numCalcControlPointsU]
B_con = B_con[:, 0:numControlPointsU]


# calculate pseudo-inverses of B_con and C for use in generating control points
pinvB = np.linalg.pinv(B_con)
pinvC = np.linalg.pinv(np.transpose(C))

# solve for control points
Vx = np.matmul(pinvB, Px)
Vx = np.transpose(np.matmul(Vx, pinvC))

Vy = np.matmul(pinvB, Py)
Vy = np.transpose(np.matmul(Vy, pinvC))

Vz = np.matmul(pinvB, Pz)
Vz = np.transpose(np.matmul(Vz, pinvC))

# duplicate last 'degree' control points
newVx = np.zeros((numControlPointsV, numCalcControlPointsU))
newVy = np.zeros((numControlPointsV, numCalcControlPointsU))
newVz = np.zeros((numControlPointsV, numCalcControlPointsU))

newVx[:, 0:numControlPointsU] = Vx
newVy[:, 0:numControlPointsU] = Vy
newVz[:, 0:numControlPointsU] = Vz

newVx[:, numControlPointsU:numCalcControlPointsU] = Vx[:, 0:degree]
newVy[:, numControlPointsU:numCalcControlPointsU] = Vy[:, 0:degree]
newVz[:, numControlPointsU:numCalcControlPointsU] = Vz[:, 0:degree]

Vx = newVx
Vy = newVy
Vz = newVz

# evaluate closed spline to see what it looks like
lengthV, lengthU = np.shape(U)

# evaluate tensor product to get surface points. Operation is timed because it tends to be the slowest step
startTime = time.perf_counter()
X, Y, Z = splineTools.EvaluateTensorProduct(Vx, Vy, Vz, tauU, tauV, degree, U, V)
stopTime = time.perf_counter()
print("Tensor product evaluation took {} seconds".format(stopTime-startTime))


# calculate the error of the fit between surface and resampled data
errorInFitMatrix = (X - resampX)**2 + (Y-resampY)**2 + (Z-resampZ)**2
errorInSurfaceFit = np.sum(errorInFitMatrix)
totalResampleError = errorInSurfaceFit
maxErrorInSurfaceFit = np.max(errorInFitMatrix)
avgErrorInSurfaceFit = errorInSurfaceFit / (numSlices * numPointsPerContour)

########################################################################################################################
# plot all data on same scale
minX = np.min(Vx)
minY = np.min(Vy)
minZ = np.min(Vz)
maxX = np.max(Vx)
maxY = np.max(Vy)
maxZ = np.max(Vz)
azimuth = -40
elevation = 15

# # # if using the synthetic tube, set up number of points in each contour
# numPointsEachContour = np.ones((1, numSlices)) * numPointsPerContour
# origX = x
# origY = y
# origZ = z

# # plot the original data
# fig = plt.figure()
# ax = fig.add_subplot(2, 2, 1, projection='3d')
# ax.view_init(elevation, azimuth)
# ax.set_xlim(minX, maxX)
# ax.set_ylim(minY, maxY)
# ax.set_zlim(minZ, maxZ)
# ax.set_title('Original Data')
#
# for i in range(numSlices):
#     limit = numPointsEachContour[i]
#     #limit = numPointsPerContour
#     ax.plot(origX[i, 0:limit], origY[i, 0:limit], origZ[i, 0:limit])
#
# # plot the resampled data
# ax = fig.add_subplot(2, 2, 2, projection='3d')
# ax.view_init(elevation, azimuth)
# ax.set_xlim(minX, maxX)
# ax.set_ylim(minY, maxY)
# ax.set_zlim(minZ, maxZ)
# ax.set_title('Resampled Data; Error: {0}'.format(round(totalResampleError, 2)))
# for i in range(numSlices):
#     ax.plot(resampX[i, :], resampY[i, :], resampZ[i, :])
#
#
# # plot the control mesh
# ax = fig.add_subplot(2, 2, 3, projection='3d')
# ax.scatter(Vx, Vy, Vz, s=4)
# ax.set_title('Control point Mesh: {0}x{1}'.format(numControlPointsU, numControlPointsV))
#
# # add horizontal connections
# for i in range(numControlPointsV):
#     ax.plot(Vx[i, 0:numCalcControlPointsU], Vy[i, 0:numCalcControlPointsU], Vz[i, 0:numCalcControlPointsU], 'r')
#
# # add vertical connections
# for i in range(numCalcControlPointsU):
#     ax.plot(Vx[0:numControlPointsV, i], Vy[0:numControlPointsV, i], Vz[0:numControlPointsV, i], 'r')
#
# # now plot the surface
# ax = fig.add_subplot(2, 2, 4, projection='3d')
# ax.set_title('{0}x{1}'.format(lengthU, lengthV))  # can add error metrics to title later if necessary
# ax.plot_surface(X, Y, Z)
#
# # now that all subplots have been generated, display them on a single figure
# plt.show()

# plot control points and the surface on the same plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Combined plot')
ax.view_init(elevation, azimuth)
ax.set_xlim(minX, maxX)
ax.set_ylim(minY, maxY)
ax.set_zlim(minZ, maxZ)

# plot control points
ax.scatter(Vx, Vy, Vz, s=4)

# add horizontal connections
for i in range(numControlPointsV):
    ax.plot(Vx[i, 0:numCalcControlPointsU], Vy[i, 0:numCalcControlPointsU], Vz[i, 0:numCalcControlPointsU], 'r')

# add vertical connections
for i in range(numCalcControlPointsU):
    ax.plot(Vx[0:numControlPointsV, i], Vy[0:numControlPointsV, i], Vz[0:numControlPointsV, i], 'r')

# plot the surface
ax.plot_surface(X, Y, Z)

# read fat points from file and add them to the scene
fatX, fatY, fatZ, numFatPointsPerSlice = splineTools.readSlicePoints(fatName, startFrame, stopFrame)
ax.scatter(fatX, fatY, fatZ, marker='s', s=4, c='yellow')

# generate normal vectors
# crossX, crossY, crossZ = splineTools.generateNormalVectors(X, Y, Z)
crossX, crossY, crossZ = splineTools.simpleNormalVectors(X, Y, Z, numPointsPerContour, numSlices)
#ax.quiver(X, Y, Z, crossX, crossY, crossZ, length=10, color='purple', arrow_length_ratio=0.1)

# measure fat thickness at each normal vector
thicknessByPoint, xFatPoints, yFatPoints = splineTools.measureFatThickness(X, Y, crossX, crossY, fatX, fatY, numSlices,
                                                                           numPointsPerContour, numFatPointsPerSlice)


sortedX, sortedY, sortedCrossX, sortedCrossY = splineTools.pairNormalVectors(X, Y, crossX, crossY, numSlices,
                                                                             numPointsPerContour)

for j in range(numPointsPerContour):
    vector = ax.quiver(sortedX[:, j], sortedY[:, j], Z[:, j], crossX[:, j], crossY[:, j], crossZ[:, j],
                       length=10, color='purple', arrow_length_ratio=0.1)
    plt.draw()
    plt.pause(2)
    vector.remove()
# #plot each normal vector one at a time along with the fat points associated with it
#
#for i in range(numSlices):
#    count = 0
#    for j in range(numPointsPerContour):
#         x = xFatPoints[i, j, :]
#         y = yFatPoints[i, j, :]
#         z = fatZ[i]
#         points = ax.scatter(xFatPoints[i, j, :], yFatPoints[i, j, :], fatZ[i], s=4, c='black')
#         pointX = X[i, j] + 100*crossX[i, j]
#         pointY = Y[i, j] + 100*crossY[i, j]
#         pointZ = Z[i, j]
#         points2 = ax.scatter(pointX, pointY, pointZ, marker='s', s=4, c='purple')
#         vector = ax.quiver(X[i, j], Y[i, j], Z[i, j], crossX[i, j], crossY[i, j], crossZ[i, j], length=10,
#                            color='purple', arrow_length_ratio=0.1)
#         count += 1
#         message = "Slice {}, Point {}: Fat thickness of {}".format(i, j, thicknessByPoint[i, j])
#         plt.title(message)
#         plt.draw()
#         plt.pause(0.5)
#         points.remove()
#         points2.remove()
#         vector.remove()
#     print(count)

# display plot with control point mesh, surface, and fat points
#plt.show()

########################################################################################################################

# # get points that will be used to create fat spline surface
# fatSurfaceX, fatSurfaceY, fatSurfaceZ = splineTools.getFatSurfacePoints(thicknessByPoint, xFatPoints, yFatPoints, X, Y,
#                                                                         Z, numSlices, numPointsPerContour)

# generate a list of fat deposits to be converted into individual b-spline surfaces
# fatSurfaceX, fatSurfaceY, fatSurfaceZ = splineTools.getFatDeposits(thicknessByPoint, xFatPoints, yFatPoints, X, Y, Z,
#                                                                    numSlices, numPointsPerContour)

# create a spline surface for the fat that is open in both directions
# XFat, YFat, ZFat = splineTools.fitSplineOpen3D(fatSurfaceX, fatSurfaceY, fatSurfaceZ, numSlices, numPointsPerContour)
# XFat = []
# YFat = []
# ZFat = []
# for x, y, z in zip(fatSurfaceX, fatSurfaceY, fatSurfaceZ):
#     xf, yf, zf = splineTools.fitSplineOpen3D(x, y, z, numSlices, numPointsPerContour)
#     XFat.append(xf)
#     YFat.append(yf)
#     ZFat.append(zf)

# plot control points and the surface on the same plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Combined plot')
ax.view_init(elevation, azimuth)
ax.set_xlim(minX, maxX)
ax.set_ylim(minY, maxY)
ax.set_zlim(minZ, maxZ)

# plot control points
#ax.scatter(Vx, Vy, Vz, s=4)

# add horizontal connections
# for i in range(numControlPointsV):
#     ax.plot(Vx[i, 0:numCalcControlPointsU], Vy[i, 0:numCalcControlPointsU], Vz[i, 0:numCalcControlPointsU], 'r')
#
# # add vertical connections
# for i in range(numCalcControlPointsU):
#     ax.plot(Vx[0:numControlPointsV, i], Vy[0:numControlPointsV, i], Vz[0:numControlPointsV, i], 'r')

# plot the surface
#ax.plot_surface(X, Y, Z)

# plot the fat surface
ax.plot_surface(XFat, YFat, ZFat)

# display the plot with both surfaces on it
plt.show()

