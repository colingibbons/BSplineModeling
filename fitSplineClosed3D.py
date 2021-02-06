import numpy as np
import matplotlib
from os.path import isdir
from os import mkdir


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import splineTools

# define parameters for reading from file (hardcoded for now, but should be easy to integrate into PATS)
# fileName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/303-POST/outsidePoints/combined_slice_'
# fatName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/303-POST/outsidePoints/fat_slice_'

# fileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/outsidePoints/combined_slice_'
# fatName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/outsidePoints/fat_slice_'
# rightFileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/outsidePoints/right_slice_'
# leftFileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/outsidePoints/left_slice_'
# vtkPath = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/vtkModels/'

fileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/306-POST/outsidePoints/combined_slice_'
fatName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/306-POST/outsidePoints/fat_slice_'
rightFileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/306-POST/outsidePoints/right_slice_'
leftFileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/306-POST/outsidePoints/left_slice_'
vtkPath = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/306-POST/vtkModels/'

startFrame = 3
stopFrame = 8
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

# call function to perform outside point spline fitting
X, Y, Z, Vx, Vy, Vz, tri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU,
                                                         numControlPointsV, degree, numPointsPerContour, numSlices)

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

# ########################################################################################################################
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
crossX, crossY, crossZ = splineTools.generateNormalVectors(X, Y, Z, numPointsPerContour, numSlices)

# measure fat thickness at each normal vector
thicknessByPoint, xFatPoints, yFatPoints = splineTools.measureFatThickness(X, Y, crossX, crossY, fatX, fatY, numSlices,
                                                                           numPointsPerContour, numFatPointsPerSlice)

# get points that will be used to create fat spline surface
fatSurfaceX, fatSurfaceY, fatSurfaceZ = splineTools.getFatSurfacePoints(thicknessByPoint, xFatPoints, yFatPoints, X, Y,
                                                                        Z, numSlices, numPointsPerContour)

ax.scatter(fatSurfaceX, fatSurfaceY, fatSurfaceZ, marker='s', s=4, c='black')
plt.show()

# generate an array that groups areas of nonzero fat thickness into separate fat deposits
deposits, numDeposits = splineTools.getFatDeposits(thicknessByPoint, numSlices)

# create arrays for appropriately spaced plots of fat thickness
x = np.linspace(0, numSlices - 1, numSlices)
x = 10 * np.transpose(np.tile(x, (numPointsPerContour - 1, 1)))
xStem = np.ravel(x)
y = np.linspace(0, numPointsPerContour - 1, numPointsPerContour)
y = np.delete(y, -1, 0)
yStem = np.tile(y, numSlices)
y = np.tile(y, (numSlices, 1))
zStem = np.ravel(thicknessByPoint)

# # create "mountain" plot of fat thickness
# fig = plt.figure()
# ax = fig.add_subplot(2, 2, 2, projection='3d')
# ax.plot_surface(x, y, thicknessByPoint)
# ax.set_title('Mountain plot of fat thickness')
# ax.set_xlabel('Slice level')
# ax.set_ylabel('Azimuth')
# ax.set_zlabel('Fat thickness')
#
# # create stem plot on same axes
# ax = fig.add_subplot(2, 2, 4, projection='3d')
# for xx, yy, zz in zip(xStem, yStem, zStem):
#     plt.plot([xx, xx], [yy, yy], [0, zz], '-', color='yellow')
# ax.set_title('Stem plot of fat thickness')
# ax.set_xlabel('Slice level')
# ax.set_ylabel('Azimuth')
# ax.set_zlabel('Fat thickness')
#
# # plot the thickness array as a heatmap
# fig.add_subplot(2, 2, 1)
# plt.imshow(thicknessByPoint, cmap='hot', aspect='4.0')
# plt.title('Fat Thickness Map')
# plt.xlabel('Azimuth ({})'.format(numPointsPerContour))
# plt.ylabel('Elevation ({})'.format(numSlices))
# plt.colorbar()
#
# # plot the segmentation output
# fig.add_subplot(2, 2, 3)
# plt.imshow(deposits, cmap='inferno', aspect='4.0')
# plt.title('Fat Deposit Segmentation')
# plt.xlabel('Azimuth ({})'.format(numPointsPerContour))
# plt.ylabel('Elevation ({})'.format(numSlices))
# plt.show()

# create individual B-spline curves representing each of the fat deposits
fatDepositsX, fatDepositsY, fatDepositsZ, fatDepositTriangles = splineTools.generateFatDepositSplines(X, Y, Z,
                                                                                                      fatSurfaceX,
                                                                                                      fatSurfaceY,
                                                                                                      fatSurfaceZ,
                                                                                                      deposits,
                                                                                                      numDeposits)

# # plot each normal vector one at a time along with the fat points associated with it
# for i in range(numSlices):
#    count = 0
#    for j in range(numPointsPerContour - 1):
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
#         print(count)

# display plot with control point mesh, surface, and fat points
#plt.show()

########################################################################################################################

# # plot control points and the surface on the same plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_title('Heart surface with fat deposits')
# ax.view_init(elevation, azimuth)
# ax.set_xlim(minX, maxX)
# ax.set_ylim(minY, maxY)
# ax.set_zlim(minZ, maxZ)
#
# # plot the myocardium surface
# ax.plot_surface(X, Y, Z)
#
# # plot the fat surfaces
# for i in range(len(fatDepositsX)):
#     ax.plot_surface(fatDepositsX[i], fatDepositsY[i], fatDepositsZ[i], color='yellow')
#
# # show plot with fat surfaces
# plt.show()

########################################################################################################################
# perform spline routine for right side
# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(rightFileName, startFrame, stopFrame)

# resample the data so each slice has the same number of points
# do this by fitting each slice with B-spline curve
resampleNumControlPoints = 7
degree = 3
resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
    splineTools.reSampleAndSmoothPoints(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)

rightX, rightY, rightZ, rVx, rVy, rVz, rTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ,
                                                                            numControlPointsU, numControlPointsV,
                                                                            degree, numPointsPerContour, numSlices)

# perform spline routine for left side
# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(leftFileName, startFrame, stopFrame)

# resample the data so each slice has the same number of points
# do this by fitting each slice with B-spline curve
resampleNumControlPoints = 7
degree = 3
resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
    splineTools.reSampleAndSmoothPoints(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)

leftX, leftY, leftZ, lVx, lVy, lVz, lTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU,
                                                                         numControlPointsV, degree, numPointsPerContour,
                                                                         numSlices)
# plot both sides of the surface along with the fat splines
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('Heart spline model with fat deposits')
ax.view_init(elevation, azimuth)
ax.set_xlim(minX, maxX)
ax.set_ylim(minY, maxY)
ax.set_zlim(minZ, maxZ)

# plot right side
ax.plot_surface(rightX, rightY, rightZ, color='blue')

# plot left side
ax.plot_surface(leftX, leftY, leftZ, color='red')

# plot fat splines
for i in range(len(fatDepositsX)):
    ax.plot_surface(fatDepositsX[i], fatDepositsY[i], fatDepositsZ[i], color='yellow')

# show all splines in the same plot
plt.show()

########################################################################################################################
# check for vtk model folder, and create it if it does not already exist
if not isdir(vtkPath):
    mkdir(vtkPath)

# generate vtk model for right myo
rightPath = vtkPath + 'rightSide'
splineTools.createVTKModel(rightX, rightY, rightZ, rTri, rightPath)

# generate vtk model for left myo
leftPath = vtkPath + 'leftSide'
splineTools.createVTKModel(leftX, leftY, leftZ, lTri, leftPath)

numFatDeposits = len(fatDepositsX)

for i in range(numFatDeposits):
    depositPath = vtkPath + 'fatDeposit_' + str(i + 1)
    splineTools.createVTKModel(fatDepositsX[i], fatDepositsY[i], fatDepositsZ[i], fatDepositTriangles[i], depositPath)

