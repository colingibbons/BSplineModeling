import numpy as np
import matplotlib
from os.path import isdir
from os import mkdir
import yaml
import cv2
import pyvista as pv

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import splineTools

# define parameters for reading from file (hardcoded for now, but should be easy to integrate into PATS)
fileName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/303-POST/outsidePoints/combined_slice_'
fatName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/303-POST/outsidePoints/fat_slice_'
rightFileName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/303-POST/outsidePoints/right_slice_'
leftFileName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/303-POST/outsidePoints/left_slice_'
vtkPath = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/303-POST/vtkModels/'

# fileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/outsidePoints/combined_slice_'
# fatName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/outsidePoints/fat_slice_'
# rightFileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/outsidePoints/right_slice_'
# leftFileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/outsidePoints/left_slice_'
# vtkPath = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/310-PRE/vtkModels/'

# fileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/303-POST/outsidePoints/combined_slice_'
# fatName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/303-POST/outsidePoints/fat_slice_'
# rightFileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/303-POST/outsidePoints/right_slice_'
# leftFileName = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/303-POST/outsidePoints/left_slice_'
# vtkPath = 'C:/Users/cogibbo/Desktop/3D-MRI-Data/303-POST/vtkModels/'

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

# call function to perform outside point spline fitting
X, Y, Z, Vx, Vy, Vz, U, V, tri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU,
                                                               numControlPointsV, degree, numPointsPerContour,
                                                               numSlices, upsample=True)

# update number of points per contour in case resampling was applied
numPointsPerContour = X.shape[1]

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

# if using the synthetic tube, set up number of points in each contour
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
# ax.set_title('Resampled Data')
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
# ax.set_title('3D Spline Surface')  # can add error metrics to title later if necessary
# ax.plot_surface(X, Y, Z)
#
# # now that all subplots have been generated, display them on a single figure
# plt.show()

# ########################################################################################################################
# # plot control points and the surface on the same plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_title('Outside Point spline surface with Control Mesh and Fat Points')
# ax.view_init(elevation, azimuth)
# ax.set_xlim(minX, maxX)
# ax.set_ylim(minY, maxY)
# ax.set_zlim(minZ, maxZ)
#
# # plot control points
# ax.scatter(Vx, Vy, Vz, s=4)
#
# # add horizontal connections
# for i in range(numControlPointsV):
#     ax.plot(Vx[i, 0:numCalcControlPointsU], Vy[i, 0:numCalcControlPointsU], Vz[i, 0:numCalcControlPointsU], 'r')
#
# # add vertical connections
# for i in range(numCalcControlPointsU):
#     ax.plot(Vx[0:numControlPointsV, i], Vy[0:numControlPointsV, i], Vz[0:numControlPointsV, i], 'r')
#
# # plot the surface
# ax.plot_surface(X, Y, Z)

# read fat points from file and add them to the scene
fatX, fatY, fatZ, numFatPointsPerSlice = splineTools.readSlicePoints(fatName, startFrame, stopFrame)
# ax.scatter(fatX, fatY, fatZ, marker='s', s=4, c='yellow')
#
# plt.show()

# generate normal vectors
crossX, crossY, crossZ = splineTools.generateNormalVectors(X, Y, Z)

# measure fat thickness at each normal vector
thicknessByPoint, xFatPoints, yFatPoints = splineTools.measureFatThickness(X, Y, crossX, crossY, fatX, fatY, numSlices,
                                                                           numPointsPerContour, numFatPointsPerSlice)


# # plot each normal vector one at a time along with the fat points associated with it
# zz = np.linspace(np.min(Z), np.max(Z), numSlices)
# zz = np.transpose(np.tile(zz, (numPointsPerContour, 1)))
# scaleFactor = X.shape[0] / numSlices
# for i in range(numSlices):
#    count = 0
#    index = np.floor(i*scaleFactor).astype(int)
#    for j in range(numPointsPerContour - 1):
#         x = xFatPoints[i, j, :]
#         y = yFatPoints[i, j, :]
#         z = fatZ[i]
#         points = ax.scatter(xFatPoints[i, j, :], yFatPoints[i, j, :], fatZ[i], s=4, c='black')
#         pointX = X[index, j] + 100*crossX[index, j]
#         pointY = Y[index, j] + 100*crossY[index, j]
#         pointZ = zz[i, j]
#         points2 = ax.scatter(pointX, pointY, pointZ, marker='s', s=4, c='purple')
#         vector = ax.quiver(X[index, j], Y[index, j], zz[i, j], crossX[index, j], crossY[index, j], crossZ[index, j],
#                            length=10, color='purple', arrow_length_ratio=0.1)
#         count += 1
#         message = "Slice {}, Point {}: Fat thickness of {}".format(i, j, thicknessByPoint[i, j])
#         plt.title(message)
#         plt.draw()
#         plt.pause(0.05)
#         points.remove()
#         points2.remove()
#         vector.remove()
#         print(count)


# generate an array that groups areas of nonzero fat thickness into separate fat deposits
deposits, numDeposits = splineTools.getFatDeposits(X, thicknessByPoint, numSlices)

# # create arrays for appropriately spaced plots of fat thickness
x = np.linspace(0, numSlices - 1, numSlices)
x = 10 * np.transpose(np.tile(x, (numPointsPerContour, 1)))
xStem = np.ravel(x)
y = np.linspace(0, numPointsPerContour, numPointsPerContour)
yStem = np.tile(y, numSlices)
y = np.tile(y, (numSlices, 1))
zStem = np.ravel(thicknessByPoint)
#
# # create "mountain" plot of fat thickness
# fig = plt.figure()
# ax = fig.add_subplot(2, 2, 2, projection='3d')
# ax.plot_surface(x, y, thicknessByPoint)
# ax.set_title('Mountain plot of fat thickness')
# ax.set_xlabel('Slice level')
# ax.set_ylabel('Azimuth')
# ax.set_zlabel('Fat thickness')

# # create stem plot on same axes
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for xx, yy, zz in zip(xStem, yStem, zStem):
#     plt.plot([xx, xx], [yy, yy], [0, zz], '-', color='yellow')
# ax.set_title('Stem plot of fat thickness')
# ax.set_xlabel('Slice level')
# ax.set_ylabel('Azimuth')
# ax.set_zlabel('Fat thickness')
# plt.show()

degree = 3
mX, mY, mZ, mTri = splineTools.mountainPlot(x, y, thicknessByPoint, degree, numSlices, numPointsPerContour,
                                            upsample=True)

scaleFactor = np.max(thicknessByPoint) / np.max(mZ)
mZ *= scaleFactor

# fig = plt.figure()
# fig.suptitle('Fat Thickness Heatmap')
# ax = fig.add_subplot(1, 3, 1)
# t1 = plt.imshow(mZ, cmap='inferno')
# plt.title('No threshold')
# fig.colorbar(t1, ax=ax, shrink=0.6)
# ax = fig.add_subplot(1, 3, 2)
# mZ[mZ < 0] = 0
# t2 = plt.imshow(mZ, cmap='inferno')
# plt.title('threshold = 1')
# fig.colorbar(t2, ax=ax, shrink=0.6)
# ax = fig.add_subplot(1, 3, 3)
# mZ[mZ < 2] = 0
# t3 = plt.imshow(mZ, cmap='inferno')
# fig.colorbar(t3, ax=ax, shrink=0.6)
# plt.title('threshold = 2')
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(np.ravel(mX), np.ravel(mY), np.ravel(mZ), triangles=mTri.triangles, antialiased=True)
# plt.title('Fat Mountain Plot. Degree: {}'.format(degree))
# plt.show()

# generate fat surface points
fatPolyData = splineTools.moreAltFatSurfacePoints(X, Y, Z, U, V, crossX, crossY, mZ, deposits, 1.5)
fatPolyData = fatPolyData.clean().triangulate().smooth(100)
slc = fatPolyData.slice((1, 1, 1))
p = pv.Plotter()
p.add_mesh(fatPolyData, color='yellow')
p.show()

########################################################################################################################
# perform spline routine for right side
# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(rightFileName, startFrame, stopFrame)

rightX, rightY, rightZ, rTri = splineTools.fitSplineOpen3D(origX, origY, origZ, numSlices, numPointsEachContour,
                                                           upsample=True)

# perform spline routine for left side
# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(leftFileName, startFrame, stopFrame)

# resample the data so each slice has the same number of points
# do this by fitting each slice with B-spline curve
resampleNumControlPoints = 7
degree = 3
resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
    splineTools.reSampleAndSmoothPoints(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)

leftX, leftY, leftZ, _, _, _, _, _, lTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU,
                                                                         numControlPointsV, degree, numPointsPerContour,
                                                                         numSlices, upsample=True)

# lVert = np.column_stack((np.ravel(leftX), np.ravel(leftY), np.ravel(leftZ)))
# LL = pv.PolyData(lVert)
#
# rVert = np.column_stack((np.ravel(rightX), np.ravel(rightY), np.ravel(rightZ)))
# RR = pv.PolyData(rVert)
#
# pl = pv.Plotter()
# pl.add_mesh(LL)
# pl.add_mesh(RR)
# pl.show()
########################################################################################################################
# check for vtk model folder, and create it if it does not already exist
if not isdir(vtkPath):
    mkdir(vtkPath)

# generate vtk model for right myo
rightPath = vtkPath + 'rightSide.vtk'
splineTools.createVTKModel(rightX, rightY, rightZ, rTri, rightPath)

# generate vtk model for left myo
leftPath = vtkPath + 'leftSide.vtk'
splineTools.createVTKModel(leftX, leftY, leftZ, lTri, leftPath)

fatPath = vtkPath + 'fat.vtk'
fatPolyData.save(fatPath, binary=False)
# populate YAML file data in preparation for writing
# numFat = len(fatDepositsX)
surfaces = [None]*3
surfaces[0] = {'name': 'Left Myocardium', 'filename': 'LeftMyocardium.vtk', 'opacity3D': 1.0, 'color':
               {'r': 1, 'g': 0, 'b': 0}}
surfaces[1] = {'name': 'Right Myocardium', 'filename': 'RightMyocardium.vtk', 'opacity3D': 1.0, 'color':
               {'r': 0, 'g': 0, 'b': 1}}
surfaces[2] = {'name': 'Fat', 'filename': 'Fat.vtk', 'opacity3D': 0.7, 'color':
               {'r': 1, 'g': 1, 'b': 0}}

# create and write the YAML file
data = {'surfaces': surfaces}
ymlPath = vtkPath + 'fatHeartModel.yml'
with open(ymlPath, 'w+') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)



