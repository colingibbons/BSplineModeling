import numpy as np
import matplotlib
from os.path import isdir
from os import mkdir
import yaml
import pyvista as pv
import time
from scipy.signal import savgol_filter

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import splineTools

# define parameters for reading from file (hardcoded for now, but should be easy to integrate into PATS)
# fileName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/306-POST/outsidePoints/combined_slice_'
# fatName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/306-POST/outsidePoints/fat_slice_'
# rightFileName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/306-POST/outsidePoints/right_slice_'
# leftFileName = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/306-POST/outsidePoints/left_slice_'
# vtkPath = 'C:/Users/colin/Desktop/school docs/Research/3D-MRI-Files/306-POST/vtkModels/'

# fileName = 'C:/Users/cogibbo/Desktop/538 Project Files/Cases/MF0310-POST/ES/outsidePoints/combined_slice_'
# fatName = 'C:/Users/cogibbo/Desktop/538 Project Files/Cases/MF0310-POST/ES/outsidePoints/fat_slice_'
# rightFileName = 'C:/Users/cogibbo/Desktop/538 Project Files/Cases/MF0310-POST/ES/outsidePoints/right_slice_'
# leftFileName = 'C:/Users/cogibbo/Desktop/538 Project Files/Cases/MF0310-POST/ES/outsidePoints/left_slice_'
# vtkPath = 'C:/Users/cogibbo/Desktop/538 Project Files/Cases/MF0310-POST/ES/vtkModels/'

# fileName = 'C:\\Users\\cogibbo\\Desktop\\3D-MRI-Data\\303-POST\\outsidePoints\\combined_slice_'
# fatName = 'C:\\Users\cogibbo\\Desktop\\3D-MRI-Data\\303-POST\\outsidePoints\\fat_slice_'
# rightFileName = 'C:\\Users\\cogibbo\Desktop\\3D-MRI-Data\\303-POST\\outsidePoints\\right_slice_'
# leftFileName = 'C:\\Users\\cogibbo\\Desktop\\3D-MRI-Data\\303-POST\\outsidePoints\\left_slice_'

fileName = 'C:\\Users\\cogibbo\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\outsidePoints\\combined_slice_'
fatName = 'C:\\Users\\cogibbo\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\outsidePoints\\fat_slice_'
rightFileName = 'C:\\Users\\cogibbo\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\outsidePoints\\right_slice_'
leftFileName = 'C:\\Users\\cogibbo\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\outsidePoints\\left_slice_'
vtkPath = 'C:\\Users\\cogibbo\\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\vtkModels\\'

startFrame = 1
stopFrame = 7
numSlices = (stopFrame - startFrame) + 1

#######################################################################################################################
# resampleNumControlPoints = 6
# degree = 3
# numControlPointsU = numSlices + degree
# numControlPointsV = numSlices
#
# origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(rightFileName, startFrame, stopFrame, 10)
#
# resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
#     splineTools.reSampleAndSmoothRightMyo(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)
#
#
# resampleNumControlPoints = 30
# degree = 3
# rx, ry, rz, _, _, _, _, _ = splineTools.reSampleAndSmoothRightMyo(origX, origY, origZ, numPointsEachContour,
#                                                                   resampleNumControlPoints, degree)
# degree = 3
#
# for s in range(len(rx)):
#     fig = plt.figure()
#     # plt.scatter(resampX[s, :], resampY[s, :], c='red')
#     plt.scatter(origX[s, :], origY[s, :], c='green')
#     plt.scatter(rx[s, :], ry[s, :], c='blue')
#     plt.show()
#
# rightX, rightY, rightZ, _, _, _, _, _, rTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ,
#                                                                             numControlPointsU, numControlPointsV,
#                                                                             degreeU, degreeV, numPointsPerContour,
#                                                                             numSlices, fix_samples=True)
#
# # perform spline routine for left side
# # read in points from files
# origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(leftFileName, startFrame, stopFrame, 10)
#
# # resample the data so each slice has the same number of points
# # do this by fitting each slice with B-spline curve
# resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
#     splineTools.reSampleAndSmoothPoints(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)
#
#
# leftX, leftY, leftZ, _, _, _, _, _, lTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU,
#                                                                          numControlPointsV, degreeU, degreeV,
#                                                                          numPointsPerContour, numSlices,
#                                                                          fix_samples=True)
#
# rxx = np.ravel(rightX)
# ryy = np.ravel(rightY)
# rzz = np.ravel(rightZ)
#
# rpts = np.column_stack((rxx, ryy, rzz))
#
# rtris = rTri.triangles
# threes = np.full((len(rtris), 1), 3)
# rtris = np.concatenate((threes, rtris), axis=1)
#
# lxx = np.ravel(leftX)
# lyy = np.ravel(leftY)
# lzz = np.ravel(leftZ)
#
# lpts = np.column_stack((lxx, lyy, lzz))
#
# ltris = lTri.triangles
# threes = np.full((len(ltris), 1), 3)
# ltris = np.concatenate((threes, ltris), axis=1)
#
# lPoly = pv.PolyData(lpts, ltris)
# rPoly = pv.PolyData(rpts, rtris)
# p = pv.Plotter()
# p.add_mesh(lPoly, color='blue')
# p.add_mesh(rPoly, color='red')
# p.show()


#######################################################################################################################

# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(fileName, startFrame, stopFrame, 10)

# resample the data so each slice has the same number of points
# do this by fitting each slice with B-spline curve
resampleNumControlPoints = 20
degree = 12
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
# plt.title(f'Uniformly sampled slice contours, degree={degree}')
# plt.show()

degree = 3
# set up parameters for spline fit
numControlPointsU = numSlices + degree
numControlPointsV = numSlices
degreeU = 10
degreeV = 3
numCalcControlPointsU = numControlPointsU + degree

# call function to perform outside point spline fitting
X, Y, Z, Vx, Vy, Vz, U, V, tri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU,
                                                               numControlPointsV, degreeU, degreeV, numPointsPerContour,
                                                               numSlices, fix_samples=True)

# num = np.full((len(tri.triangles), 1), 3)
# tris = np.concatenate((num, tri.triangles), axis=1).astype(int)
#
# threeDPoints = np.column_stack((np.ravel(X), np.ravel(Y), np.ravel(Z)))
#
# # plot the resulting polydata
# poly = pv.PolyData(threeDPoints, tris)
# p = pv.Plotter()
# p.add_mesh(poly, color='red')
# p.show()

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
# # ax.set_title('{0}x{1}'.format(lengthU, lengthV))  # can add error metrics to title later if necessary
# ax.plot_surface(X, Y, Z)
#
# # now that all subplots have been generated, display them on a single figure
# plt.show()

# ######################################################################################################################
# # plot control points and the surface on the same plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_title('Combined plot')
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
fatX, fatY, fatZ, numFatPointsPerSlice = splineTools.readSlicePoints(fatName, startFrame, stopFrame, 10)
# ax.scatter(fatX, fatY, fatZ, marker='s', s=4, c='yellow')
#
# plt.show()

# generate normal vectors
crossX, crossY, crossZ = splineTools.generateNormalVectors(X, Y, Z)

# # measure fat thickness at each normal vector
# thicknessByPoint, xFatPoints, yFatPoints, indices = splineTools.measureFatThickness(X, Y, crossX, crossY, fatX, fatY,
#                                                                                     numSlices, numPointsPerContour,
#                                                                                     numFatPointsPerSlice)

thicknessByPoint, fatPointsX, fatPointsY, indices = splineTools.altFatThickness(X, Y, crossX, crossY, fatX, fatY, numSlices,
                                                                                 numPointsPerContour, numFatPointsPerSlice)

for s in range(numSlices):
    ind = indices[s]
    fig = plt.figure()
    plt.plot(X[ind, :], Y[ind, :], c='blue')
    plt.scatter(fatX[s, :], fatY[s, :], c='yellow')
    plt.scatter(fatPointsX[s, :], fatPointsY[s, :], c='black')
    plt.show()

# plot each normal vector one at a time along with the fat points associated with it
# zz = np.linspace(np.min(Z), np.max(Z), numSlices)
# zz = np.transpose(np.tile(zz, (numPointsPerContour, 1)))
# scaleFactor = X.shape[0] / numSlices
# indices = np.floor(np.arange(0, X.shape[0], scaleFactor)).astype(int)
# for i in range(numSlices):
#     index = indices[i]
#     fig = plt.figure()
#     plt.plot(X[index, :], Y[index, :], c='blue')
#     plt.scatter(fatX[i, :], fatY[i, :], c='yellow')
#     count = 0
#     for j in range(numPointsPerContour - 1):
#         vector = plt.quiver(X[index, j], Y[index, j], crossX[index, j], crossY[index, j], scale=10,
#                             headwidth=0, color='black')
#         thisVectorPoints = plt.scatter(xFatPoints[i, j, :], yFatPoints[i, j, :], c='black')
#         message = f'Slice {i}, Point {j}: Fat thickness of {thicknessByPoint[i, j]} mm'
#         plt.title(message)
#         plt.draw()
#         plt.pause(0.05)
#         thisVectorPoints.remove()
#         vector.remove()

# create arrays for appropriately spaced plots of fat thickness
x = np.linspace(0, numSlices - 1, numSlices)
x = 10 * np.transpose(np.tile(x, (numPointsPerContour, 1)))
xStem = np.ravel(x)
y = np.linspace(0, numPointsPerContour, numPointsPerContour)
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

# might consider adjusting degree of this fit to see how it affects fidelity to input segmentations
# wrap mode uses values from other side of array, which is what we want to see given that these thickness measurements
# wrap around the surface of the heart
#thicknessByPoint2 = savgol_filter(thicknessByPoint, 77, 10, mode='wrap')

degreeU = 10
degreeV = 2
mX, mY, mZ, mTri = splineTools.mountainPlot(x, y, thicknessByPoint, degreeU, degreeV, numSlices, numPointsPerContour,
                                            fix_samples=True)

for s in range(numSlices):
    ind = indices[s]
    thisX = X[ind, :] + (mZ[ind, :] * crossX[ind, :])
    thisY = Y[ind, :] + (mZ[ind, :] * crossY[ind, :])

    fig = plt.figure()
    plt.plot(X[ind, :], Y[ind, :], c='blue')
    plt.scatter(fatX[s, :], fatY[s, :], c='yellow')
    plt.scatter(thisX, thisY, c='black')
    plt.show()


mxx = np.ravel(mX)
myy = np.ravel(mY)
mzz = np.ravel(mZ)
mTris = mTri.triangles

threes = np.full((len(mTris), 1), 3)
mTris = np.concatenate((threes, mTris), axis=1)

# pts = np.column_stack((mxx, myy, mzz))
# mountainPoly = pv.PolyData(pts, mTris)
# p = pv.Plotter()
# p.add_mesh(mountainPoly)
# p.show()

# fig = plt.figure()
# ax = fig.add_subplot(3, 1, 1)
# plt.imshow(mZ, cmap='inferno')
# plt.title('fat thickness mountain heat map')
# ax = fig.add_subplot(3, 1, 2)
# mZ[mZ < 0] = 0
# plt.imshow(mZ, cmap='inferno')
# plt.title('threshold = 1')
# ax = fig.add_subplot(3, 1, 3)
# mZ[mZ < 2] = 0
# plt.imshow(mZ, cmap='inferno')
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(np.ravel(mX), np.ravel(mY), np.ravel(mZ), triangles=mTri.triangles, antialiased=True)
# plt.title('Fat Mountain Plot. Degree: {}'.format(degree))
# plt.show()

# call the fat triangulation function to create a 3D fat surface from the mountain plot
start = time.perf_counter()
fatPolyData = splineTools.fatTriangulation(X, Y, Z, crossX, crossY, mZ, 1)
stop = time.perf_counter()
print(f'Fat surface generation took {stop-start} seconds')

########################################################################################################################
# perform spline routine for right side
# read in points from files

origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(rightFileName, startFrame, stopFrame, 10)

resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
    splineTools.reSampleAndSmoothRightMyo(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)

rightX, rightY, rightZ, _, _, _, _, _, rTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ,
                                                                            numControlPointsU, numControlPointsV,
                                                                            degreeU, degreeV, numPointsPerContour,
                                                                            numSlices, fix_samples=True)

# perform spline routine for left side
# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(leftFileName, startFrame, stopFrame, 10)

# resample the data so each slice has the same number of points
# do this by fitting each slice with B-spline curve
resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
    splineTools.reSampleAndSmoothPoints(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)

leftX, leftY, leftZ, _, _, _, _, _, lTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU,
                                                                         numControlPointsV, degreeU, degreeV, numPointsPerContour,
                                                                         numSlices, fix_samples=True)

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



