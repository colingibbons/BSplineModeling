import numpy as np
import matplotlib
from os.path import isdir
from os import mkdir
import yaml
import pyvista as pv
import time
from scipy.signal import savgol_filter
from PIL import Image

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import splineTools

fileName = 'C:\\Users\\cogibbo\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\outsidePoints\\combined_slice_'
fatName = 'C:\\Users\\cogibbo\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\outsidePoints\\fat_slice_'
rightFileName = 'C:\\Users\\cogibbo\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\outsidePoints\\right_slice_'
leftFileName = 'C:\\Users\\cogibbo\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\outsidePoints\\left_slice_'
vtkPath = 'C:\\Users\\cogibbo\\Desktop\\538 Project Files\\Cases\\MF0303-POST\\ED\\vtkModels\\'

startFrame = 1
stopFrame = 7
numSlices = (stopFrame - startFrame) + 1

########################################################################################################################

# set up parameters for 2D resampling
resampleNumControlPoints = 20
degree = 12

# set up parameters for 3D spline surface fit
numControlPointsU = numSlices + degree
numControlPointsV = numSlices
degreeU = 10
degreeV = 3
numCalcControlPointsU = numControlPointsU + degree

# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(rightFileName, startFrame, stopFrame, 10)

resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
    splineTools.reSampleAndSmoothPoints(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)

# perform spline routine for right side
rightX, rightY, rightZ, _, _, _, _, _, rTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ,
                                                                            numControlPointsU, numControlPointsV,
                                                                            degreeU, degreeV, numPointsPerContour,
                                                                            numSlices, fix_samples=True)

# read in points from files
origX, origY, origZ, numPointsEachContour = splineTools.readSlicePoints(leftFileName, startFrame, stopFrame, 10)

resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
    splineTools.reSampleAndSmoothPoints(origX, origY, origZ, numPointsEachContour, resampleNumControlPoints, degree)

# perform spline routine for left side
leftX, leftY, leftZ, _, _, _, _, _, lTri = splineTools.fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU,
                                                                         numControlPointsV, degreeU, degreeV,
                                                                         numPointsPerContour, numSlices,
                                                                         fix_samples=True)

# read in fat points from files
fatX, fatY, fatZ, numFatPointsPerSlice = splineTools.readSlicePoints(fatName, startFrame, stopFrame, 10)


# set up surface objects for each tisue type
numL = np.full((len(lTri.triangles), 1), 3)
Ltris = np.concatenate((numL, lTri.triangles), axis=1).astype(int)

numR = np.full((len(rTri.triangles), 1), 3)
Rtris = np.concatenate((numR, rTri.triangles), axis=1).astype(int)

lPoints = np.column_stack((np.ravel(leftX), np.ravel(leftY), np.ravel(leftZ)))
rPoints = np.column_stack((np.ravel(rightX), np.ravel(rightY), np.ravel(rightZ)))
fatPoints = np.column_stack((np.ravel(fatX), np.ravel(fatY), np.ravel(fatZ)))

# plot the resulting polydata
lPoly = pv.PolyData(lPoints, Ltris)
rPoly = pv.PolyData(rPoints, Rtris)
fatPoly = pv.PolyData(fatPoints)
p = pv.Plotter()
p.add_mesh(lPoly, color='red', culling='front')
p.add_mesh(rPoly, color='blue', culling='front')
p.add_mesh(fatPoly, color='yellow', culling='front', opacity=0)
p.show()

indices = np.linspace(0, leftX.shape[0]-1, numSlices, dtype=int)

for i in range(numSlices):
    ind = indices[i]
    fig = plt.figure()
    plt.plot(rightX[ind, :], rightY[ind, :], c='blue')
    plt.plot(leftX[ind, :], leftY[ind, :], c='red')
    plt.scatter(fatX[i], fatY[i], c='yellow')
    plt.title('Fat Points and Myocardium Contours')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


print('end')