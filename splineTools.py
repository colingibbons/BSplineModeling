import numpy as np
from scipy.ndimage.measurements import label
from matplotlib.tri import triangulation as mtra
from matplotlib.tri import TriAnalyzer
import matplotlib.pyplot as plt
import pyvista as pv
from PIL import Image
import time
from scipy.spatial import Delaunay, delaunay_plot_2d, Voronoi, voronoi_plot_2d, KDTree

# this function performs a polar reordering of points
def reOrder(points):
    numPoints = len(points)

    # calculate the centroid for this point set
    centroidX = np.mean(points[:, 0])
    centroidY = np.mean(points[:, 1])
    centroid = np.asarray((centroidX, centroidY))

    # reposition points around centroid
    newPoints = points - np.tile(centroid, (numPoints, 1))

    # calculate the arctangent of each point and adjust to 0-to-2pi scale
    angles = np.arctan2(newPoints[:, 1], newPoints[:, 0])
    angles = angles + ((2*np.pi) * (angles < 0))

    # obtain indices corresponding with sorting angles by magnitude
    sortedAngles = np.argsort(angles)

    # re-order the points based on the above indices and populate the output array
    outPoints = np.zeros((len(points), 2))
    for i in range(len(points)):
        outPoints[i] = points[sortedAngles[i]]

    return outPoints

def reOrderOpen(points):
    # find the proper location of the "gap" in the open curve by searching for the successive points that are the
    # furthest away, and moving points such that the gap is between the first and last points
    largestDist = 0
    idx = 0
    for i in range(-1, len(points)-1):
        thisPoint = points[i]
        nextPoint = points[i+1]

        # calculate distance between successive points
        dist = np.sqrt((thisPoint[0] - nextPoint[0])**2 + (thisPoint[1] - nextPoint[1])**2)
        if dist > largestDist:
            # update largest distance variable and index of gap if new largest is found
            largestDist = dist
            idx = i+1

    # split the array at the index of largest gap between points
    segment_1 = points[0:idx, :]
    segment_2 = points[idx:, :]

    # recombine such that the largest distance gap is "between" the first and last points, rather than in
    # the middle of the array
    reordered_points = np.concatenate((segment_2, segment_1), axis=0)

    return reordered_points

def parameterizeClosedCurve(points, knots, degree):
    # get number of points and create parameter array
    numPoints = len(points)
    parameters = np.zeros(numPoints)

    # calculate the total distance
    totalDist = 0
    for i in range(numPoints - 1):
        distance = np.sqrt((points[i+1, 0] - points[i, 0])**2 + (points[i+1, 1] - points[i, 1])**2)
        totalDist += distance

    # get the first and last valid knots
    firstKnot = knots[degree]
    lastKnot = knots[len(knots) - degree - 1]

    # calculate length of parameterization space
    paramLength = lastKnot - firstKnot

    # fill parameter vector based on distance between interpolation points
    parameters[0] = firstKnot
    cumDist = 0
    for j in range(1, numPoints):
        distance = np.sqrt((points[j, 0] - points[j-1, 0])**2 + (points[j, 1] - points[j-1, 1])**2)
        cumDist += distance
        parameters[j] = (cumDist / totalDist)*paramLength + firstKnot

    return parameters

def parameterizeOpenCurve(points):

    numPoints = len(points)
    parameters = np.zeros(numPoints)

    totalDist = 0
    for i in range(numPoints - 1):
        xPart = (points[i+1, 0] - points[i, 0])**2
        yPart = (points[i+1, 1] - points[i, 1])**2
        totalDist += np.sqrt(xPart + yPart)

    parameters[0] = 0
    cumDist = 0
    for i in range(1, numPoints):
        xPart = (points[i, 0] - points[i-1, 0])**2
        yPart = (points[i, 1] - points[i-1, 1])**2
        cumDist += np.sqrt(xPart + yPart)
        parameters[i] = cumDist / totalDist

    return parameters

# Normalized B-spline by recursion
# TODO make a more efficient version of this because it's called so many times?
def NVal(tau, t, i, d, r):
    if i + d + 2 > len(tau) or d < 0 or r < 0:
        N = 0
        return N
    if t < tau[i+1] or t > tau[i+d+2]:
        N = 0
        return N

    if r > 0:
        if tau[i+d+1] > tau[i+1]:
            f1 = d / (tau[i+d+1] - tau[i+1])
        else:
            f1 = 0

        if tau[i+d+2] > tau[i+2]:
            f2 = d / (tau[i+d+2] - tau[i+2])
        else:
            f2 = 0
        N = f1*NVal(tau, t, i, d-1, r-1) - f2*NVal(tau, t, i+1, d-1, r-1)
        return N

    if d == 0:
        if t >= tau[i+1] and t < tau[i+2]:
            N = 1
        elif t >= tau[i+1] and t <= tau[i+2] and tau[i+1] < tau[i+2] and tau[i+2] == tau[len(tau) - 1]:
            N = 1
        else:
            N = 0
        return N

    if tau[i+d+1] > tau[i+1]:
        f1 = (t - tau[i+1]) / (tau[i+d+1] - tau[i+1])
    else:
        f1 = 0

    if tau[i+d+2] > tau[i+2]:
        f2 = (tau[i+d+2] - t) / (tau[i+d+2] - tau[i+2])
    else:
        f2 = 0

    N = f1*NVal(tau, t, i, d-1, r) + f2*NVal(tau, t, i+1, d-1, r)

    return N


# This function evaluates the B-Spline series at points in the vector t
def BSVal(b, tau, t, r):

    m = len(tau) - 1
    n = len(b[1]) - 1
    d = m - n - 1

    x = np.zeros((len(b), len(t)))
    for i in range(len(t)):
        for j in range(-1, n):
            x[:, i] += b[:, j+1] * NVal(tau, t[i], j, d, r)

    return x

# generates synthetic tube data for testing 3D spline algorithm
def generateTubeData(numPointsPerSlice, numSlices, r):

    theta = np.linspace(0, 2*np.pi, numPointsPerSlice + 1)
    theta = theta[0:numPointsPerSlice]
    xRow = r * np.cos(theta)
    yRow = r * np.sin(theta)
    zCol = (r/2) * np.linspace(0, numSlices-1, numSlices)
    Z = np.transpose(np.tile(zCol, (numPointsPerSlice, 1)))
    X = np.tile(xRow, (numSlices, 1))
    Y = np.tile(yRow, (numSlices, 1))

    return X, Y, Z

def parameterizeTube(X, Y, Z, tauU, tauV, degree):

    # get number of slices/points
    numSlices, numPointsPerContour = np.shape(X)

    # try doing V by taking Z distances
    # each column of V will be the same
    # scale a column of Z to parameter range
    firstKnotV = tauV[degree]
    lastKnotV = tauV[-degree-1]
    vRange = lastKnotV - firstKnotV
    zRange = Z[-1, 0] - Z[0, 0]
    zVect = Z[:, 0]
    scaleFactor = vRange / zRange
    offset = zVect[0] * scaleFactor
    vVect = (zVect * scaleFactor) - offset

    # now figure out what U will be. Use same for each row because
    # we've already sampled along contours uniformly.
    # each row will be uniform in valid parameter support
    firstKnotU = tauU[degree]
    lastKnotU = tauU[-degree-1]
    uVect = np.linspace(firstKnotU, lastKnotU, numPointsPerContour)

    # compute U and V with meshgrid since row/column dimensions are the same
    U, V = np.meshgrid(uVect, vVect)

    return U, V, firstKnotU, lastKnotU, firstKnotV, lastKnotV

def parameterizeFat(X, Y, Z, tauU, tauV, degree):

    numSlices, numPointsPerContour = np.shape(X)
    numSamples = len(tauU)

    vVect = np.linspace(0, 1, numSamples)
    uVect = np.linspace(0, 1, numPointsPerContour)

    U, V = np.meshgrid(uVect, vVect)

    return U, V


def EvaluateTensorProduct(Vx, Vy, Vz, tauU, tauV, degree, U, V, progressBar=None):

    numParamPointsV, numParamPointsU = np.shape(U)
    numControlPointsV, numControlPointsU = np.shape(Vx)
    numParam = numParamPointsU * numParamPointsV
    progressIncrement = (1 / numParam) * 100
    progressBarValue = 0

    X = np.zeros((numParamPointsV, numParamPointsU))
    Y = np.zeros((numParamPointsV, numParamPointsU))
    Z = np.zeros((numParamPointsV, numParamPointsU))

    for m in range(numParamPointsV):
        for n in range(numParamPointsU):

            sumx = 0
            sumy = 0
            sumz = 0

            for j in range(numControlPointsV):
                for i in range(numControlPointsU):
                    tProduct = NVal(tauU, U[m, n], i-1, degree, 0) * NVal(tauV, V[m, n], j-1, degree, 0)
                    sumx += Vx[j, i] * tProduct
                    sumy += Vy[j, i] * tProduct
                    sumz += Vz[j, i] * tProduct

            X[m, n] = sumx
            Y[m, n] = sumy
            Z[m, n] = sumz

            if progressBar:
                progressBarValue += progressIncrement
                progressBar.setValue(progressBarValue)
            else:
                print('param values:', m, n)

    return X, Y, Z

def readSlicePoints(baseName, startFrame, stopFrame, zSpacing):

    numSlices = (stopFrame - startFrame) + 1
    numPointsPerContour = np.zeros(numSlices)

    for i in range(numSlices):
        filePath = baseName + str(startFrame + i) + '.txt'
        # TODO fix issue where an empty file will crash the program. e.g. if there is no right side points on a slice
        with open(filePath) as f:
            for j, k in enumerate(f):
                pass
            numPointsPerContour[i] = j + 1
            f.close()

    # convert entries to integers or indexing won't work
    numPointsPerContour = numPointsPerContour.astype(int)
    # now allocate for X, Y, and Z matrices based on the number of points in the contours
    X = np.zeros((numSlices, max(numPointsPerContour)))
    Y = np.zeros((numSlices, max(numPointsPerContour)))
    Z = np.zeros((numSlices, max(numPointsPerContour)))

    # read data from files
    for i in range(numSlices):
        filePath = baseName + str(startFrame + i) + '.txt'
        sliceFile = open(filePath, 'r')
        X[i, 0:numPointsPerContour[i]] = np.loadtxt(sliceFile, dtype=int, usecols=0)
        # for some reason loadtxt won't work twice in a row unless the file is closed and reopened
        sliceFile.close()
        sliceFile = open(filePath, 'r')
        Y[i, 0:numPointsPerContour[i]] = np.loadtxt(sliceFile, dtype=int, usecols=1)
        sliceFile.close()
        # for now, just evenly space the slices along the z-axis. Can figure out the conversion from the 1cm
        # gap between slices in reality and the coordinate space in which the splines are visualized later
        Z[i, 0:numPointsPerContour[i]] = (startFrame + i) * zSpacing

    return X, Y, Z, numPointsPerContour

def fitSplineClosed2D(points, numControlPoints, degree):
    # perform angular reordering of points
    orderedPoints = reOrder(points)

    # copy another point to the end for a closed spline curve
    orderedPoints = np.append(orderedPoints, [orderedPoints[0]], 0)
    numDataPoints = len(orderedPoints)

    # set up parameters for spline fit
    n = numControlPoints - 1
    numCalcControlPoints = numControlPoints + degree

    # generate the knots (numKnots = n + 2d + 2)
    numKnots = n + 2 * degree + 2
    tau = np.zeros(numKnots)
    numOpenKnots = n + degree + 1
    tau[0:numOpenKnots] += np.linspace(0, 1, numOpenKnots)
    for i in range(0, degree + 1):
        diff = tau[i + 1] - tau[i]
        tau[numOpenKnots + i] = tau[numOpenKnots + i - 1] + diff

    # set up parameterization
    t = parameterizeClosedCurve(orderedPoints, tau, degree)

    p_mat = np.transpose(orderedPoints)
    A_mat = np.zeros((numDataPoints, numCalcControlPoints))

    for j in range(numDataPoints):
        for k in range(numCalcControlPoints):
            A_mat[j][k] = NVal(tau, t[j], k - 1, degree, 0)

    # create a constrained A matrix
    A_mat_con = A_mat
    A_mat_con[:, 0:degree] = A_mat_con[:, 0:degree] + A_mat_con[:, (numCalcControlPoints - degree):numCalcControlPoints]
    temp_A_mat_con = A_mat_con[:, 0:numControlPoints]
    A_mat_con = temp_A_mat_con

    # solve matrix equations for control points
    b_mat = np.linalg.lstsq(A_mat_con, orderedPoints, rcond=None)[0]
    b = np.transpose(b_mat)

    # duplicate last 'degree' control points
    new_b = np.zeros((2, numCalcControlPoints))
    new_b[:, 0:numControlPoints] = b
    new_b[:, numControlPoints:numControlPoints + degree] = b[:, 0:degree]
    b = new_b

    # calculate the spline
    interpCurve = BSVal(b, tau, t, 0)

    try:
        errorVector = interpCurve - orderedPoints
    except ValueError:
        errorVector = interpCurve - np.transpose(orderedPoints)

    errorInFit = np.sum(errorVector[0, :] ** 2 + errorVector[1, :] ** 2)

    return b, tau, errorInFit

# performs a 2D spline fit without reordering points, such that right myo shape is retained.
def fitSplineRightMyo2D(points, numControlPoints, degree):

    # copy another point to the end for a closed spline curve
    points = np.append(points, [points[0]], 0)
    numDataPoints = len(points)

    # set up parameters for spline fit
    n = numControlPoints - 1
    numCalcControlPoints = numControlPoints + degree

    # generate the knots (numKnots = n + 2d + 2)
    numKnots = n + 2 * degree + 2
    tau = np.zeros(numKnots)
    numOpenKnots = n + degree + 1
    tau[0:numOpenKnots] += np.linspace(0, 1, numOpenKnots)
    for i in range(0, degree + 1):
        diff = tau[i + 1] - tau[i]
        tau[numOpenKnots + i] = tau[numOpenKnots + i - 1] + diff

    # set up parameterization
    t = parameterizeClosedCurve(points, tau, degree)

    A_mat = np.zeros((numDataPoints, numCalcControlPoints))

    for j in range(numDataPoints):
        for k in range(numCalcControlPoints):
            A_mat[j][k] = NVal(tau, t[j], k - 1, degree, 0)

    # create a constrained A matrix
    A_mat_con = A_mat
    A_mat_con[:, 0:degree] = A_mat_con[:, 0:degree] + A_mat_con[:, (numCalcControlPoints - degree):numCalcControlPoints]
    temp_A_mat_con = A_mat_con[:, 0:numControlPoints]
    A_mat_con = temp_A_mat_con

    # solve matrix equations for control points
    b_mat = np.linalg.lstsq(A_mat_con, points, rcond=None)[0]
    b = np.transpose(b_mat)

    # duplicate last 'degree' control points
    new_b = np.zeros((2, numCalcControlPoints))
    new_b[:, 0:numControlPoints] = b
    new_b[:, numControlPoints:numControlPoints + degree] = b[:, 0:degree]
    b = new_b

    # calculate the spline
    interpCurve = BSVal(b, tau, t, 0)

    try:
        errorVector = interpCurve - points
    except ValueError:
        errorVector = interpCurve - np.transpose(points)

    errorInFit = np.sum(errorVector[0, :] ** 2 + errorVector[1, :] ** 2)

    return b, tau, errorInFit

# resamples points for a closed spline fit such that an equal number of points are present in each slice
def reSampleAndSmoothPoints(X, Y, Z, numPointsEachContour, numControlPoints, degree):

    # data is returned in 3 matrices, each with 'numSlices' rows
    # number of columns is what the data was resampled to

    # allocate for control points and new data
    numPointsPerContour = max(numPointsEachContour)
    numSlices = np.size(X, 0)
    newX = np.zeros((numSlices, numPointsPerContour))
    newY = np.zeros((numSlices, numPointsPerContour))
    newZ = np.zeros((numSlices, numPointsPerContour))

    # number of control points for a closed curve is increased by degree of curve
    numCalcControlPoints = numControlPoints + degree
    newXControl = np.zeros((numSlices, numCalcControlPoints))
    newYControl = np.zeros((numSlices, numCalcControlPoints))
    newZControl = np.zeros((numSlices, numCalcControlPoints))

    # now loop through the slices and fit each with a cubic B-spline curve
    totalError = 0
    for i in range(numSlices):
        # set up this array of points
        points = np.zeros((numPointsEachContour[i], 2))
        # points[:, 0] = X[i, 0:numPointsEachContour[i]]
        # points[:, 1] = Y[i, 0:numPointsEachContour[i]]

        points[:, 0] = X[i, X[i] != 0]
        points[:, 1] = Y[i, Y[i] != 0]

        # call the fitting function
        b, tau, errorInFit = fitSplineClosed2D(points, numControlPoints, degree)
        totalError += errorInFit

        # determine the support of the spline in the parameterization
        firstKnot = tau[degree]
        lastKnot = tau[-degree-1]

        # use uniform parameterization for the resampling
        t = np.linspace(firstKnot, lastKnot, numPointsPerContour)

        # calculate the new points
        interpCurve = BSVal(b, tau, t, 0)

        # put points in out arrays
        newX[i, 0:numPointsPerContour] = interpCurve[0, :]
        newY[i, 0:numPointsPerContour] = interpCurve[1, :]
        newZ[i, 0:numPointsPerContour] = Z[i, 0]

        newXControl[i, :] = b[0, :]
        newYControl[i, :] = b[1, :]
        newZControl[i, :] = Z[i, 0]

    return newX, newY, newZ, newXControl, newYControl, newZControl, numPointsPerContour, totalError

# this function performs a resampling and 2D spline fit for each right myo slice. it differs from the routine used
# for other surfaces in that it does not perform the angle-centroid-based point reordering, such that the distinct
# shape of the right myocardium is retained
def reSampleAndSmoothRightMyo(X, Y, Z, numPointsEachContour, numControlPoints, degree):

    # data is returned in 3 matrices, each with 'numSlices' rows
    # number of columns is what the data was resampled to

    # allocate for control points and new data
    numPointsPerContour = max(numPointsEachContour)
    numSlices = np.size(X, 0)
    newX = np.zeros((numSlices, numPointsPerContour))
    newY = np.zeros((numSlices, numPointsPerContour))
    newZ = np.zeros((numSlices, numPointsPerContour))

    # number of control points for a closed curve is increased by degree of curve
    numCalcControlPoints = numControlPoints + degree
    newXControl = np.zeros((numSlices, numCalcControlPoints))
    newYControl = np.zeros((numSlices, numCalcControlPoints))
    newZControl = np.zeros((numSlices, numCalcControlPoints))

    # now loop through the slices and fit each with a cubic B-spline curve
    totalError = 0
    for i in range(numSlices):
        # set up this array of points
        points = np.zeros((numPointsEachContour[i], 2))
        # points[:, 0] = X[i, 0:numPointsEachContour[i]]
        # points[:, 1] = Y[i, 0:numPointsEachContour[i]]

        points[:, 0] = X[i, X[i] != 0]
        points[:, 1] = Y[i, Y[i] != 0]

        # call the fitting function
        b, tau, errorInFit = fitSplineRightMyo2D(points, numControlPoints, degree)
        totalError += errorInFit

        # determine the support of the spline in the parameterization
        firstKnot = tau[degree]
        lastKnot = tau[-degree-1]

        # use uniform parameterization for the resampling
        t = np.linspace(firstKnot, lastKnot, numPointsPerContour)

        # calculate the new points
        interpCurve = BSVal(b, tau, t, 0)

        # put points in out arrays
        newX[i, 0:numPointsPerContour] = interpCurve[0, :]
        newY[i, 0:numPointsPerContour] = interpCurve[1, :]
        newZ[i, 0:numPointsPerContour] = Z[i, 0]

        newXControl[i, :] = b[0, :]
        newYControl[i, :] = b[1, :]
        newZControl[i, :] = Z[i, 0]

    return newX, newY, newZ, newXControl, newYControl, newZControl, numPointsPerContour, totalError

# this function generates "normal vectors" that hard-codes the Z component of each normal vector as 0, such that
# the fat thickness in each (2D) slice can be assessed
def generateNormalVectors(X, Y, Z):

    numSlices, numPointsPerContour = X.shape

    crossX = np.zeros((numSlices, numPointsPerContour))
    crossY = np.zeros((numSlices, numPointsPerContour))
    crossZ = np.zeros((numSlices, numPointsPerContour))

    # loop through all points and compute normals. First and last points on each contour are the same, so the (redundant)
    # normal is not computed for the last point
    for i in range(-1, numSlices - 1):
        for j in range(numPointsPerContour - 1):

            # starting point
            vector1 = (X[i, j], Y[i, j], Z[i, j])
            # u differential
            vector2 = (X[i+1, j], Y[i+1, j], Z[i+1, j])
            uDer = np.subtract(vector2, vector1)
            # v differential
            vector3 = (X[i, j+1], Y[i, j+1], Z[i, j+1])
            vDer = np.subtract(vector3, vector1)

            # compute the cross product
            crossVector = np.cross(vDer, uDer)
            crossVector = crossVector / np.linalg.norm(crossVector)
            # assign values to output arrays. Z component of each vector is hardcoded at 0 so that vector points
            # straight outward towards the fat at each slice
            if np.isnan(crossVector[0] or np.isnan(crossVector[1])):
                print('stop here')
            crossX[i, j] = crossVector[0]
            crossY[i, j] = crossVector[1]
            crossZ[i, j] = 0

        # check to see if the computed normal vectors have the correct orientation (i.e. point outward from the surface)
        # if they do not, reverse the direction by negating the vectors. Z-component is ignored because it's always 0
        if crossX[i, int(numPointsPerContour / 2)] > 0:
            crossX[i, :] *= -1
            crossY[i, :] *= -1

    # set the normal vectors for the last column to equal the vectors for the first - this is because the last point
    # is the same as the first for closed contours
    crossX[:, -1] = crossX[:, 0]
    crossY[:, -1] = crossY[:, 0]

    return crossX, crossY, crossZ

# measures the fat thickness at each unit normal vector of the surface.
def measureFatThickness(X, Y, crossX, crossY, fatX, fatY, numSlices, numPointsPerContour, fatPointsPerSlice):
    # TODO some mechanism to account for the "amount" of each point that the vector passes through. Since each fat point
    # represents a voxel, the vector passing through the corner of a voxel vs directly through will contribute unevenly
    # to the overall thickness. Maybe do first check and then use magnitude of distance to scale the contribution to the
    # overall thickness by that pixel
    thicknessByPoint = np.zeros((numSlices, numPointsPerContour))
    xFatPoints = np.zeros((numSlices, numPointsPerContour, max(fatPointsPerSlice)))
    yFatPoints = np.zeros((numSlices, numPointsPerContour, max(fatPointsPerSlice)))
    # fill point array with -1. Because of the way tissue data is extracted from PATS, all actual values will be
    # positive. Using -1 prevents filler points and actual data from being mistaken for one another
    xFatPoints.fill(-1)
    yFatPoints.fill(-1)

    # generate X and Y indices at which fat measurements should take place. This looks a little different for this
    # function because the slice loop begins at -1 to accommodate
    scaleFactor = X.shape[0] / numSlices
    indices = np.floor(np.arange(0, X.shape[0], scaleFactor)).astype(int)
    for i in range(-1, numSlices - 1):
        index = indices[i]
        for j in range(numPointsPerContour):
            # generate a point arbitrarily far along the normal vector
            xDir = X[index, j] + (200 * crossX[index, j])
            yDir = Y[index, j] + (200 * crossY[index, j])
            thickness = 0
            # check each fat point for proximity to the line defined by normal vector
            for k in range(fatPointsPerSlice[i]):

                ab = np.sqrt((xDir - X[index, j]) ** 2 + (yDir - Y[index, j]) ** 2)
                ac = np.sqrt((xDir - fatX[i, k]) ** 2 + (yDir - fatY[i, k]) ** 2)
                bc = np.sqrt((X[index, j] - fatX[i, k]) ** 2 + (Y[index, j] - fatY[i, k]) ** 2)

                is_on_segment = abs(ac + bc - ab) < 0.02

                # update thickness measure if a point is found to be sufficiently close
                if is_on_segment:
                    thickness += 1
                    xFatPoints[i, j, k] = fatX[i, k]
                    yFatPoints[i, j, k] = fatY[i, k]

            thicknessByPoint[i, j] = thickness

    return thicknessByPoint, xFatPoints, yFatPoints

def generateFatSurfacePoints(X, Y, Z, U, V, crossX, crossY, fatThicknessZ, threshold):

    numU, numV = X.shape

    uParam = U[fatThicknessZ > threshold]

    fLength = len(uParam)
    fatPointsX = np.zeros(2 * fLength)
    fatPointsY = np.zeros(2 * fLength)
    fatPointsZ = np.zeros(2 * fLength)

    # Since the fat thickness can be likened to the diagonal of a square, the x and y components must each be divided
    # by a factor of sqrt(2) for the fat point location to accurately reflect the magnitude of the thickness
    scaleFactor = 1 / np.sqrt(2)

    index = 0
    for i in range(numU):
        for j in range(numV):
            if fatThicknessZ[i, j] > threshold:
                # calculate point location along normal vector based on fat thickness at that point
                thisX = X[i, j] + (scaleFactor * fatThicknessZ[i, j] * crossX[i, j])
                thisY = Y[i, j] + (scaleFactor * fatThicknessZ[i, j] * crossY[i, j])
                thisZ = Z[i, j]

                fatPointsX[index] = thisX
                fatPointsY[index] = thisY
                fatPointsZ[index] = thisZ

                index += 1

    # add surface points to close contour
    for i in range(numU):
        for j in range(numV):
            if fatThicknessZ[i, j] > threshold:
                fatPointsX[index] = X[i, j]
                fatPointsY[index] = Y[i, j]
                fatPointsZ[index] = Z[i, j]

                index += 1

    # create polydata from points and perform a triangulation on them
    data = np.column_stack((fatPointsX, fatPointsY, fatPointsZ))
    polydata = pv.PolyData(data)
    pp = polydata.delaunay_3d(alpha=3.0, tol=0.005)
    # delaunay routine returns an unstructured grid, so convert back to polydata
    surf = pp.extract_surface()

    return surf

def fatTriangulation(X, Y, Z, crossX, crossY, fatThicknessZ, threshold):

    # define the scale factor that will determine where fat-points are located in 3D space
    scaleFactor = 1 / np.sqrt(2)

    # get the shape of the input parameterization
    numU, numV = fatThicknessZ.shape

    # apply threshold
    fatThicknessZ[fatThicknessZ < threshold] = 0

    # lists will hold fat data by slice and by region/deposit
    fatSlices = []
    fatRegions = []

    # allocate image to hold resulting contour data for each slice
    numRegionsPerSlice = np.zeros(numU)

    # loop through each slice of the parameterized surface
    for i in range(numU):

        # determine which fat points belong to separate "deposits" by finding regions separated by
        thisSlice = fatThicknessZ[i]
        thisRegion = []

        beginLength = None

        if np.count_nonzero(thisSlice == 0) == 0:
            # indices have to be handled differently if entire slice is nonzero elements
            indices = [np.arange(0, len(thisSlice), 1)]
            numRegionsPerSlice[i] = 2

            # for a fully defined slice, we expect the fat intersection to consist of two (roughly) concentric contours
            # as such, define an "inside" and "outside" array
            fatPointsInside = np.zeros((len(thisSlice), 3))
            fatPointsOutside = np.zeros_like(fatPointsInside)

            # inside fat contour will be same as myocardium surface
            fatPointsInside[:, 0] = X[i, :]
            fatPointsInside[:, 1] = Y[i, :]
            fatPointsInside[:, 2] = Z[i, :]

            # append inside contour to running tally of fat arrays
            thisRegion.append(fatPointsInside)
            fatRegions.append(fatPointsInside)

            # outside fat contour is comprised of fat points whose distance from the myocardium is defined by
            # the parameterized fat thickness map
            fatPointsOutside[:, 0] = X[i, :] + (scaleFactor * thisSlice * crossX[i, :])
            fatPointsOutside[:, 1] = Y[i, :] + (scaleFactor * thisSlice * crossY[i, :])
            fatPointsOutside[:, 2] = Z[i, :]

            # append outside to fat arrays
            thisRegion.append(fatPointsOutside)
            fatRegions.append(fatPointsOutside)
        elif np.all(thisSlice == 0):
            # skip this slice if the fat thickness is 0 across the entire slice
            numRegionsPerSlice[i] = 0
        else:
            # label regions based on connectivity such that separate fat deposits may be handled separately
            labelled, _ = label(thisSlice)
            # if deposits are identified at start and end of label array, combine them into a single deposit. Since the
            # parameterization "wraps around" the surface of the heart, regions on either end of the array are actually
            # connected when viewed in 3D space
            if labelled[0] != 0 and labelled[-1] != 0:
                beginLength = np.count_nonzero(labelled == labelled[0])
                labelled[labelled == labelled[-1]] = labelled[0]

            # get indices of fat points belonging to each deposit
            indices = [np.asarray(np.nonzero(labelled == k)) for k in np.unique(labelled)[1:]]

            # deposits that "wrap around" the array need to be handled differently to ensure proper connectivity
            # between points on either side of the array
            if beginLength:
                for j, dep in enumerate(indices):
                    # if a deposit has points at indices 0 and 99, it wraps around the array
                    wrap = (0 in dep) and (99 in dep)
                    if wrap:
                        indices[j] = np.roll(dep, -beginLength)

            # make note of number of regions in each slice
            numRegionsPerSlice[i] = len(indices)

            for region in indices:
                lenRegion = max(region.shape)
                fatPointsThisRegion = np.zeros((2 * lenRegion + 1, 2))

                # generate a fat surface point by traveling along the direction of the normal vector in accordance
                # with the magnitude of fat thickness at the current parameter location
                thisX = X[i, region] + (scaleFactor * thisSlice[region] * crossX[i, region])
                thisY = Y[i, region] + (scaleFactor * thisSlice[region] * crossY[i, region])

                # add fat points to list
                fatPointsThisRegion[0:lenRegion] = np.column_stack((np.squeeze(thisX), np.squeeze(thisY)))

                # flip region index array such that points are added to overall array in "contour order"
                region = np.fliplr(region)

                # add surface points corresponding with fat points to array, such that a closed contour can be generated
                # from the surface points
                fatPointsThisRegion[lenRegion:-1] = np.column_stack((np.squeeze(X[i, region]), np.squeeze(Y[i, region])))

                # duplicate first point for closed contour
                fatPointsThisRegion[-1, :] = fatPointsThisRegion[0, :]

                # add z-axis coordinates to fat point array
                zz = np.full((2 * lenRegion + 1, 1), Z[i, 0])
                fatPointsThisRegion = np.concatenate((fatPointsThisRegion, zz), axis=1)

                # append the points from this region to the overall list
                thisRegion.append(fatPointsThisRegion)

                fatRegions.append(fatPointsThisRegion)

        # collapse fat regions into single point array and append
        if len(thisRegion) > 0:
            thisRegion = np.concatenate(thisRegion)
            fatSlices.append(thisRegion)

    # concatenate points and lines from all slices into a single numpy array
    threeDPoints = np.concatenate(fatRegions)

    # loop through each fat slice
    lines = []
    linesAround = []
    depositIndex = 0
    pointIndex = 0
    nextSliceIndex = 0
    for j in range(len(fatSlices)):

        # skip this slice index if there are no fat points for the slice
        if int(numRegionsPerSlice[j]) == 0:
            continue
        # get number of points in this slice
        thisSliceLen = len(fatSlices[j])

        # update KDTree to reflect points of the next slice - unless the current slice is the uppermost slice
        if j != len(fatSlices) - 1:
            thisTree = KDTree(fatSlices[j+1][:, 0:2])
            # update next slice index so that lines connect properly to the next slice up
            nextSliceIndex += thisSliceLen

        # perform the KDTree query for each individual deposit
        for k in range(int(numRegionsPerSlice[j])):

            # grab this specific fat deposit from the region list
            thisFatDeposit = fatRegions[depositIndex + k]

            # define contour lines array
            thisLinesAround = np.zeros((len(thisFatDeposit), 2))
            thisLinesAround[:, 0] = np.arange(pointIndex, pointIndex+len(thisFatDeposit), 1)
            thisLinesAround[:, 1] = np.roll(thisLinesAround[:, 0], -1)
            linesAround.append(thisLinesAround)

            # generate lines connecting to next slice up, unless this is region belongs to the topmost slice
            if j != len(fatSlices) - 1:
                # define lines array
                thisLines = np.zeros((len(thisFatDeposit), 2))
                thisLines[:, 0] = np.arange(pointIndex, pointIndex + len(thisFatDeposit), 1)

                # perform KDTree query for this fat deposit
                dists, ids = thisTree.query(thisFatDeposit[:, 0:2])
                mask = dists < 5

                # add point index to line ids so that they connect to points on the correct slice
                ids += nextSliceIndex

                # finish defining line connectivity and append lines from this deposit to the overall list
                thisLines[:, 1] = ids
                # mask off connections that generate excessively long lines - these connections contribute to the
                # formation of undesirable triangles in the output
                thisLines = thisLines[mask, :]
                # append this region's lines to the overall list of lines
                lines.append(thisLines)

            # update point index
            pointIndex += len(thisFatDeposit)

        # update indices before moving on to next slice
        depositIndex += int(numRegionsPerSlice[j])

    # p = pv.Plotter()
    # p.add_mesh(threeDPoints, point_size=10)
    # for i in range(len(linesAround)):
    #     ll = np.ravel(linesAround[i]).astype(int)
    #     p.add_lines(threeDPoints[ll], color='red')
    #
    # for i in range(len(lines)):
    #     ll = np.ravel(lines[i]).astype(int)
    #     p.add_lines(threeDPoints[ll], color='blue')
    # p.show()

    # restructure array such that diagonal lines are explicitly defined. This is accomplished by taking each consecutive
    # pair of point indices and storing them as a row in a new array
    # TODO do this without more_itertools
    from more_itertools import pairwise
    newLines = []
    for l in lines:

        if l.shape[0] != 0:
            l = np.ravel(l)
            new = np.asarray(list(pairwise(l)))

            # remove redundant lines from output
            new = np.sort(new, axis=1)
            _, indices = np.unique(new, return_index=True, axis=0)
            new = new[np.sort(indices)]
            newLines.append(new)
        else:
            continue

    # concatenate the line lists into a single numpy array containing every line
    flatLines = np.concatenate(newLines)
    flatLinesAround = np.concatenate(linesAround)
    allLines = np.concatenate((flatLines, flatLinesAround), axis=0)

    # loop through the list of vertices and establish a connectivity list for each
    connectedVertices = []
    for k in range(len(threeDPoints)):
        lX, lY = np.where(allLines == k)
        lY = lY ^ 1

        # get indices of vertices to which the current vertex is connected by a known line
        cV = allLines[lX, lY].astype(int)
        connectedVertices.append(cV)

    # generate the triangles that will comprise the output polydata by looping through each vertex and checking the
    # points to which it is connected. If a vertex is connected to two other vertices and those two vertices are also
    # connected to one another, the three vertices are saved as a triangle.
    tris = []
    for m in range(len(threeDPoints)):
        thisVertex = connectedVertices[m]

        for neighbor in thisVertex:
            conn = connectedVertices[neighbor]
            common = np.intersect1d(thisVertex, conn)
            if len(common) > 0:
                for match in common:
                    newTri = [m, neighbor, match]
                    tris.append(newTri)

    # add a column of 3's to the triangle array because pyvista requires that the number of edges in a face be specified
    tris = np.asarray(tris)
    num = np.full((len(tris), 1), 3)
    tris = np.concatenate((num, tris), axis=1).astype(int)

    # # plot the resulting polydata
    poly = pv.PolyData(threeDPoints, tris)
    # p = pv.Plotter()
    # p.add_mesh(poly, color='yellow')
    # p.show(interactive_update=True)

    return poly

# This is the primary spline fitting routine, used to generate the spline surface which is open on the top and bottom,
# but closed "around" the heart
def fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU, numControlPointsV, degree, numPointsPerContour,
                      numSlices, fix_samples=False, progressBar=None):
    numCalcControlPointsU = numControlPointsU + degree
    m = numControlPointsU - 1
    n = numControlPointsV - 1
    M = numPointsPerContour - 1
    N = numSlices - 1

    # Figure out what set of two knots will be for u parameter direction (around each contour)
    numKnotsU = m + 2 * degree + 2
    tauU = np.zeros(numKnotsU)
    numOpenKnotsU = m + degree + 1
    tauU[0:numOpenKnotsU] = np.linspace(0, 1, numOpenKnotsU)
    for i in range(degree + 1):
        diff = tauU[i + 1] - tauU[i]
        tauU[numOpenKnotsU + i] = tauU[numOpenKnotsU + i - 1] + diff

    numKnotsV = n + degree + 2
    numInteriorKnotsV = numKnotsV - 2 * degree
    tauV = np.zeros(numKnotsV)
    tauV[degree:numInteriorKnotsV + degree] = np.linspace(0, 1, numInteriorKnotsV)
    tauV[numInteriorKnotsV + degree:numKnotsV] = np.ones(degree)

    # set up parameterization
    U, V, firstKnotU, lastKnotU, firstKnotV, lastKnotV = parameterizeTube(resampX, resampY, resampZ,
                                                                                      tauU, tauV, degree)

    # now we need to set up matrices to solve for mesh of control points
    # (B*V*T^T = P)

    # set up B matrix
    B = np.zeros((M + 1, numCalcControlPointsU))
    for r in range(M + 1):
        for i in range(numCalcControlPointsU):
            uVal = U[0, r]
            B[r, i] = NVal(tauU, uVal, i - 1, degree, 0)

    # set up C matrix
    C = np.zeros((N + 1, n + 1))
    for s in range(N + 1):
        for j in range(n + 1):
            vVal = V[s, 0]
            C[s, j] = NVal(tauV, vVal, j - 1, degree, 0)

    # now set up Px, Py, and Pz matrices
    Px = np.transpose(resampX)
    Py = np.transpose(resampY)
    Pz = np.transpose(resampZ)

    # constrain the B matrix so last three control points of each slice
    # equal the first three (for cubic)
    B_con = B
    B_con[:, 0:degree] = B_con[:, 0:degree] + B_con[:, numCalcControlPointsU - degree:numCalcControlPointsU]
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

    # generate a larger parameterization array before evaluating the tensor product
    if fix_samples:
        uMin = np.min(U)
        uMax = np.max(U)
        uVect = np.linspace(uMin, uMax, 100)
        vVect = np.linspace(0, 1, 100)
        U, V = np.meshgrid(uVect, vVect)

    # generate triangles for use in VTK models
    tri = mtra.Triangulation(np.ravel(U), np.ravel(V))

    # evaluate tensor product to get surface points. Operation is timed because it tends to be the slowest step
    startTime = time.perf_counter()
    X, Y, Z = EvaluateTensorProduct(Vx, Vy, Vz, tauU, tauV, degree, U, V, progressBar=progressBar)
    stopTime = time.perf_counter()
    print("Tensor product evaluation took {} seconds".format(stopTime - startTime))

    return X, Y, Z, Vx, Vy, Vz, U, V, tri

# creates a VTK file containing the data for a given element of the B-spline curve
def createVTKModel(X, Y, Z, triangles, filePath):

    # ravel the arrays in preparation for writing VTK file
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    verts = np.column_stack((X, Y, Z))

    # get number of points - points will act as vertices for the triangular cells
    numVerts = triangles.triangles.shape[0]
    numSides = np.full((1, numVerts), 3)
    faces = np.concatenate((numSides.T, triangles.triangles), axis=1)

    # generate the polydata object for this data and save it to a VTK file
    surf = pv.PolyData(verts, faces)
    surf.save(filePath, binary=False)

def mountainPlot(x, y, thickness, degree, numSlices, numPointsPerContour, fix_samples=False, progressBar=None):

    # set up parameters for spline fit
    numControlPointsU = 4
    numControlPointsV = 4
    m = numControlPointsU - 1
    n = numControlPointsV - 1
    numCalcControlPointsU = numControlPointsU + degree
    numCalcControlPointsV = numControlPointsV + degree

    # generate the knots (numKnots = n + 2d + 2)
    numKnotsU = m + (2*degree) + 2
    tauU = np.zeros(numKnotsU)
    numOpenKnotsU = numKnotsU - (2*degree)
    tauU[degree:-degree] = np.linspace(0, 1, numOpenKnotsU)
    tauU[-degree:] = 1

    numKnotsV = n + 2 * degree + 2
    tauV = np.zeros(numKnotsV)
    numOpenKnotsV = numKnotsV - (2*degree)
    tauV[degree:-degree] = np.linspace(0, 1, numOpenKnotsV)
    tauV[-degree:] = 1

    # set up parameterization
    uVect = np.linspace(0, 1, numPointsPerContour)
    vVect = np.linspace(0, 1, numSlices)
    U, V = np.meshgrid(uVect, vVect)

    # U, V = parameterizeFat(resampX, resampY, resampZ, tauU, tauV, degree)

    # now we need to set up matrices to solve for mesh of control points
    # (B*V*T^T = P)

    B = np.zeros((numPointsPerContour, numCalcControlPointsU))
    for r in range(numPointsPerContour):
        for i in range(numCalcControlPointsU):
            uVal = U[0, r]
            B[r, i] = NVal(tauU, uVal, i - 1, degree, 0)

    # set up C matrix
    C = np.zeros((numSlices, numCalcControlPointsV))
    for s in range(numSlices):
        for j in range(numCalcControlPointsV):
            vVal = V[s, 0]
            C[s, j] = NVal(tauV, vVal, j-1, degree, 0)

    # now set up Px, Py, and Pz matrices
    Px = np.transpose(x)
    Py = np.transpose(y)
    Pz = np.transpose(thickness)

    # calculate pseudo-inverses of B and C for use in generating control points
    pinvB = np.linalg.pinv(B)
    pinvC = np.linalg.pinv(np.transpose(C))

    # solve for control points
    Vx = np.matmul(pinvB, Px)
    Vx = np.transpose(np.matmul(Vx, pinvC))

    Vy = np.matmul(pinvB, Py)
    Vy = np.transpose(np.matmul(Vy, pinvC))

    Vz = np.matmul(pinvB, Pz)
    Vz = np.transpose(np.matmul(Vz, pinvC))

    # fix parameterization sample numbers to 100x100
    if fix_samples:
        uMin = np.min(U)
        uMax = np.max(U)
        uVect = np.linspace(uMin, uMax, 100)
        vVect = np.linspace(0, 1, 100)
        U, V = np.meshgrid(uVect, vVect)

    tri = mtra.Triangulation(np.ravel(U), np.ravel(V))

    # evaluate tensor product to get surface points. Operation is timed because it tends to be the slowest step
    startTime = time.perf_counter()
    X, Y, Z = EvaluateTensorProduct(Vx, Vy, Vz, tauU, tauV, degree, U, V, progressBar=progressBar)
    stopTime = time.perf_counter()
    print("Tensor product evaluation took {} seconds".format(stopTime - startTime))

    return X, Y, Z, tri

