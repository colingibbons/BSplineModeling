import numpy as np
from scipy.spatial.distance import euclidean
from skimage import measure
from skimage import segmentation
import time

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


def EvaluateTensorProduct(Vx, Vy, Vz, tauU, tauV, degree, U, V):

    numParamPointsV, numParamPointsU = np.shape(U)
    numControlPointsV, numControlPointsU = np.shape(Vx)

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

            print('param values:', m, n)

    return X, Y, Z

def readSlicePoints(baseName, startFrame, stopFrame):

    numSlices = (stopFrame - startFrame) + 1
    numPointsPerContour = np.zeros(numSlices)

    for i in range(numSlices):
        filePath = baseName + str(startFrame + i) + '.txt'

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
        Z[i, 0:numPointsPerContour[i]] = (startFrame + i) * 5

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
    b_mat = np.linalg.lstsq(A_mat_con, orderedPoints)[0]
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

def fitSplineOpen2D(points, numControlPoints, degree):

    numDataPoints = len(points)

    # set up parameters for spline fit
    n = numControlPoints - 1
    numCalcControlPoints = numControlPoints + degree

    # generate the knots (numKnots = n + 2d + 2)
    numKnots = n + (2 * degree) + 2
    tau = np.zeros(numKnots)
    numOpenKnots = numKnots - (2*degree)
    tau[degree:-degree] = np.linspace(0, 1, numOpenKnots)
    tau[-degree:] = 1

    # set up parameterization
    t = parameterizeClosedCurve(points, tau, degree)

    p_mat = np.transpose(points)
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
    b_mat = np.linalg.lstsq(A_mat_con, points)[0]
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

def reSampleAndSmoothPointsOpen(X, Y, Z, numPointsEachContour, numControlPoints, degree):

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
        b, tau, errorInFit = fitSplineOpen2D(points, numControlPoints, degree)
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
def generateNormalVectors(X, Y, Z, numPointsPerContour, numSlices):

    crossX = np.zeros((numSlices, numPointsPerContour - 1))
    crossY = np.zeros((numSlices, numPointsPerContour - 1))
    crossZ = np.zeros((numSlices, numPointsPerContour - 1))

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
            crossX[i, j] = crossVector[0]
            crossY[i, j] = crossVector[1]
            crossZ[i, j] = 0

        # check to see if the computed normal vectors have the correct orientation (i.e. point outward from the surface)
        # if they do not, reverse the direction by negating the vectors. Z-component is ignored because it's always 0
        if crossX[i, int(numPointsPerContour / 2)] > 0:
            crossX[i, :] *= -1
            crossY[i, :] *= -1

    return crossX, crossY, crossZ

# measures the fat thickness at each unit normal vector of the surface.
def measureFatThickness(X, Y, crossX, crossY, fatX, fatY, numSlices, numPointsPerContour, fatPointsPerSlice):
    # TODO some mechanism to account for the "amount" of each point that the vector passes through. Since each fat point
    # represents a voxel, the vector passing through the corner of a voxel vs directly through will contribute unevenly
    # to the overall thickness. Maybe do first check and then use magnitude of distance to scale the contribution to the
    # overall thickness by that pixel
    thicknessByPoint = np.zeros((numSlices, numPointsPerContour - 1))
    xFatPoints = np.zeros((numSlices, numPointsPerContour, max(fatPointsPerSlice)))
    yFatPoints = np.zeros((numSlices, numPointsPerContour, max(fatPointsPerSlice)))
    # fill point array with -1. Because of the way tissue data is extracted from PATS, all actual values will be
    # positive. Using -1 prevents filler points and actual data from being mistaken for one another
    xFatPoints.fill(-1)
    yFatPoints.fill(-1)
    for i in range(-1, numSlices - 1):
        for j in range(numPointsPerContour - 1):
            # generate a point arbitrarily far along the normal vector
            xDir = X[i, j] + (200 * crossX[i, j])
            yDir = Y[i, j] + (200 * crossY[i, j])
            thickness = 0
            # check each fat point for proximity to the line defined by normal vector
            for k in range(fatPointsPerSlice[i]):
                ab = np.sqrt((xDir - X[i, j])**2 + (yDir - Y[i, j])**2)
                ac = np.sqrt((xDir-fatX[i, k])**2 + (yDir - fatY[i, k])**2)
                bc = np.sqrt((X[i, j] - fatX[i, k])**2 + (Y[i, j] - fatY[i, k])**2)

                is_on_segment = abs(ac + bc - ab) < 0.02

                # update thickness measure if a point is found to be sufficiently close
                if is_on_segment:
                    thickness += 1
                    xFatPoints[i, j, k] = fatX[i, k]
                    yFatPoints[i, j, k] = fatY[i, k]

            thicknessByPoint[i, j] = thickness

    return thicknessByPoint, xFatPoints, yFatPoints

# get the set of points that will be used to generate a single fat spline surface
def getFatSurfacePoints(thicknessByPoint, xFatPoints, yFatPoints, X, Y, Z, numSlices, numPointsPerContour):

    fatSurfaceX = np.zeros((numSlices, numPointsPerContour - 1))
    fatSurfaceY = np.zeros((numSlices, numPointsPerContour - 1))
    fatSurfaceZ = Z[:, :numPointsPerContour - 1]
    for i in range(numSlices):
        for j in range(numPointsPerContour - 1):
            surfacePoint = (X[i, j], Y[i, j])
            if thicknessByPoint[i, j] == 0:
                fatSurfaceX[i, j] = X[i, j]
                fatSurfaceY[i, j] = Y[i, j]
            else:
                dist = 0
                for x, y in zip(xFatPoints[i, j, :], yFatPoints[i, j, :]):
                    if x == y == -1:
                        pass
                    else:
                        fatDist = abs(euclidean(surfacePoint, (x, y)))
                        if fatDist > dist:
                            fatSurfaceX[i, j] = x
                            fatSurfaceY[i, j] = y
                            dist = fatDist

    return fatSurfaceX, fatSurfaceY, fatSurfaceZ

# generate a list of fat deposits to create a collection of fat splines that reflect the non-continuous nature
# of fat surrounding the myocardium
def getFatDeposits(thicknessByPoint, numSlices):
    # create binarized version of thickness array for segmentation purposes
    thicknessBinary = np.copy(thicknessByPoint)
    thicknessBinary[thicknessBinary > 0] = 255

    # isolate fat deposits and "label" them with different numeric values to differentiate them
    fatDeposits = thicknessBinary > thicknessBinary.mean()
    deposit_labels, numDeposits = measure.label(fatDeposits, background=0, return_num=True, connectivity=1)

    # consider this as an alternative
    #deposit_labels = segmentation.watershed(thicknessByPoint, 15, mask=thicknessBinary)

    # combine deposits on opposite ends of the azimuth into a single deposit if they are adjacent
    for i in range(numSlices):
        if deposit_labels[i, 0] != 0 and deposit_labels[i, -1] != 0:
            obj = deposit_labels[i, -1]
            deposit_labels[deposit_labels == obj] = deposit_labels[i, 0]

    return deposit_labels, numDeposits

# This is the primary spline fitting routine, used to generate the spline surface which is open on the top and bottom,
# but closed "around" the heart
def fitSplineClosed3D(resampX, resampY, resampZ, numControlPointsU, numControlPointsV, degree, numPointsPerContour,
                      numSlices):
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

    # evaluate tensor product to get surface points. Operation is timed because it tends to be the slowest step
    startTime = time.perf_counter()
    X, Y, Z = EvaluateTensorProduct(Vx, Vy, Vz, tauU, tauV, degree, U, V)
    stopTime = time.perf_counter()
    print("Tensor product evaluation took {} seconds".format(stopTime - startTime))

    return X, Y, Z, Vx, Vy, Vz

# generates an open, 3D fat spline to represent a fat deposit at a given location around the myocardium
def fitSplineOpen3D(fatX, fatY, fatZ, numSlices, numPointsEachContour):
    # resample the data so each slice has the same number of points
    # do this by fitting each slice with B-spline curve

    # TODO define this to reflect the number of fat points in each slice of a given deposit
    resampleNumControlPoints = 4
    degree = 3
    resampX, resampY, resampZ, newXControl, newYControl, newZControl, numPointsPerContour, totalResampleError = \
        reSampleAndSmoothPointsOpen(fatX, fatY, fatZ, numPointsEachContour, resampleNumControlPoints, degree)

    # set up parameters for spline fit
    numControlPointsU = 6
    numControlPointsV = 6
    degree = 3
    m = numControlPointsU - 1
    n = numControlPointsV - 1
    numCalcControlPointsU = numControlPointsU + degree
    numCalcControlPointsV = numControlPointsV + degree
    M = numPointsPerContour - 1
    N = numSlices - 1

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
    U, V, firstKnotU, lastKnotU, firstKnotV, lastKnotV = parameterizeTube(resampX, resampY, resampZ, tauU, tauV, degree)

    # now we need to set up matrices to solve for mesh of control points
    # (B*V*T^T = P)

    B = np.zeros((M + 1, numCalcControlPointsU))
    for r in range(M + 1):
        for i in range(numCalcControlPointsU):
            uVal = U[0, r]
            B[r, i] = NVal(tauU, uVal, i - 1, degree, 0)

    # set up C matrix
    C = np.zeros((N + 1, numCalcControlPointsV))
    for s in range(N + 1):
        for j in range(numCalcControlPointsV):
            vVal = V[s, 0]
            C[s, j] = NVal(tauV, vVal, j - 1, degree, 0)


    # now set up Px, Py, and Pz matrices
    Px = np.transpose(resampX)
    Py = np.transpose(resampY)
    Pz = np.transpose(resampZ)

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

    # evaluate tensor product to get surface points. Operation is timed because it tends to be the slowest step
    startTime = time.perf_counter()
    X, Y, Z = EvaluateTensorProduct(Vx, Vy, Vz, tauU, tauV, degree, U, V)
    stopTime = time.perf_counter()
    print("Tensor product evaluation took {} seconds".format(stopTime - startTime))

    return X, Y, Z

def generateFatDepositSplines(X, Y, Z, fatSurfaceX, fatSurfaceY, fatSurfaceZ, deposits, numDeposits):
    # loop through each deposit and generate a spline surface for that deposit if it is sufficiently large
    # TODO figure out how to handle very small deposits/deposits that appear on only one slice
    fatDepositsX = []
    fatDepositsY = []
    fatDepositsZ = []
    for i in range(1, numDeposits + 1):
        # copy fat surface points into array
        currentDepositX = np.copy(fatSurfaceX)
        currentDepositY = np.copy(fatSurfaceY)
        currentDepositZ = np.copy(fatSurfaceZ)

        # copy myocardium points for use in completing the b-spline surface
        myoPointsX = np.copy(X)
        myoPointsY = np.copy(Y)
        myoPointsZ = np.copy(Z)

        # remove last column of each array (last surface point is repeat of first)
        myoPointsX = np.delete(myoPointsX, -1, 1)
        myoPointsY = np.delete(myoPointsY, -1, 1)
        myoPointsZ = np.delete(myoPointsZ, -1, 1)

        # remove points not corresponding with the current deposit from both sets of arrays
        currentDepositX[deposits != i] = 0
        currentDepositY[deposits != i] = 0

        myoPointsX[deposits != i] = 0
        myoPointsY[deposits != i] = 0

        # get number of nonzero points in each slice
        numPointsEachContour = (currentDepositX != 0).sum(1)

        # trim fat deposit array down to only include slices with nonzero thickness values
        currentDepositX = currentDepositX[numPointsEachContour != 0, :]
        currentDepositY = currentDepositY[numPointsEachContour != 0, :]
        currentDepositZ = currentDepositZ[numPointsEachContour != 0, :]

        # # trim myo array to only include surface points whose normal vectors have fat deposits
        # myoPointsX = myoPointsX[numPointsEachContour != 0, :]
        # myoPointsY = myoPointsY[numPointsEachContour != 0, :]
        # myoPointsZ = myoPointsZ[numPointsEachContour != 0, :]
        #
        # currentDepositX = np.concatenate((currentDepositX, myoPointsX), axis=1)
        # currentDepositY = np.concatenate((currentDepositY, myoPointsY), axis=1)
        # currentDepositZ = np.concatenate((currentDepositZ, myoPointsZ), axis=1)

        numSlicesNonZero = 0
        for j in numPointsEachContour:
            if j > 0:
                numSlicesNonZero += 1

        # only generate a spline surface if multiple slices have points for that deposit. Otherwise won't work
        if numSlicesNonZero > 1:
            # generate spline surface for the current fat deposit

            depositPointsEachContour = numPointsEachContour[numPointsEachContour > 0]
            depositSplineX, depositSplineY, depositSplineZ = fitSplineOpen3D(currentDepositX, currentDepositY,
                                                                             currentDepositZ, numSlicesNonZero,
                                                                             depositPointsEachContour)
            fatDepositsX.append(depositSplineX)
            fatDepositsY.append(depositSplineY)
            fatDepositsZ.append(depositSplineZ)

    return fatDepositsX, fatDepositsY, fatDepositsZ

