import math
import numpy as np
from point import ControlPoint
from scipy.spatial import Delaunay, Voronoi
from sklearn.neighbors import NearestNeighbors


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def ConstructNPArray(points):
    np_points = np.array([])
    if len(points) > 0:
        np_points = np.array([[points[0].x, points[0].y]])
        # Construct np array from sample points.
        for point in points[1:]:
            np_points = np.append(np_points, [[point.x, point.y]], axis=0)
    return np_points

def GetVoronoiVerts(sample_points):
    S = ConstructNPArray(sample_points)
    V = np.array([])
    if len(sample_points) > 0:
        V = ArrayToCP(Voronoi(S).vertices)
    return V

def Crust(sample_points):
    reconstructed = []
    S = ConstructNPArray(sample_points)
    V = Voronoi(S).vertices
    SuV = np.concatenate((S, V), axis=0)
    del_SuV = Delaunay(SuV)
    len_S = len(S)
    print(del_SuV.simplices)
    for idx, simplex in enumerate(del_SuV.simplices):
        # For each simplex
        if (simplex[0] < len_S and simplex[1] < len_S and 
            [simplex[0], simplex[1]] not in reconstructed and [simplex[1], simplex[0]] not in reconstructed):
            # If edge 1 is in sample points
            #print("Added {0}, in reconstructed? {1}".format([simplex[0], simplex[1]], ([simplex[0], simplex[1]] and [simplex[1], simplex[0]]) in reconstructed))
            reconstructed.append([simplex[0], simplex[1]])
        if (simplex[1] < len_S and simplex[2] < len_S and
            [simplex[1], simplex[2]] not in reconstructed and [simplex[2], simplex[1]] not in reconstructed):
            # If edge 2 is in sample points
            #print("Added {0}, in reconstructed? {1}".format([simplex[1], simplex[2]], ([simplex[1], simplex[2]] and [simplex[2], simplex[1]]) in reconstructed))
            reconstructed.append([simplex[1], simplex[2]])
        if (simplex[0] < len_S and simplex[2] < len_S and 
            [simplex[0], simplex[2]] not in reconstructed and [simplex[2], simplex[0]] not in reconstructed):
            # If edge 3 is in sample points
            #print("Added {0}, in reconstructed? {1}".format([simplex[0], simplex[2]], ([simplex[0], simplex[2]] and [simplex[2], simplex[0]]) in reconstructed))
            reconstructed.append([simplex[0], simplex[2]])

    print(reconstructed)
    # Order simplices based on shared verts.
    E = [reconstructed.pop(0)]
    while(len(reconstructed) > 0):
        for idx, simplex in enumerate(reconstructed):
            if E[-1][1] == simplex[0]:
                E.append(reconstructed.pop(idx))
                continue
            elif E[-1][1] == simplex[1]:
                E.append(np.flipud(reconstructed.pop(idx)))
                continue

    # Extract point ordering from ordered simplices.
    ordered_points = []
    for s in E:
        ordered_points.append(ControlPoint(SuV[s[0]][0], 
                                           SuV[s[0]][1]))
    ordered_points.append(ordered_points[0])
    return ordered_points


def IdxArrayToCP(source_array, idx_array):
    points = []
    for s in idx_array:
        points.append(ControlPoint(source_array[s[0]][0], 
                                   source_array[s[0]][1]))
    return points

def ArrayToCP(np_array):
    points = []
    for s in np_array:
        points.append(ControlPoint(s[0], s[1]))
    return points
    

def NNCrust(sample_points):
    reconstructed = []
    np_points = ConstructNPArray(sample_points)
    del_p = Delaunay(np_points)
    for idx, point in enumerate(np_points):
        closest_point = None;
        opposite_closest_point = None;
        shortest_length = None;
        second_shortest_length = None;
        for simplex in del_p.simplices:
            if idx in simplex:
                for j in simplex:
                    if j != idx: # for two possible pq
                        length = vector_length(point, np_points[j])
                        if shortest_length is None or length < shortest_length:
                            shortest_length = length
                            closest_point = j
        # Closest point should now be incident to shortest edge
        v1 = np_points[closest_point] - np_points[idx]
        for simplex in del_p.simplices:
            if idx in simplex:
                for j in simplex:
                    if j != idx:
                        length = vector_length(point, np_points[j])
                        v2 = np_points[j] - np_points[idx]
                        print(angle_between(v1, v2))
                        if ((angle_between(v1, v2) > math.pi/2) and 
                            ((second_shortest_length is None) or 
                             (second_shortest_length > length))):
                            second_shortest_length = length
                            opposite_closest_point = j
        reconstructed.append([closest_point, idx, opposite_closest_point])

    # Order simplices based on shared edges.
    E = [reconstructed.pop(0)]
    while(len(reconstructed) > 0):
        for idx, simplex in enumerate(reconstructed):
            if simplex[0] == E[-1][1] and simplex[1] == E[-1][2]:
                E.append(reconstructed.pop(idx))
                continue
            if simplex[1] == E[-1][2] and simplex[2] == E[-1][1]:
                E.append(np.flipud(reconstructed.pop(idx)))
                continue

    # Extract point ordering from ordered simplices.
    ordered_points = []
    for s in E:
        ordered_points.append(ControlPoint(np_points[s[0]][0], 
                                           np_points[s[0]][1]))
    ordered_points.append(ordered_points[0])
    return ordered_points
        

def vector_length(a, b):
    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))
