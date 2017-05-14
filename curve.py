import math
import numpy as np
from point import ControlPoint, ControlPoint3D
from poly import ControlPolygon

class Bezier:
    @classmethod
    def OneSubdivide(cls, control_points, poly1, poly2, u):
        new_control_points = []
        n = len(control_points) - 1
        if n == 0:
            return poly1.points + [control_points[0]] + poly2.points
        else:
            poly1.points = poly1.points + [control_points[0]]
            poly2.points = [control_points[len(control_points) - 1]] + poly2.points
            for i, point in enumerate(control_points[:n]):
                new_point = (point.add(control_points[i+1].sub(point).mult(u)))
                new_control_points.append(new_point)
            return cls.OneSubdivide(new_control_points, poly1, poly2, u)

    @classmethod
    def Subdivide(cls, control_points, m, u, closed=False):
        temp_control_points = list(control_points)
        if closed:
            temp_control_points.append(control_points[0])

        if m == 1:
            return cls.OneSubdivide(temp_control_points, ControlPolygon(), ControlPolygon(), u)
        else:
            new_control_points = cls.OneSubdivide(temp_control_points, ControlPolygon(), ControlPolygon(), u)
            two_n = len(new_control_points)
            return cls.Subdivide(new_control_points[0:int(two_n/2) + 1], m-1, u) + cls.Subdivide(new_control_points[int(two_n/2):two_n], m-1, u)
            
    @classmethod
    def GetPointsDirectly(cls, control_points, num_samples, closed=False):
        temp_control_points = list(control_points)
        if closed:
            temp_control_points.append(control_points[0])

        # Calculate sampling u's
        u_lst = []
        base_division = 1/(num_samples-1)
        for u in range(num_samples):
            u_lst.append(u*base_division)

        curve_points = []
        n = len(temp_control_points) - 1
        for u in u_lst:
            point = ControlPoint(0, 0)
            for i in range(n+1):
                n_choose_i = (math.factorial(n)/(math.factorial(i)*math.factorial(n - i)))
                basis = n_choose_i*math.pow(u, i)*math.pow(1 - u, n - i)
                point = point.add(temp_control_points[i].mult(basis))
            curve_points.append(point)

        return curve_points

    @classmethod
    def Basis(cls, i, n, u):
        return (math.factorial(n)/(math.factorial(i)*math.factorial(n-i))*(math.pow(u, i)*math.pow(1-u, n-i)))

    @classmethod
    def GetSurfacePoint(cls, u, w, point_matrix, u_closed=False, v_closed=False):
        if u_closed:
            pass
        if v_closed:
            pass

        outer_point = ControlPoint3D(0,0,0)
        for i, row in enumerate(point_matrix):
            m = len(point_matrix) - 1
            for j, column in enumerate(row):
                n = len(row) - 1
                basis_u = cls.Basis(i,m,u)
                basis_w = cls.Basis(j,n,w)
                temp_p = ControlPoint3D(point_matrix[i][j].GetCo(0), point_matrix[i][j].GetCo(1), point_matrix[i][j].GetCo(2))
                temp_p = temp_p.mult(basis_w*basis_u)
                outer_point = outer_point.add(temp_p)
        return outer_point

class BSpline:
    class Cubic:
        D = 4 # degree (3) + 1

        @classmethod
        def CreateKnotVector(cls, control_points, D):
            n = len(control_points) - 1
            length_T = n+D+1
            T = [0]*length_T
            
            for i in range(length_T):
                T[i] = i
                
            return T

        @classmethod
        def CreateUniformKnotVector(cls, n, D):
            length_T = n+D+1
            T = [0]*length_T
            
            for i in range(length_T):
                T[i] = i

            return T

        @classmethod
        def Basis(cls, t, i, d, u):
            if d == 1:
                if (u >= t[i] and u < t[i + 1]) or (np.isclose(t[i],t[i + 1]) and np.isclose(unicode_iterator, [i])):
                    return 1
                else:
                    return 0
            else:
                first_den = t[i+d-1]-t[i]
                second_den = t[i+d]-t[i+1]
                first_elem = 0
                second_elem = 0
                if first_den != 0: # to avoid divide by 0 errors
                    first_elem = ((u-t[i])*BSpline.Cubic.Basis(t, i, d-1, u))/(first_den)
                if second_den != 0:
                    second_elem = ((t[i+d]-u)*BSpline.Cubic.Basis(t, i+1, d-1, u))/(second_den)
                return first_elem + second_elem

        @classmethod
        def GetPointsDirectly(cls, control_points, num_samples, closed=False):
            temp_control_points = list(control_points)
            if closed and len(temp_control_points) > 3:
                for i in range(0, 3):
                    temp_control_points.append(control_points[i])
                
            # Calculate sampling u's
            u_lst = []
            if (num_samples-1) != 0:
                base_division = 1/(num_samples-1)
            else:
                base_division = 0
            for i in range(3, len(temp_control_points)):
                for u in range(num_samples):
                    u_lst.append(i+u*(base_division))

            curve_points = []
            n = len(temp_control_points) - 1

            T = BSpline.Cubic.CreateKnotVector(temp_control_points, BSpline.Cubic.D)
            for u in u_lst:
                point = ControlPoint(0, 0)
                for i in range(0, n+1):
                    basis = BSpline.Cubic.Basis(T, i, BSpline.Cubic.D, u)
                    point = point.add(temp_control_points[i].mult(basis))
                curve_points.append(point)
            return curve_points

        @classmethod
        def GetSurfacePoint(cls, u, w, point_matrix):
            outer_point = ControlPoint3D(0,0,0)
            for i, row in enumerate(point_matrix):
                m = len(point_matrix) - 1
                Tu = BSpline.Cubic.CreateUniformKnotVector(m, BSpline.Cubic.D)
                for j, column in enumerate(row):
                    n = len(row) - 1
                    Tw = BSpline.Cubic.CreateUniformKnotVector(n, BSpline.Cubic.D)
                    basis_u = cls.Basis(Tu, i, BSpline.Cubic.D, u)
                    basis_w = cls.Basis(Tw, j, BSpline.Cubic.D, w)
                    temp_p = ControlPoint3D(point_matrix[i][j].GetCo(0), point_matrix[i][j].GetCo(1), point_matrix[i][j].GetCo(2))
                    temp_p = temp_p.mult(basis_w*basis_u)
                    outer_point = outer_point.add(temp_p)
            return outer_point

    class Quadric:
        M = (1/4)*np.array([[3.0, 1.0, 0.0],
                            [1.0, 3.0, 0.0],
                            [0.0, 3.0, 1.0],
                            [0.0, 1.0, 3.0]], dtype="float64")

        D = 3 # degree (2) + 1

        @classmethod
        def CreateKnotVector(cls, control_points, D):
            n = len(control_points) - 1
            length_T = n+D+1
            T = [0]*length_T
            for i in range(length_T):
                T[i] = i
            return T
        
        
        @classmethod
        def OneSubdivide(cls, control_points, knots, closed=False):
            new_control_points = []
            n = len(control_points)
            splitting_matrix = (1/4)*np.matrix([[3, 1, 0],
                                                [1, 3, 0],
                                                [0, 3, 1],
                                                [0, 1, 3]])

            for i, point in enumerate(control_points[1:-1]):
                Px = np.matrix([control_points[i].x, control_points[i+1].x, control_points[i+2].x]).transpose()
                Py = np.matrix([control_points[i].y, control_points[i+2].y, control_points[i+2].y]).transpose()
                new_points_x = splitting_matrix.dot(Px)
                new_points_y = splitting_matrix.dot(Py)
                split_points = [control_points[i].mult((3/4)).add(control_points[i+1].mult((1/4))),
                                control_points[i].mult((1/4)).add(control_points[i+1].mult((3/4))),
                                control_points[i+1].mult((3/4)).add(control_points[i+2].mult((1/4))),
                                control_points[i+1].mult((1/4)).add(control_points[i+2].mult((3/4)))]
                new_control_points = new_control_points[:-2] + split_points
            return new_control_points

        @classmethod
        def Subdivide(cls, control_points, m, u, closed=False):
            new_control_points = list(control_points)
            if closed and len(new_control_points) > 2:
                for i in range(0, 3):
                    new_control_points.append(control_points[i])

            knots = cls.CreateKnotVector(control_points, BSpline.Quadric.D)
            for j in range(m):
                new_control_points = cls.OneSubdivide(new_control_points, knots, closed=closed)
            return new_control_points

        @classmethod
        def GetPointsDirectly(cls, control_points, num_samples):
            # Calculate sampling u's
            u_values = []
            base_division = 1/(num_samples-1)
            for u in range(num_samples):
                u_values.append(u*base_division)
            curve_points = []
            n = len(control_points) - 1

            T = BSpline.Cubic.CreateKnotVector(control_points, BSpline.Cubic.D)
            for i in range(1, len(control_points) - BSpline.Quadric.D + 2):
                for u in u_values:
                    pointx = 0
                    pointy = 0
                    u_arr = np.array([u*u, u, 1], dtype="float64")
                    Px = np.array([control_points[i-1].x, control_points[i].x, control_points[i+1].x], dtype="float64").transpose()
                    Py = np.array([control_points[i-1].y, control_points[i].y, control_points[i+1].y], dtype="float64").transpose()
                    pointx = u_arr.dot(BSpline.Quadric.M).dot(Px)
                    pointy = u_arr.dot(BSpline.Quadric.M).dot(Py)
                    curve_points.append(ControlPoint(pointx, pointy))
            return curve_points
