import math
import mesh as ms
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from tkinter import *
from tkinter import messagebox, filedialog
from curve import BSpline, Bezier
from point import ControlPoint
from poly import ControlPolygon
from mesh import Doo_Sabin, Catmull_Clark, Loop
from reconstruction import NNCrust, Crust, GetVoronoiVerts

class Observable:
    def __init__(self, initial_value=[]):
        self.control_points = initial_value
        self.callbacks = {}

    def addCallback(self, func):
        self.callbacks[func] = 1

    def delCallback(self, func):
        del self.callback[func]

    def _docallbacks(self):
        for func in self.callbacks:
            func(self.control_points)

    def set(self, control_points):
        self.control_points = control_points
        self._docallbacks()

    def get(self):
        return self.control_points

    def unset(self):
        self.control_points = None

class Curve:
    def __init__(self):
        self.control_points = Observable()
        self.curve_closed = False
        self.curve_points = []
        self.curve_type = 1
        self.point_is_moving = False
        self.moving_point = None        
        self.m = 1

class Model:
    def __init__(self):
        self.surface_curve = Curve()
        self.sweep_curve = Curve()
        self.active_curve = self.surface_curve
        self.clearPoints()
        self.mesh = ms.Mesh()
        self.smoothed_mesh = ms.Mesh()

    def getControlPoints(self):
        return self.active_curve.control_points.get()

    def setControlPoints(self, new_control_points):
        self.active_curve.control_points.set(new_control_points)

    def setCurvePoints(self, new_curve_points):
        self.active_curve.curve_points = new_curve_points

    def getM(self):
        return self.active_curve.m

    def getCurveClosed(self):
        return self.active_curve.curve_closed

    def setCurveClosed(self, closed):
        self.active_curve.curve_closed = closed

    def addPoint(self, point):
        self.active_curve.control_points.get().append(point)
        self.active_curve.control_points.set(self.active_curve.control_points.get())
        return point

    def removePoint(self, point):
        self.getControlPoints().remove(point)
        self.setControlPoints(self.getControlPoints())

    def clearPoints(self):
        self.setControlPoints([])

    def setM(self, m):
        self.active_curve.m = m

    def generateMesh(self, gen_technique, slices=None):
        self.mesh = ms.Mesh()
        points = self.surface_curve.curve_points
        if gen_technique == 0:
            # Extrusion
            slice_size = 20
            try:
                slices = int(slices)
                for i in range(0, slices):
                    for j in range(0, len(points)-1):
                        vert_11 = [points[j].x, points[j].y, slice_size*i]
                        vert_12 = [points[j+1].x, points[j+1].y, slice_size*i]
                        vert_22 = [points[j].x, points[j].y, slice_size*(i+1)]
                        vert_21 = [points[j+1].x, points[j+1].y, slice_size*(i+1)]
                        direction = None
                        if j == 0:
                            direction = 'v'
                        else:
                            direction = 'u'
                        self.mesh.AddFacetQuad(vert_11[0], vert_11[1], vert_11[2],
                                               vert_12[0], vert_12[1], vert_12[2],
                                               vert_21[0], vert_21[1], vert_21[2],
                                               vert_22[0], vert_22[1], vert_22[2], direction=direction)

            except ValueError:
                if slices != "":
                    messagebox.showinfo("Value Error", "That is not a valid slice value. Mesh will not be generated.")
        elif gen_technique == 1:
            # Rotation
            try:
                slices = int(slices)
                theta = (360 / slices)*(math.pi/180)
                for i in range(0, slices):
                    for j in range(0, len(points)-1):
                        direction = None
                        if j == 0:
                            direction = 'v'
                        else:
                            direction = 'u'
                        vert_11 = [points[j].x, points[j].y, 0]
                        vert_12 = [points[j+1].x, points[j+1].y, 0]
                        vert_22 = [points[j].x, points[j].y, 0]
                        vert_21 = [points[j+1].x, points[j+1].y, 0]
                        
                        rot_mat_1 = np.matrix([[math.cos(theta*i), 0, math.sin(theta*i)],
                                               [0, 1, 0],
                                               [math.sin(theta*i)*(-1), 0, math.cos(theta*i)]])

                        rot_mat_2 = np.matrix([[math.cos(theta*(i+1)), 0, math.sin(theta*(i+1))],
                                               [0, 1, 0],
                                               [math.sin(theta*(i+1))*(-1), 0, math.cos(theta*(i+1))]])

                        vert_11 = (vert_11*rot_mat_1).tolist()[0]
                        vert_12 = (vert_12*rot_mat_1).tolist()[0]
                        vert_22 = (vert_22*rot_mat_2).tolist()[0]
                        vert_21 = (vert_21*rot_mat_2).tolist()[0]

                        self.mesh.AddFacetQuad(vert_11[0], vert_11[1], vert_11[2],
                                               vert_12[0], vert_12[1], vert_12[2],
                                               vert_21[0], vert_21[1], vert_21[2],
                                               vert_22[0], vert_22[1], vert_22[2], direction=direction)

            except ValueError:
                if slices != "":
                    messagebox.showinfo("Value Error", "That is not a valid slice value. Mesh will not be generated.")
        elif gen_technique == 2:
            sweep_points = self.sweep_curve.curve_points
            # Sweep
            try:
                slices = len(sweep_points)
                for i in range(0, slices-1):
                    # Find offset from one point on surface curve to one point on sweep curve.
                    offset_1 = [points[0].x, points[0].y - sweep_points[i].x, 0 - sweep_points[i].y]
                    offset_2 = [points[0].x, points[0].y - sweep_points[i+1].x, 0 - sweep_points[i+1].y]
                    for j in range(0, len(points)-1):
                        direction = None
                        if j == 0:
                            direction = 'v'
                        else:
                            direction = 'u'
                        vert_11 = np.array([points[j].x, points[j].y, 0]) - offset_1
                        vert_12 = np.array([points[j+1].x, points[j+1].y, 0]) - offset_1
                        vert_22 = np.array([points[j].x, points[j].y, 0]) - offset_2
                        vert_21 = np.array([points[j+1].x, points[j+1].y, 0]) - offset_2
                        self.mesh.AddFacetQuad(vert_11[0], vert_11[1], vert_11[2],
                                               vert_12[0], vert_12[1], vert_12[2],
                                               vert_21[0], vert_21[1], vert_21[2],
                                               vert_22[0], vert_22[1], vert_22[2], direction=direction)

            except ValueError:
                if slices != "":
                    messagebox.showinfo("Value Error", "That is not a valid slice value. Mesh will not be generated.")

        print("# Faces: {0}".format(self.mesh.GetNumberFacets()))        
        if self.mesh.SelfIntersection():
            messagebox.showinfo("Intersection", "The generated mesh may be self-intersecting.")

        with open("output.off", "w") as f:
            num_verts = self.mesh.GetNumberVertices()
            num_faces = self.mesh.GetNumberFacets()
            #f.write("{0} {1}\n".format(num_verts, num_faces))
            f.write("OFF\n")
            f.write("{0} {1} 0\n".format(num_verts, num_faces, 0))

            vert_list = []
            for i in range(0, num_verts):
                vert_list.append(self.mesh.GetGeomVertex(i))

            for vert in vert_list:
                f.write("{0} {1} {2}\n".format(vert.GetCo(0), vert.GetCo(1), vert.GetCo(2)))

            for i in range(0, num_faces):
                face = self.mesh.GetFacet(i)
                face_verts = []
                num_face_verts = face.GetNumberVertices()
                for j in range(0, num_face_verts):
                    face_verts.append(face.GetVertexInd(j))
                f.write("{0} ".format(num_face_verts))
                for v in face_verts:
                    f.write("{0} ".format(v))
                f.write("\n")

    def generateSmooth(self, smooth_technique, u_var, v_var, u_closed, v_closed):
        self.smoothed_mesh = ms.Mesh()

        if smooth_technique == 0: # Bezier surface
            u = int(u_var)
            w = int(v_var)
            u_range = []
            u_increment = 1/(u-1)
            w_range = []
            w_increment = 1/(w-1)
            for i in range(u):
                u_range.append(i*u_increment)
            for j in range(w):
                w_range.append(j*w_increment)

            for i, u_cur in enumerate(u_range[:-1]):
                for j, w_cur in enumerate(w_range[:-1]):
                    point_11 = Bezier.GetSurfacePoint(u_cur, w_cur, self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed) 
                    point_22 = Bezier.GetSurfacePoint(u_range[i+1], w_cur, self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed) 
                    point_12 = Bezier.GetSurfacePoint(u_cur, w_range[j+1], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed) 
                    point_21 = Bezier.GetSurfacePoint(u_range[i+1], w_range[j+1], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    self.smoothed_mesh.AddFacetQuad(point_11.x, point_11.y, point_11.z,
                                                    point_12.x, point_12.y, point_12.z,
                                                    point_21.x, point_21.y, point_21.z,
                                                    point_22.x, point_22.y, point_22.z)
            if u_closed:
                for j, w_cur in enumerate(w_range[:-1]):
                    point_11 = Bezier.GetSurfacePoint(w_cur, u_range[0], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    point_22 = Bezier.GetSurfacePoint(w_cur, u_range[-1], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    point_12 = Bezier.GetSurfacePoint(w_range[j+1], u_range[0], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    point_21 = Bezier.GetSurfacePoint(w_range[j+1], u_range[-1], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    self.smoothed_mesh.AddFacetQuad(point_11.x, point_11.y, point_11.z,
                                                    point_12.x, point_12.y, point_12.z,
                                                    point_21.x, point_21.y, point_21.z,
                                                    point_22.x, point_22.y, point_22.z)
            if v_closed:
                for i, u_cur in enumerate(u_range[:-1]):
                    point_11 = Bezier.GetSurfacePoint(w_range[-1], u_range[i], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    point_22 = Bezier.GetSurfacePoint(w_range[0], u_range[i+1], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    point_12 = Bezier.GetSurfacePoint(w_range[-1], u_range[i], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    point_21 = Bezier.GetSurfacePoint(w_range[0], u_range[i+1], self.mesh.uvMatrix, u_closed=u_closed, v_closed=v_closed)
                    self.smoothed_mesh.AddFacetQuad(point_11.x, point_11.y, point_11.z,
                                                    point_12.x, point_12.y, point_12.z,
                                                    point_21.x, point_21.y, point_21.z,
                                                    point_22.x, point_22.y, point_22.z)
                    
            print("Non-Planarity measure: {0}".format(self.smoothed_mesh.GetPlanarity()))
        if smooth_technique == 1: # Bicubic B-Spline surface
            # Assuming quad control mesh.
            u = int(u_var)
            w = int(v_var)
            u_range = []
            u_increment = 1/(u-1)
            w_range = []
            w_increment = 1/(w-1)

            temp_uvMatrix = self.mesh.uvMatrix

            if u_closed:
                for row in temp_uvMatrix:
                    row.append(row[0])
                    row.append(row[1])
                    row.append(row[2])

            if v_closed:
                temp_uvMatrix.append(temp_uvMatrix[0])
                temp_uvMatrix.append(temp_uvMatrix[1])
                temp_uvMatrix.append(temp_uvMatrix[2])

            # Adjust for i.
            for i in range(3, len(temp_uvMatrix)):
                for u_cur in range(u):
                    u_range.append(i+u_cur*(u_increment))
            for i in range(3, len(temp_uvMatrix[0])):
                for w_cur in range(w):
                    w_range.append(i+w_cur*(w_increment))

            for i, u_cur in enumerate(u_range[:-1]):
                for j, w_cur in enumerate(w_range[:-1]):
                    point_11 = BSpline.Cubic.GetSurfacePoint(u_cur, w_cur, temp_uvMatrix) 
                    point_22 = BSpline.Cubic.GetSurfacePoint(u_range[i+1], w_cur, temp_uvMatrix) 
                    point_12 = BSpline.Cubic.GetSurfacePoint(u_cur, w_range[j+1], temp_uvMatrix) 
                    point_21 = BSpline.Cubic.GetSurfacePoint(u_range[i+1], w_range[j+1], temp_uvMatrix) 
                    self.smoothed_mesh.AddFacetQuad(point_11.x, point_11.y, point_11.z,
                                                    point_12.x, point_12.y, point_12.z,
                                                    point_21.x, point_21.y, point_21.z,
                                                    point_22.x, point_22.y, point_22.z)

            print("Non-Planarity measure: {0}".format(self.smoothed_mesh.GetPlanarity()))

        if smooth_technique == 2: # Doo-Sabin
            self.smoothed_mesh = Doo_Sabin.Subdivide(self.mesh)
        if smooth_technique == 3: # Bicubic Catmull-Clark
            self.smoothed_mesh = Catmull_Clark.Subdivide(self.mesh)
        if smooth_technique == 4: # Loop
            self.smoothed_mesh = Loop.Subdivide(self.mesh)

        with open("output_smoothed.off", "w") as f:
            num_verts = self.smoothed_mesh.GetNumberVertices()
            num_faces = self.smoothed_mesh.GetNumberFacets()
            #f.write("{0} {1}\n".format(num_verts, num_faces))
            f.write("OFF\n")
            f.write("{0} {1} 0\n".format(num_verts, num_faces, 0))

            vert_list = []
            for i in range(0, num_verts):
                vert_list.append(self.smoothed_mesh.GetGeomVertex(i))

            for vert in vert_list:
                f.write("{0} {1} {2}\n".format(vert.GetCo(0), vert.GetCo(1), vert.GetCo(2)))

            for i in range(0, num_faces):
                face = self.smoothed_mesh.GetFacet(i)
                face_verts = []
                num_face_verts = face.GetNumberVertices()
                for j in range(0, num_face_verts):
                    face_verts.append(face.GetVertexInd(j))
                f.write("{0} ".format(num_face_verts))
                for v in face_verts:
                    f.write("{0} ".format(v))
                f.write("\n")

    def loadMesh(self, filename):
        self.mesh = ms.Mesh()
        with open(filename, 'r') as f:
            f.readline() # Throw away OFF line
            line_1 = f.readline().split()
            num_verts = int(line_1[0])
            num_faces = int(line_1[1])
            verts = []
            for line in f:
                split = line.split()
                if len(split) > 0 and "#" not in split[0]:
                    if len(split) == 3:
                        verts.append(ms.GeomVert(float(split[0]), float(split[1]), float(split[2])))
                    if len(split) > 3:
                        num_face_verts = int(split[0])
                        indices = list(map(int, split[1:1+num_face_verts]))
                        if num_face_verts == 3:
                            self.mesh.AddFacetTri(verts[indices[0]].GetCo(0), verts[indices[0]].GetCo(1), verts[indices[0]].GetCo(2),
                                                  verts[indices[1]].GetCo(0), verts[indices[1]].GetCo(1), verts[indices[1]].GetCo(2),
                                                  verts[indices[2]].GetCo(0), verts[indices[2]].GetCo(1), verts[indices[2]].GetCo(2))
                        elif num_face_verts == 4:
                            self.mesh.AddFacetQuad(verts[indices[0]].GetCo(0), verts[indices[0]].GetCo(1), verts[indices[0]].GetCo(2),
                                                   verts[indices[1]].GetCo(0), verts[indices[1]].GetCo(1), verts[indices[1]].GetCo(2),
                                                   verts[indices[2]].GetCo(0), verts[indices[2]].GetCo(1), verts[indices[2]].GetCo(2),
                                                   verts[indices[3]].GetCo(0), verts[indices[3]].GetCo(1), verts[indices[3]].GetCo(2))

        print("Finished loading mesh.")
        print("# Faces = {0}, # Verts = {1}".format(self.mesh.GetNumberFacets(), self.mesh.GetNumberVertices()))

class View(Toplevel):
    def __init__(self, master, curve_var, surface_var, curve_type_var, axis_var, subdivision_var, slice_var, smooth_var, u_var, v_var, u_closed_var, v_closed_var):
        self.CANVAS_WIDTH = 400
        self.CANVAS_HEIGHT = 400
        
        Toplevel.__init__(self, master)
        self.protocol('WM_DELETE_WINDOW', self.master.destroy)
        curve_var.set(1)
        surface_var.set(0)
        curve_type_var.set(0)
        axis_var.set(0)

        curve_type_frame = Frame(self, bd='2', relief=SUNKEN)
        curve_type_frame.grid(row=0, column=0, columnspan=2, sticky=W+E+N+S)
        Radiobutton(curve_type_frame, text="Bezier (DeCasteljau Sub)", variable=curve_var, value=2).grid(row=0, column=1)
        Radiobutton(curve_type_frame, text="Cubic B-Spline", variable=curve_var, value=3).grid(row=0, column=2)
        Radiobutton(curve_type_frame, text="Direct Bezier", variable=curve_var, value=1).grid(row=0, column=0)
        Radiobutton(curve_type_frame, text="Quadric B-Spline (Subdivision)", variable=curve_var, value=4).grid(row=0, column=3)
        Radiobutton(curve_type_frame, text="Points", variable=curve_var, value=5).grid(row=0, column=4)
        self.b_clear = Button(curve_type_frame, text="Clear Control Points")
        self.b_clear.grid(row=0, column=5)

        surface_type = Frame(self, bd='2', relief=SUNKEN)
        surface_type.grid(row=1, column=0, sticky=W+E+N+S)
        Label(surface_type, text="Surface Type", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0)
 
        Radiobutton(surface_type, text="Extrude", value=0, variable=surface_var, justify=LEFT).grid(row=1,column=0, sticky=W)
        Radiobutton(surface_type, text="Revolution", value=1, variable=surface_var, justify=LEFT).grid(row=2,column=0, sticky=W)
        Radiobutton(surface_type, text="Sweep", value=2, variable=surface_var, justify=LEFT).grid(row=3,column=0, sticky=W)
        
        surface_options = Frame(self, bd='2', relief=SUNKEN)
        surface_options.grid(row=2, column=0, sticky=W+E+N+S)
        Label(surface_options, text="Surface Options", font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2)
        Label(surface_options, text="Subdivisions", font=('TKDefaultFont', 10, 'bold')).grid(row=1, column=0)
        Label(surface_options, text="Slices", font=('TKDefaultFont', 10, 'bold')).grid(row=2, column=0)
        Label(surface_options, text="u", font=('TKDefaultFont', 10, 'bold')).grid(row=3, column=0)
        Label(surface_options, text="v", font=('TKDefaultFont', 10, 'bold')).grid(row=4, column=0)
        self.subdivisions = Entry(surface_options, textvariable=subdivision_var).grid(row=1, column=1)
        self.slice = Entry(surface_options, textvariable=slice_var).grid(row=2, column=1)
        self.u = Entry(surface_options, textvariable=u_var).grid(row=3, column=1)
        self.v = Entry(surface_options, textvariable=v_var).grid(row=4, column=1)
        self.u_closed = Checkbutton(surface_options, text="Close u", variable=u_closed_var).grid(row=5, column=0)
        self.v_closed = Checkbutton(surface_options, text="Close v", variable=v_closed_var).grid(row=5, column=1)

        surface_or_swoop = Frame(self, bd='2', relief=SUNKEN)
        surface_or_swoop.grid(row=3, column=0, rowspan=2, sticky=W+E+N+S)
        Radiobutton(surface_or_swoop, text="Surface", value=0, variable=curve_type_var, justify=LEFT).grid(row=0, column=0, sticky=W)
        Radiobutton(surface_or_swoop, text="Sweep", value=1, variable=curve_type_var, justify=LEFT).grid(row=1, column=0, sticky=W)

        axis_of_revolution = Frame(self, bd='2', relief=SUNKEN)
        axis_of_revolution.grid(row=4, column=0, rowspan=2, sticky=W+E+N+S)
        Label(axis_of_revolution, text="Axis of Revolution", font=('TKDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=3)
        Radiobutton(axis_of_revolution, text="x", variable=axis_var, value=0).grid(row=1, column=0)
        Radiobutton(axis_of_revolution, text="y", variable=axis_var, value=1).grid(row=1, column=1)
        Radiobutton(axis_of_revolution, text="z", variable=axis_var, value=2).grid(row=1, column=2)

        gen = Frame(self, bd='2', relief=SUNKEN)
        gen.grid(row=5, column=0, rowspan=2, sticky=W+E+N+S)
        self.b_generate = Button(gen, text="Generate")
        self.b_generate.grid(row=0, column=0, sticky = W+E+N+S, padx=5, pady=5)

        smooth = Frame(self, bd='2', relief=SUNKEN)
        smooth.grid(row=6, column=0, rowspan=2, sticky=W+E+N+S)
        self.b_smooth = Button(smooth, text="Generate Smoothed")
        self.b_smooth.grid(row=0, column=0, sticky = W+E+N+S, padx=5, pady=5)

        load_file = Frame(self, bd='2', relief=SUNKEN)
        load_file.grid(row=7, column=0, sticky=W+E+N+S)
        self.b_load_file = Button(load_file, text="Load Mesh")
        self.b_load_file.grid(row=0, column=0, sticky = W+E+N+S, padx=5, pady=5)

        smooth_picker = Frame(self, bd='2', relief=SUNKEN)
        smooth_picker.grid(row=6, column=1, sticky=W+N+E+S, padx=7, pady=7)
        Radiobutton(smooth_picker, text="Bezier Surface", value=0, variable=smooth_var, justify=LEFT).grid(row=0, column=0, sticky=W)
        Radiobutton(smooth_picker, text="BSpline Surface", value=1, variable=smooth_var, justify=LEFT).grid(row=0, column=1, sticky=W)
        Radiobutton(smooth_picker, text="Doo-Sabin Surface", value=2, variable=smooth_var, justify=LEFT).grid(row=0, column=2, sticky=W)
        Radiobutton(smooth_picker, text="Catmull-Clark Surface", value=3, variable=smooth_var, justify=LEFT).grid(row=0, column=3, sticky=W)
        Radiobutton(smooth_picker, text="Loop Surface", value=4, variable=smooth_var, justify=LEFT).grid(row=0, column=4, sticky=W)

        reconstruct_frame = Frame(self, bd='2', relief=SUNKEN)
        reconstruct_frame.grid(row=7, column=1, sticky=W+N+E+S, padx=7, pady=7)
        self.b_nncrust = Button(reconstruct_frame, text="NN-Crust")
        self.b_nncrust.grid(row=0, column=0)
        self.b_crust = Button(reconstruct_frame, text="Crust")
        self.b_crust.grid(row=0, column=1)

        self.canvas = Canvas(self, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT)
        self.canvas.grid(row=1, column=1, columnspan=5, rowspan=5, sticky=W+N)
        self.canvas.create_rectangle(0, 0, self.CANVAS_WIDTH, self.CANVAS_HEIGHT, fill="white")
        self.canvas.bind("<Button-1>", self.refresh)

    """
    Called whenever screen is refreshed.
    """
    def refresh(self, control_points, m, curve_type=1, curve_closed=False, plane=0):
        self.canvas.create_rectangle(-self.CANVAS_WIDTH, -self.CANVAS_HEIGHT, self.CANVAS_WIDTH, self.CANVAS_HEIGHT, fill="white")

        if curve_type == 1:
            self.drawControlPolygon(control_points, closed=curve_closed)
            self.drawBezierDirect(control_points, m, curve_closed)
        elif curve_type == 2:
            self.drawControlPolygon(control_points, closed=curve_closed)
            self.drawBezierSubdiv(control_points, m, curve_closed)
        elif curve_type == 3:
            self.drawControlPolygon(control_points, closed=curve_closed)
            self.drawBSplineDirect(control_points, m, curve_closed)
        elif curve_type == 4:
            self.drawControlPolygon(control_points, closed=curve_closed)
            self.drawBSplineSubdiv(control_points, m, curve_closed)
        elif curve_type == 5:
            # points for reconstruction
            self.drawControlPoints(control_points)
        if plane == 0:
            self.canvas.create_text(10, 20, text="x", fill="red")
            self.canvas.create_text(20, 10, text="y", fill="green")
            self.canvas.create_line(0, 0, self.CANVAS_WIDTH, 0, width=3, fill="green")
            self.canvas.create_line(0, 0, 0, self.CANVAS_HEIGHT, width=3, fill="red")
        elif plane == 1:
            self.canvas.create_text(10, 20, text="z", fill="red")
            self.canvas.create_text(20, 10, text="y", fill="green")
            self.canvas.create_line(0, 0, self.CANVAS_WIDTH, 0, width=3, fill="green")
            self.canvas.create_line(0, 0, 0, self.CANVAS_HEIGHT, width=3, fill="red")
        if len(control_points) > 2:
            self.drawVoronoiVerts(control_points)

    def drawControlPoints(self, control_points):
        for i, point in enumerate(control_points):
            self.drawPoint(point)

    def drawVoronoiVerts(self, control_points):
        verts = GetVoronoiVerts(control_points)
        for point in verts:
            self.drawPoint(point, color="yellow")

    def drawControlPolygon(self, control_points, show_points=True, color="black", closed=False):
        for i, point in enumerate(control_points):
            if (show_points == True):
                self.drawPoint(point)
            if i > 0:
                self.connectPoints(control_points[i-1], control_points[i], color=color)
        if closed and len(control_points) > 1:
            self.connectPoints(control_points[-1], control_points[0], color=color)

    def drawBezierDirect(self, control_points, m, curve_closed):
        points = []
        if len(control_points) > 1:
            points = Bezier.GetPointsDirectly(control_points, m*len(control_points), closed=curve_closed)
            self.drawBezierSegments(points, color="green")

    def drawBezierSubdiv(self, control_points, m, curve_closed):
        segments = []
        if len(control_points) > 1:
            segments += Bezier.Subdivide(control_points, m, 0.5, closed=curve_closed)
            self.drawBezierSegments(segments, color="red")

    def drawBezierSegments(self, segments, color="black"):
        for i, point in enumerate(segments[:-1]):
            self.canvas.create_line(point.x, point.y, segments[i+1].x, segments[i+1].y, fill=color)

    def drawBSplineDirect(self, control_points, m, curve_closed):
        points = []
        if len(control_points) > 1:
            points = BSpline.Cubic.GetPointsDirectly(control_points, m*len(control_points), closed=curve_closed)
            self.drawBezierSegments(points, color="blue")

    def drawBSplineSubdiv(self, control_points, m, curve_closed):
        """
        points = []
        if len(control_points) > 0:
            points = BSpline.Quadric.GetPointsDirectly(control_points, 10)
            self.drawBezierSegments(points, color="purple")
        """
        points = []
        if len(control_points) > 0:
            points = BSpline.Quadric.Subdivide(control_points, m, 0.5, closed=curve_closed)
            self.drawBezierSegments(points, color="purple")

    def drawPoint(self, event, color="black"):
        self.canvas.create_oval(event.x-1, event.y-1, event.x+1, event.y+1, fill=color, outline=color)

    def connectPoints(self, point1, point2, color):
        self.canvas.create_line(point1.x, point1.y, point2.x, point2.y, fill=color) 

class Controller:
    def __init__(self, root):
        self.model = Model()
        self.curve_var = IntVar()
        self.surface_var = IntVar()
        self.curve_type_var = IntVar()
        self.axis_var = IntVar()
        self.subdivision_var = StringVar()
        self.subdivision_var.set("1")
        self.slice_var = StringVar()
        self.slice_var.set("1")
        self.smooth_var = IntVar()
        self.smooth_var.set(0)
        self.active_curve = 0
        self.u_var = StringVar()
        self.v_var = StringVar()
        self.u_var.set("1")
        self.v_var.set("1")
        self.u_closed_var = IntVar()
        self.v_closed_var = IntVar()
        self.view1 = View(root, self.curve_var, self.surface_var, self.curve_type_var, self.axis_var, self.subdivision_var, self.slice_var, self.smooth_var, self.u_var, self.v_var, self.u_closed_var, self.v_closed_var)
        self.view1.b_clear.config(command=self.clearPoints)
        self.view1.canvas.bind("<Button-1>", self.addPoint)
        self.view1.canvas.bind("<Button-3>", self.onRightClick)
        self.view1.canvas.bind("<Motion>", self.movePoint)
        self.view1.canvas.bind("<d>", self.deletePoint)
        self.view1.canvas.bind("<c>", self.closeCurve)
        self.view1.b_smooth.config(command=self.generateSmooth)
        self.view1.b_nncrust.config(command=self.generateNNReconstruction)
        self.view1.b_crust.config(command=self.generateReconstruction)
        #self.view1.slider.bind("<ButtonRelease-1>", self.changeM)
        self.curve_var.trace('w', self.changeRadio)
        self.curve_type_var.trace('w', self.changeCurve)
        self.subdivision_var.trace('w', self. changeSubdivision)
        self.view1.b_generate.config(command=self.generateMesh)
        self.view1.b_load_file.config(command=self.loadFile)
        self.view1.refresh([], self.curve_var)

    def addPoint(self, event):
        point1 = self.model.addPoint(ControlPoint(event.x, event.y))
        lst = self.model.getControlPoints()
        self.view1.refresh(lst, self.model.active_curve.m, curve_type=self.curve_var.get(), curve_closed=self.model.active_curve.curve_closed, plane=self.active_curve)

    def onRightClick(self, event):
        self.view1.canvas.focus_set()

        if self.model.active_curve.point_is_moving:
            self.dropPoint(event)
        else:
            self.pickupPoint(event)

    def pickupPoint(self, event):
        cp_lst = self.model.getControlPoints()
        for point in cp_lst:
            if math.fabs(point.x-event.x) <= 3 and math.fabs(point.y-event.y) <= 3:
                self.model.active_curve.point_is_moving = True
                self.model.active_curve.moving_point = point
                break

    def dropPoint(self, event):
        self.model.active_curve.point_is_moving = False

    def removePoint(self, x, y):
        self.model.removePoint(ControlPoint(x,y))

    def clearPoints(self):
        self.model.clearPoints()
        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), plane=self.active_curve)
        self.view1.drawControlPolygon(self.model.getControlPoints())

    def changeM(self, event):
        self.model.setM(self.view1.slider.get())
        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

    def changeRadio(self, *args):
        self.model.active_curve.curve_type = self.curve_var.get()
        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

    def changeSubdivision(self, *args):
        try:
            new_m = int(self.subdivision_var.get())
            if  new_m > 10:
                messagebox.showinfo("Value Error", "For performance reasons, please do not use values greater than 10.")
                new_m = 10
            if new_m <= 0:
                messagebox.showinfo("Value Error", "Subdivision value must be greater than 0.")
                new_m = 1
            self.model.active_curve.m = new_m
            self.subdivision_var.set(str(new_m))
        except ValueError:
            if self.subdivision_var.get() != "":
                messagebox.showinfo("Value Error", "That is not a valid subdivision value.")
                self.subdivision_var.set("")
        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

    def movePoint(self, event):
        if self.model.active_curve.point_is_moving:
            self.model.active_curve.moving_point.x = event.x
            self.model.active_curve.moving_point.y = event.y
            self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

    def deletePoint(self, event):
        cp_lst = self.model.getControlPoints()
        for point in cp_lst:
            if math.fabs(point.x-event.x) <= 3 and math.fabs(point.y-event.y) <= 3:
                self.model.getControlPoints().remove(point)
                break
        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

    def generateMesh(self):
        control_points = self.model.getControlPoints()

        # Save points on curve.
        if self.curve_var.get() == 1:
            self.model.setCurvePoints(Bezier.GetPointsDirectly(control_points,
                                                               int(self.subdivision_var.get())*len(control_points), 
                                                               closed=self.model.getCurveClosed()))
        elif self.curve_var.get() == 2:
            self.model.setCurvePoints(Bezier.Subdivide(control_points, 
                                                       int(self.subdivision_var.get()), 0.5, 
                                                       closed=self.model.getCurveClosed()))
        elif self.curve_var.get() == 3:
            self.model.setCurvePoints(BSpline.Cubic.GetPointsDirectly(control_points, 
                                                                      int(self.subdivision_var.get())*len(control_points), 
                                                                      closed=self.model.getCurveClosed()))
        elif self.curve_var.get() == 4:
            points = BSpline.Quadric.Subdivide(control_points,
                                               int(self.subdivision_var.get()),
                                               0.5,
                                               closed=self.model.getCurveClosed())
            print("POINTS {0}".format(len(points)))
            self.model.setCurvePoints(points)

        # Generate mesh from saved points.
        self.model.generateMesh(self.surface_var.get(), slices=self.slice_var.get())

    def generateSmooth(self):
        self.model.generateSmooth(self.smooth_var.get(), self.u_var.get(), self.v_var.get(), self.u_closed_var.get(), self.v_closed_var.get())

    def closeCurve(self, event):
        self.model.setCurveClosed(not self.model.getCurveClosed())
        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

    def changeCurve(self, *args):
        if self.curve_type_var.get() == 0:
            self.model.active_curve = self.model.surface_curve
            self.active_curve = 0
        else:
            self.model.active_curve = self.model.sweep_curve
            self.active_curve = 1

        # Reset all GUI widgets to match curve values.
        self.curve_var.set(self.model.active_curve.curve_type)
        self.subdivision_var.set(str(self.model.getM()))

        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

    def loadFile(self):
        filename = filedialog.askopenfilename()
        self.model.loadMesh(filename)

    def generateReconstruction(self):
        ordered_points = Crust(self.model.getControlPoints())
        self.model.setControlPoints(ordered_points)
        self.model.setCurveClosed(True)
        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

    def generateNNReconstruction(self):
        ordered_points = NNCrust(self.model.getControlPoints())
        self.model.setControlPoints(ordered_points)
        self.model.setCurveClosed(True)
        self.view1.refresh(self.model.getControlPoints(), self.model.getM(), curve_type=self.curve_var.get(), curve_closed=self.model.getCurveClosed(), plane=self.active_curve)

if __name__ == '__main__':
    root = Tk()
    root.withdraw()
    app = Controller(root)
    root.mainloop()
