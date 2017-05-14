from point import ControlPoint3D

class Vec2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dot(self, b):
        return self.x*b.x + self.y*b.y

class GeomVert:
    def __init__(self, x, y, z):
        self.mCo = [0]*3
        self.mCo[0] = x
        self.mCo[1] = y
        self.mCo[2] = z

    def GetCo(self, axis):
        return self.mCo[axis]

    def __eq__(self, geom_vert):
        if type(self) == type(geom_vert):
            return (self.GetCo(0) == geom_vert.GetCo(0) and
                    self.GetCo(1) == geom_vert.GetCo(1) and
                    self.GetCo(2) == geom_vert.GetCo(2))
        else:
            return False

class TopoVert:
    def __init__(self, generator=None):
        self.mIncVerts = set()
        self.mIncEdges = []
        self.mIncFacets = []
        self.vert_point = None
        self.generator = generator #If gen from existing vert, store here.

    def __del__(self):
        self.mIncVerts.clear()
        del self.mIncEdges[:]
        del self.mIncFacets[:]

    def AddIncVert(self, vert_ind):
        self.mIncVerts.add(vert_ind)

    def AddIncEdge(self, edge_ind):
        self.mIncEdges.append(edge_ind)

    def AddIncFacet(self, facet_ind):
        self.mIncFacets.append(facet_ind)

    def GetNumberIncVertices(self):
        return len(self.mIncVerts)

    def GetIncVertex(self, vert_ind):
        sit = iter(self.mIncVerts)
        cur = next(sit)
        for i in range(0, vert_ind):
            cur = next(sit)
        return cur

    def GetNumberIncEdges(self):
        return len(self.mIncEdges)

    def GetIncEdge(self, edge_ind):
        return self.mIncEdges[edge_ind]

    def GetNumberIncFacets(self):
        return len(self.mIncFacets)

    def GetIncFacet(self, facet_ind):
        return self.mIncFacets[facet_ind]

class TopoEdge:
    def __init__(self):
        self.v1 = -1
        self.v2 = -1
        self.mIncFacets = []
        self.edge_point = None

    def GetVertex(self, ind):
        if ind == 0:
            return self.v1
        return self.v2

    def SetVertex(self, ind, v):
        if ind == 0:
            self.v1 = v
        else:
            self.v2 = v

    def __eq__(self, topo_edge):
        return (((self.v1 == topo_edge.GetVertex(0)) and
                 (self.v2 == topo_edge.GetVertex(1))) or
                ((self.v2 == topo_edge.GetVertex(0)) and
                 (self.v1 == topo_edge.GetVertex(1))))

    def AddIncFacet(self, facet_ind):
        self.mIncFacets.append(facet_ind)

    def GetNumberIncFacets(self):
        return len(self.mIncFacets)

    def GetIncFacet(self, facet_ind):
        return self.mIncFacets[facet_ind]

    def __del__(self):
        del self.mIncFacets[:]

class TopoFacet:
    def __init__(self):
        self.mIncVerts = []
        self.mIncEdges = []
        self.mIncFacets = set()
        self.face_point = None

    def __del__(self):
        del self.mIncVerts[:]
        del self.mIncEdges[:]
        self.mIncFacets.clear()

    def AddIncVertex(self, v_ind):
        self.mIncVerts.append(v_ind)

    def AddIncEdge(self, e_ind):
        self.mIncEdges.append(e_ind)

    def AddIncFacet(self, f_ind):
        self.mIncFacets.add(f_ind)

    def GetNumberVertices(self):
        return len(self.mIncVerts)

    def GetVertexInd(self, vert_ind):
        return self.mIncVerts[vert_ind]

    def FindVertex(self, v):
        for i in range(0, len(self.mIncVerts)):
            if self.mIncVerts[i] == v:
                return True
        return False

    def GetNumberEdges(self):
        return len(self.mIncEdges)

    def GetIncEdge(self, edge_ind):
        return self.mIncEdges[edge_ind]

    def GetNumberFacets(self):
        return len(self.mIncFacets)

    def GetIncFacet(self, facet_ind):
        sit = iter(self.mIncFacets)
        cur = next(sit)
        for i in range(0, facet_ind):
            cur = next(sit)
        return cur

class Mesh:
    def __init__(self):
        self.mGeomVerts = []
        self.mTopoVerts = []
        self.mTopoEdges = []
        self.mTopoFacets = []
        self.uvMatrix = [[]]

    def __del__(self):
        self.Erase()

    def AddFacetTri(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, gen_lst=None):
        geomfacet = [GeomVert(x1,y1,z1), GeomVert(x2, y2, z2), GeomVert(x3, y3, z3)]
        self.AddFacet(geomfacet, gen_lst=gen_lst)

    def AddFacetQuad(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, direction=None, gen_lst=None):
        geomfacet = [GeomVert(x1,y1,z1), GeomVert(x2, y2, z2), GeomVert(x3, y3, z3), GeomVert(x4, y4, z4)]
        self.AddFacet(geomfacet, direction, gen_lst=gen_lst)
        return self.mTopoFacets[-1]

    def AddFacet(self, geomfacet, direction=None, gen_lst=None):
        topofacet = TopoFacet()
        row_added = False
        for i in range(0, len(geomfacet)):
            v_ind = self.FindGeomVertex(geomfacet[i])
            if v_ind == -1:
                # Add geom vertex
                v_ind = len(self.mGeomVerts)
                self.mGeomVerts.append(geomfacet[i])

                # Add topo vertex
                generator = None
                if gen_lst != None:
                    generator = gen_lst[i]

                topovert = TopoVert(generator=generator)
                self.mTopoVerts.append(topovert)
                
                if direction == 'u':
                    if i == 0 or i == 1:
                        self.uvMatrix[len(self.uvMatrix)-2].append(geomfacet[i])
                    if i == 2 or i == 3:
                        # self.uvMatrix[len(self.uvMatrix)-1] = [geomfacet[i]] + self.uvMatrix[len(self.uvMatrix)-1]
                        self.uvMatrix[len(self.uvMatrix)-1].append(geomfacet[i])

                if direction == 'v':
                    if not row_added:
                        row_added = True
                        # On first loop, add new row to matrix.
                        self.uvMatrix.append([])
                    if i == 0 or i == 1:
                        self.uvMatrix[len(self.uvMatrix)-2].append(geomfacet[i])
                    if i == 2 or i == 3:
                        self.uvMatrix[len(self.uvMatrix)-1] = [geomfacet[i]] + self.uvMatrix[len(self.uvMatrix)-1]
                        #self.uvMatrix[len(self.uvMatrix)-1].append(geomfacet[i])

            topofacet.AddIncVertex(v_ind)

        # Add new topo facet to mesh
        facet_ind = len(self.mTopoFacets)
        self.mTopoFacets.append(topofacet)

        # Add edges of facet to mesh (checking if they already exist)
        for i in range(0, topofacet.GetNumberVertices()):
            prev = topofacet.GetNumberVertices()-1 if i == 0 else i - 1
            
            # Create edge
            e = TopoEdge()
            e.SetVertex(0, topofacet.GetVertexInd(prev))
            e.SetVertex(1, topofacet.GetVertexInd(i))

            # Check if exists
            e_ind = self.FindTopoEdge(e)
            if e_ind == -1:
                # Didn't exist, add to mesh
                e_ind = len(self.mTopoEdges)
                self.mTopoVerts[e.GetVertex(0)].AddIncEdge(e_ind)
                self.mTopoVerts[e.GetVertex(1)].AddIncEdge(e_ind)
                self.mTopoEdges.append(e)

            # Point edge to this facet
            self.mTopoEdges[e_ind].AddIncFacet(facet_ind)
            # Point facet to this edge
            self.mTopoFacets[facet_ind].AddIncEdge(e_ind)

        # Compute other connectivity
        for i in range(0, topofacet.GetNumberVertices()):
            # Add vertex-facet topology
            self.mTopoVerts[topofacet.GetVertexInd(i)].AddIncFacet(facet_ind)
            # Add vertex-vertex topology
            prev = topofacet.GetNumberVertices()-1 if i == 0 else i - 1
            nxt = 0 if i == topofacet.GetNumberVertices()-1 else i + 1

            self.mTopoVerts[topofacet.GetVertexInd(i)].AddIncVert(topofacet.GetVertexInd(prev))
            self.mTopoVerts[topofacet.GetVertexInd(i)].AddIncVert(topofacet.GetVertexInd(nxt))
        
        # Facet-facet adjacency
        for i in range(0, self.mTopoFacets[facet_ind].GetNumberEdges()):
            edge = self.mTopoEdges[self.mTopoFacets[facet_ind].GetIncEdge(i)]
            for j in range(0, edge.GetNumberIncFacets()):
                if edge.GetIncFacet(j) != facet_ind:
                    self.mTopoFacets[facet_ind].AddIncFacet(edge.GetIncFacet(j))
                    self.mTopoFacets[edge.GetIncFacet(j)].AddIncFacet(facet_ind)


    def GetNumberVertices(self):
        return len(self.mGeomVerts)

    def GetNumberEdges(self):
        return len(self.mTopoEdges)

    def GetNumberFacets(self):
        return len(self.mTopoFacets)

    def GetVertex(self, vert_ind):
        return self.mTopoVerts[vert_ind]

    def GetEdge(self, edge_ind):
        return self.mTopoEdges[edge_ind]

    def GetFacet(self, facet_ind):
        return self.mTopoFacets[facet_ind]

    def GetGeomVertex(self, vert_ind):
        return self.mGeomVerts[vert_ind]

    def Erase(self):
        del self.mGeomVerts[:]
        del self.mTopoVerts[:]
        del self.mTopoEdges[:]
        del self.mTopoFacets[:]

    def FindGeomVertex(self, v):
        for i in range(0, len(self.mGeomVerts)):
            if self.mGeomVerts[i] == v:
                return i
        return -1

    def FindTopoEdge(self, e):
        for i in range(0, len(self.mTopoEdges)):
            if self.mTopoEdges[i] == e:
                return i
        return -1
                
    def SelfIntersection(self):
        return False

    def GetPlanarity(self):
        epsilon_avg = 0
        for face in self.mTopoFacets:
            v1 = self.GetGeomVertex(face.GetVertexInd(0))
            v1_cp = ControlPoint3D(v1.GetCo(0), v1.GetCo(1), v1.GetCo(2))
            w1 = self.GetGeomVertex(face.GetVertexInd(2))
            w1_cp = ControlPoint3D(w1.GetCo(0), w1.GetCo(1), w1.GetCo(2))
            v2 = self.GetGeomVertex(face.GetVertexInd(1))
            v2_cp = ControlPoint3D(v2.GetCo(0), v2.GetCo(1), v2.GetCo(2))
            w2 = self.GetGeomVertex(face.GetVertexInd(3))
            w2_cp = ControlPoint3D(w2.GetCo(0), w2.GetCo(1), w2.GetCo(2))
            midpoint1 = (v1_cp.add(w1_cp)).mult(1/2)
            midpoint2 = (v2_cp.add(w2_cp)).mult(1/2)
            dist = midpoint1.dist(midpoint2)
            epsilon_avg += dist
        return (epsilon_avg/self.GetNumberFacets())

def GetProjVerts(mesh, face, axis_0, axis_1):
    num_verts = face.GetNumberVertices()
    verts = []
    for i in range(0, num_verts):
        # Get all vertices in the face.
        v = mesh.GetGeomVertex(face.GetVertexInd(i))
        v_proj = Vec2D(v.GetCo(axis_0), v.GetCo(axis_1))
        verts.append(v_proj)
    return verts

def ConvexPolygonOverlap(a_verts, b_verts):
    if FindSeparatingAxis(a_verts, b_verts):
        return False
    if FindSeparatingAxis(b_verts, a_verts):
        return False
    return True

def FindSeparatingAxis(a_verts, b_verts):
    size_a = len(a_verts)
    prev = size_a - 1
    for i in range(size_a):
        edge = Vec2D(a_verts[i].x - a_verts[prev].x, a_verts[i].y - a_verts[prev].y)
        v = Vec2D(edge.y, -edge.x)
        a_range = GatherPolygonProjectionExtents(a_verts, v)
        b_range = GatherPolygonProjectionExtents(b_verts, v)
        if (a_range[1] <= b_range[0]):
            return True
        if (b_range[1] <= a_range[0]):
            return True
        prev = i
    return False

def GatherPolygonProjectionExtents(vert_list, v):
    lst_min = v.dot(vert_list[0])
    lst_max = v.dot(vert_list[0])
    for i in range(1, len(vert_list)):
        d = v.dot(vert_list[i])
        if (d < lst_min):
            lst_min = d
        if (d > lst_max):
            lst_max = d
    return [lst_min, lst_max]

def Intersect2D(mesh, face_1, face_2, axis_0, axis_1):
    f1_min_x = None
    f1_max_x = None
    f1_min_y = None
    f1_max_y = None
    num_verts_1 = face_1.GetNumberVertices()
    for i in range(0, num_verts_1):
        # Get all vertices in the face.
        v = mesh.GetGeomVertex(face_1.GetVertexInd(i))
        # Project onto plane.
        v_proj = [v.GetCo(axis_0), v.GetCo(axis_1)]
        if f1_min_x == None or f1_min_x > v_proj[0]:
            f1_min_x = v_proj[0]
        if f1_max_x == None or f1_max_x < v_proj[0]:
            f1_max_x = v_proj[0]
        if f1_min_y == None or f1_min_y > v_proj[1]:
            f1_min_y = v_proj[1]
        if f1_max_y == None or f1_max_y < v_proj[1]:
            f1_max_y = v_proj[1]
    f2_min_x = None
    f2_max_x = None
    f2_min_y = None
    f2_max_y = None
    num_verts_2 = face_2.GetNumberVertices()
    for i in range(0, num_verts_2):
        # Get all vertices in the face.
        v = mesh.GetGeomVertex(face_2.GetVertexInd(i))
        # Project onto plane.
        v_proj = [v.GetCo(axis_0), v.GetCo(axis_1)]
        if f2_min_x == None or f2_min_x > v_proj[0]:
            f2_min_x = v_proj[0]
        if f2_max_x == None or f2_max_x < v_proj[0]:
            f2_max_x = v_proj[0]
        if f2_min_y == None or f2_min_y > v_proj[1]:
            f2_min_y = v_proj[1]
        if f2_max_y == None or f2_max_y < v_proj[1]:
            f2_max_y = v_proj[1]
    x_intersects = (f1_min_x < f2_max_x) and (f2_min_x < f1_max_x)
    y_intersects = (f1_min_y < f2_max_y) and (f2_min_y < f1_max_y)
    return x_intersects and y_intersects

class Catmull_Clark:
    @classmethod
    def Subdivide(cls, mesh):
        new_mesh = Mesh()
        new_verts = []
        # Face verts
        for face in mesh.mTopoFacets:
            num_verts = face.GetNumberVertices()
            sum_verts = ControlPoint3D(0, 0, 0)
            for i in range(num_verts):
                vert_ind = face.GetVertexInd(i)
                vert = mesh.GetGeomVertex(vert_ind)
                vert_cp = ControlPoint3D(vert.GetCo(0), vert.GetCo(1), vert.GetCo(2))
                sum_verts = sum_verts.add(vert_cp)
            f_vert = sum_verts.mult(1/num_verts)
            face.face_point = f_vert
            new_verts.append(f_vert)
        # Edge points
        for edge in mesh.mTopoEdges:
            if edge.GetNumberIncFacets() == 2:
                v = edge.GetVertex(0)
                w = edge.GetVertex(1)
                v_f = mesh.mTopoFacets[edge.GetIncFacet(0)].face_point
                w_f = mesh.mTopoFacets[edge.GetIncFacet(1)].face_point
                v_geom = mesh.GetGeomVertex(v)
                w_geom = mesh.GetGeomVertex(w)
                v_geom_cp = ControlPoint3D(v_geom.GetCo(0), v_geom.GetCo(1), v_geom.GetCo(2))
                w_geom_cp = ControlPoint3D(w_geom.GetCo(0), w_geom.GetCo(1), w_geom.GetCo(2))
                sum_verts = v_f.add(w_f.add(v_geom_cp.add(w_geom_cp)))
                e_vert = sum_verts.mult(1/4)
                edge.edge_point = e_vert
                new_verts.append(e_vert)
            if edge.GetNumberIncFacets() == 1: # Boundary edge
                v = edge.GetVertex(0)
                w = edge.GetVertex(1)
                v_f = mesh.mTopoFacets[edge.GetIncFacet(0)].face_point
                v_geom = mesh.GetGeomVertex(v)
                w_geom = mesh.GetGeomVertex(w)
                v_geom_cp = ControlPoint3D(v_geom.GetCo(0), v_geom.GetCo(1), v_geom.GetCo(2))
                w_geom_cp = ControlPoint3D(w_geom.GetCo(0), w_geom.GetCo(1), w_geom.GetCo(2))
                sum_verts = v_f.add(v_geom_cp.add(w_geom_cp))
                e_vert = sum_verts.mult(1/3)
                edge.edge_point = e_vert
                new_verts.append(e_vert)
        # Vertex points
        for i, topo_vert in enumerate(mesh.mTopoVerts):
            geom_vert = mesh.GetGeomVertex(i)
            n = topo_vert.GetNumberIncEdges()
            num_facets = topo_vert.GetNumberIncFacets()
            sum_face_points = ControlPoint3D(0, 0, 0)
            for i in range(num_facets):
                face = topo_vert.GetIncFacet(i)
                sum_face_points = sum_face_points.add(mesh.GetFacet(face).face_point)
            Q = sum_face_points.mult(1/num_facets)
            sum_edge_points = ControlPoint3D(0, 0, 0)
            for i in range(n):
                edge = mesh.GetEdge(topo_vert.GetIncEdge(i))
                edge_vert_v = mesh.GetGeomVertex(edge.GetVertex(0))
                edge_vert_w = mesh.GetGeomVertex(edge.GetVertex(1))
                v_cp = ControlPoint3D(edge_vert_v.GetCo(0), edge_vert_v.GetCo(1), edge_vert_v.GetCo(2))
                w_cp = ControlPoint3D(edge_vert_w.GetCo(0), edge_vert_w.GetCo(1), edge_vert_w.GetCo(2))
                midpoint = (v_cp.add(w_cp)).mult(1/2)
                sum_edge_points = sum_edge_points.add(midpoint)
            R = sum_edge_points.mult(1/n)
            geom_vert_cp = ControlPoint3D(geom_vert.GetCo(0), geom_vert.GetCo(1), geom_vert.GetCo(2))
            v_vert = ((Q.add(R.mult(2)).add(geom_vert_cp.mult(n-3)))).mult(1/n)
            topo_vert.vert_point = v_vert
            new_verts.append(v_vert)
        # Connect the points
        for face in mesh.mTopoFacets:
            pt_1 = face.face_point
            for i, edge in enumerate(face.mIncEdges):
                j = i + 1
                if j == len(face.mIncEdges):
                    j = 0
                pt_2 = mesh.GetEdge(edge).edge_point
                pt_3 = None
                if (mesh.GetEdge(face.mIncEdges[i]).GetVertex(1) == mesh.GetEdge(face.mIncEdges[j]).GetVertex(0) or 
                    mesh.GetEdge(face.mIncEdges[i]).GetVertex(1) == mesh.GetEdge(face.mIncEdges[j]).GetVertex(1)):
                    pt_3 = mesh.GetVertex(mesh.GetEdge(face.mIncEdges[i]).GetVertex(1)).vert_point
                if (mesh.GetEdge(face.mIncEdges[i]).GetVertex(0) == mesh.GetEdge(face.mIncEdges[j]).GetVertex(1) or 
                    mesh.GetEdge(face.mIncEdges[i]).GetVertex(0) == mesh.GetEdge(face.mIncEdges[j]).GetVertex(0)):
                    pt_3 = mesh.GetVertex(mesh.GetEdge(face.mIncEdges[i]).GetVertex(0)).vert_point
                pt_4 = mesh.GetEdge(face.mIncEdges[j]).edge_point
                new_mesh.AddFacetQuad(pt_1.x, pt_1.y, pt_1.z,
                                      pt_2.x, pt_2.y, pt_2.z,
                                      pt_3.x, pt_3.y, pt_3.z,
                                      pt_4.x, pt_4.y, pt_4.z)
        return new_mesh

class Loop:
    @classmethod
    def Subdivide(cls, mesh):
        new_mesh = Mesh()
        alpha_n = (5/8)
        for edge in mesh.mTopoEdges:
            if edge.GetNumberIncFacets() == 2:
                f1 = mesh.GetFacet(edge.GetIncFacet(0))
                f2 = mesh.GetFacet(edge.GetIncFacet(1))
                r = mesh.GetGeomVertex(edge.GetVertex(0))
                r_ind = edge.GetVertex(0)
                s = mesh.GetGeomVertex(edge.GetVertex(1))
                s_ind = mesh.GetVertex(1)
                p = q = None
                for vert in range(3):
                    if f1.GetVertexInd(vert) != r and f1.GetVertexInd(vert) != s:
                        p = mesh.GetGeomVertex(f1.GetVertexInd(vert))
                    if f2.GetVertexInd(vert) != r and f2.GetVertexInd(vert) != s:
                        q = mesh.GetGeomVertex(f2.GetVertexInd(vert))
                r_cp = ControlPoint3D(r.GetCo(0), r.GetCo(1), r.GetCo(2))
                s_cp = ControlPoint3D(s.GetCo(0), s.GetCo(1), s.GetCo(2))
                p_cp = ControlPoint3D(p.GetCo(0), p.GetCo(1), p.GetCo(2))
                q_cp = ControlPoint3D(q.GetCo(0), q.GetCo(1), q.GetCo(2))
                e_point = p_cp.mult(1/8).add(r_cp.mult(3/8)).add(s_cp.mult(3/8)).add(q_cp.mult(1/8))
                edge.edge_point = e_point
            if edge.GetNumberIncFacets() == 1:
                r = mesh.GetGeomVertex(edge.GetVertex(0))
                s = mesh.GetGeomVertex(edge.GetVertex(1))
                r_cp = ControlPoint3D(r.GetCo(0), r.GetCo(1), r.GetCo(2))
                s_cp = ControlPoint3D(s.GetCo(0), s.GetCo(1), s.GetCo(2))
                e_point = (r_cp.add(s_cp)).mult(1/2)
                edge.edge_point = e_point
        num_verts = len(mesh.mTopoVerts)
        for i in range(num_verts):
            vert_geom = mesh.GetGeomVertex(i)
            vert_topo = mesh.GetVertex(i)
            num_edges = vert_topo.GetNumberIncEdges()
            sum_p = ControlPoint3D(0,0,0)
            for i in range(num_edges):
                edge = mesh.GetEdge(vert_topo.GetIncEdge(i))
                v = mesh.GetVertex(edge.GetVertex(0))
                w = mesh.GetVertex(edge.GetVertex(1))
                v_geom = mesh.GetGeomVertex(edge.GetVertex(0))
                w_geom = mesh.GetGeomVertex(edge.GetVertex(1))
                p = None
                if v == vert_topo:
                    p = w_geom
                else:
                    p = v_geom
                p = ControlPoint3D(p.GetCo(0), p.GetCo(1), p.GetCo(2))
                sum_p = sum_p.add(p)
            vert_geom_cp = ControlPoint3D(vert_geom.GetCo(0), vert_geom.GetCo(1), vert_geom.GetCo(2))
            v_point = (sum_p.mult(1/num_edges).mult(1-alpha_n)).add(vert_geom_cp.mult(alpha_n))
            vert_topo.vertex_point = v_point

        for face in mesh.mTopoFacets:
            edge_pts = []
            for i, edge in enumerate(face.mIncEdges):
                j = i + 1
                if j == len(face.mIncEdges):
                    j = 0
                edge = mesh.GetEdge(edge)
                next_edge = mesh.GetEdge(face.mIncEdges[j])
                edge_shared_vert_ind = edge.GetVertex(1)
                if edge.GetVertex(0) == next_edge.GetVertex(0) or edge.GetVertex(0) == next_edge.GetVertex(1): 
                    edge_shared_vert_ind = edge.GetVertex(0)
                edge_shared_vert = mesh.GetVertex(edge_shared_vert_ind)
                new_mesh.AddFacetTri(edge.edge_point.x, edge.edge_point.y, edge.edge_point.z,
                                     edge_shared_vert.vertex_point.x, edge_shared_vert.vertex_point.y, edge_shared_vert.vertex_point.z, 
                                     next_edge.edge_point.x, next_edge.edge_point.y, next_edge.edge_point.z)
                edge_pts.append(edge.edge_point)
            new_mesh.AddFacetTri(edge_pts[0].x, edge_pts[0].y, edge_pts[0].z,
                                 edge_pts[1].x, edge_pts[1].y, edge_pts[1].z,
                                 edge_pts[2].x, edge_pts[2].y, edge_pts[2].z)
        return new_mesh

class Doo_Sabin:
    def __init__(self, mesh):
        self.Subdivide(mesh)

    @classmethod
    def Subdivide(cls, mesh):
        new_mesh = Mesh()
        f_faces = {}
        for face in mesh.mTopoFacets:
            avg_vert_lst = []
            gen_lst = []
            for vert in face.mIncVerts:
                geom_vert = mesh.GetGeomVertex(vert)
                avg_vert = ControlPoint3D(geom_vert.GetCo(0), geom_vert.GetCo(1), geom_vert.GetCo(2))
                topo_vert = mesh.GetVertex(vert)
                num_inc = topo_vert.GetNumberIncVertices()
                num_inc_in_face = 0
                gen_lst.append(vert)
                for i in range(num_inc):
                    inc_vert = topo_vert.GetIncVertex(i)
                    if face.FindVertex(inc_vert) != False:
                        num_inc_in_face += 1
                        inc_geom_vert = mesh.GetGeomVertex(inc_vert)
                        avg_vert = avg_vert.add(ControlPoint3D(inc_geom_vert.GetCo(0),
                                                               inc_geom_vert.GetCo(1),
                                                               inc_geom_vert.GetCo(2)))
                avg_vert = avg_vert.mult(1/(num_inc_in_face+1))
                avg_vert_lst.append(avg_vert)
            if len(avg_vert_lst) == 4:
                new_mesh.AddFacetQuad(avg_vert_lst[0].x, avg_vert_lst[0].y, avg_vert_lst[0].z,
                                      avg_vert_lst[1].x, avg_vert_lst[1].y, avg_vert_lst[1].z,
                                      avg_vert_lst[2].x, avg_vert_lst[2].y, avg_vert_lst[2].z,
                                      avg_vert_lst[3].x, avg_vert_lst[3].y, avg_vert_lst[3].z,
                                      gen_lst=gen_lst)
            elif len(avg_vert_lst) == 3:
                new_mesh.AddFacetTri(avg_vert_lst[0].x, avg_vert_lst[0].y, avg_vert_lst[0].z,
                                     avg_vert_lst[1].x, avg_vert_lst[1].y, avg_vert_lst[1].z,
                                     avg_vert_lst[2].x, avg_vert_lst[2].y, avg_vert_lst[2].z,
                                     gen_lst=gen_lst)
            f_faces[face] = new_mesh.mTopoFacets[-1]
        # E-Faces
        edge_faces = {}
        for edge in mesh.mTopoEdges:
            if edge.GetNumberIncFacets() == 2:
                v = edge.GetVertex(0)
                w = edge.GetVertex(1)
                # Get new v1, v2
                face = mesh.GetFacet(edge.GetIncFacet(0))
                f_face = f_faces.get(face)
                ff_v1_topo = ff_w1_topo = ff_v1_geom = ff_w1_geom = None
                if(f_face != None):
                    # Check if edge has vertex corresponding to v1.
                    num_verts = f_face.GetNumberVertices()
                    for vert in f_face.mIncVerts:
                        vert_topo = new_mesh.GetVertex(vert)
                        if vert_topo.generator == v:
                            ff_v1_topo = vert_topo
                            ff_v1_geom = new_mesh.GetGeomVertex(vert)
                        elif vert_topo.generator == w:
                            ff_w1_topo = vert_topo
                            ff_w1_geom = new_mesh.GetGeomVertex(vert)

                # Get new v2, w2
                face = mesh.GetFacet(edge.GetIncFacet(1))
                f_face = f_faces.get(face)
                ff_v2_topo = ff_w2_topo = ff_v2_geom = ff_w2_geom = None
                if(f_face != None):
                    # Check if edge has vertex corresponding to v1.
                    num_verts = f_face.GetNumberVertices()
                    for vert in f_face.mIncVerts:
                        vert_topo = new_mesh.GetVertex(vert)
                        if vert_topo.generator == v:
                            ff_v2_topo = vert_topo
                            ff_v2_geom = new_mesh.GetGeomVertex(vert)
                        elif vert_topo.generator == w:
                            ff_w2_topo = vert_topo
                            ff_w2_geom = new_mesh.GetGeomVertex(vert)

                new_mesh.AddFacetQuad(ff_v2_geom.GetCo(0), ff_v2_geom.GetCo(1), ff_v2_geom.GetCo(2), 
                                      ff_w2_geom.GetCo(0), ff_w2_geom.GetCo(1), ff_w2_geom.GetCo(2),  
                                      ff_w1_geom.GetCo(0), ff_w1_geom.GetCo(1), ff_w1_geom.GetCo(2),
                                      ff_v1_geom.GetCo(0), ff_v1_geom.GetCo(1), ff_v1_geom.GetCo(2))
        # For each vert in the new mesh
        for j, new_vert_topo in enumerate(new_mesh.mTopoVerts):
            fill_verts = []
            prev_vert = new_vert_topo
            next_vert = new_vert_topo
            fill_verts.append(prev_vert)
            if prev_vert.GetNumberIncVertices() == 4 and prev_vert.GetNumberIncFacets() == 3:
                while(len(fill_verts) < 4):
                    prev_vert = next_vert
                    # Has a hole to fill.
                    num_edges = prev_vert.GetNumberIncEdges()
                    for edge_idx in range(num_edges):
                        topo_edge = new_mesh.GetEdge(prev_vert.GetIncEdge(edge_idx))
                        if topo_edge.GetNumberIncFacets() == 1:
                            edge_vert_one = new_mesh.GetVertex(topo_edge.GetVertex(0))
                            edge_vert_two = new_mesh.GetVertex(topo_edge.GetVertex(1))
                            if edge_vert_one == prev_vert:
                                next_vert = edge_vert_two
                            else:
                                next_vert = edge_vert_one
                            if next_vert not in fill_verts:
                                fill_verts.append(next_vert)
                                break
            if len(fill_verts) == 4:
                print(fill_verts)
                v1 = new_mesh.GetGeomVertex(new_mesh.mTopoVerts.index(fill_verts[0]))
                v2 = new_mesh.GetGeomVertex(new_mesh.mTopoVerts.index(fill_verts[1]))
                v3 = new_mesh.GetGeomVertex(new_mesh.mTopoVerts.index(fill_verts[2]))
                v4 = new_mesh.GetGeomVertex(new_mesh.mTopoVerts.index(fill_verts[3]))
                new_mesh.AddFacetQuad(v1.GetCo(0), v1.GetCo(1), v1.GetCo(2),
                                      v2.GetCo(0), v2.GetCo(1), v2.GetCo(2),
                                      v3.GetCo(0), v3.GetCo(1), v3.GetCo(2),
                                      v4.GetCo(0), v4.GetCo(1), v4.GetCo(2))
                print("Added facet")

        """
        for j,old_vert_topo in enumerate(mesh.mTopoVerts):
            for i,vert_topo in enumerate(new_mesh.mTopoVerts):
                # For each vert in the f-face corresponding to it
                '''
                num_inc_facets = vert_topo.GetNumberIncFacets()
                inc_facets
                for i in range(num_inc_facets):
                '''
                if vert_topo.GetNumberIncEdges() > vert_topo.GetNumberIncFacets() and vert_topo.GetNumberIncEdges() > 3 and vert_topo.generator == j: # only try adding a v-face if there is place for it.
                    vert_geom = new_mesh.GetGeomVertex(i)
                    v_face = [[vert_topo, vert_geom]]
                    for vert_ind,new_vert_topo in enumerate(new_mesh.mTopoVerts):
                        # For each vert in the new mesh, see if it was also generated by the same original vertex.
                        if new_vert_topo != vert_topo and vert_topo.generator == new_vert_topo.generator:
                            new_vert_geom = new_mesh.GetGeomVertex(vert_ind)
                            # If so, add it to the list.
                            v_face.append([new_vert_topo,new_vert_geom])
                    # Order verts
                    ordered_verts = cls.OrderVerts(v_face, new_mesh)

                    if len(ordered_verts) == 3:
                        new_mesh.AddFacetTri(ordered_verts[0][1].GetCo(0), ordered_verts[0][1].GetCo(1), ordered_verts[0][1].GetCo(2),
                                             ordered_verts[1][1].GetCo(0), ordered_verts[1][1].GetCo(1), ordered_verts[1][1].GetCo(2),
                                             ordered_verts[2][1].GetCo(0), ordered_verts[2][1].GetCo(1), ordered_verts[2][1].GetCo(2))
                    if len(ordered_verts) == 4:
                        new_mesh.AddFacetQuad(ordered_verts[0][1].GetCo(0), ordered_verts[0][1].GetCo(1), ordered_verts[0][1].GetCo(2),
                                              ordered_verts[1][1].GetCo(0), ordered_verts[1][1].GetCo(1), ordered_verts[1][1].GetCo(2),
                                              ordered_verts[2][1].GetCo(0), ordered_verts[2][1].GetCo(1), ordered_verts[2][1].GetCo(2),
                                              ordered_verts[3][1].GetCo(0), ordered_verts[3][1].GetCo(1), ordered_verts[3][1].GetCo(2))
                                              """
        return new_mesh

    @classmethod
    def OrderVerts(cls, v_face, mesh, ordered_verts=[]):
        if len(ordered_verts) == 0:
            ordered_verts.append(v_face[0])
            return cls.OrderVerts(v_face, mesh, ordered_verts = ordered_verts) 
        elif len(ordered_verts) < len(v_face):
            elem = ordered_verts[-1]
            num_inc_verts = elem[0].GetNumberIncVertices()
            for inc_vert_ind in range(num_inc_verts):
                next_elem = [mesh.GetVertex(elem[0].GetIncVertex(inc_vert_ind)), mesh.GetGeomVertex(elem[0].GetIncVertex(inc_vert_ind))] 
                if next_elem in v_face and next_elem != elem:
                    ordered_verts.append(next_elem)
                    return cls.OrderVerts(v_face, mesh, ordered_verts = ordered_verts) 
        return ordered_verts
