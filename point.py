import math
class ControlPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, point2):
        return ControlPoint(self.x + point2.x, self.y + point2.y)

    def sub(self, point2):
        return ControlPoint(self.x - point2.x, self.y - point2.y)

    def mult(self, u):
        return ControlPoint(self.x*u, self.y*u)

class ControlPoint3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def add(self, point2):
        return ControlPoint3D(self.x + point2.x, self.y + point2.y, self.z + point2.z)

    def sub(self, point2):
        return ControlPoint3D(self.x - point2.x, self.y - point2.y, self.z - point2.z)

    def mult(self, u):
        return ControlPoint3D(self.x*u, self.y*u, self.z*u)

    def dist(self, point2):
        return math.sqrt((self.x - point2.x)*(self.x - point2.x) +
                         (self.y - point2.y)*(self.y - point2.y) +
                         (self.z - point2.z)*(self.z - point2.z))
