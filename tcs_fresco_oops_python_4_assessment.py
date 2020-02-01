import math 
class Point:
    def __init__(self , x,y , z):
        self.x=x
        self.y=y
        self.z=z
    def __str__(self):
        return 'point : ('+str(self.x)+', '+str(self.y)+', '+str(self.z)+')'
    def distance(self,p2):
        return math.sqrt((self.x-p2.x)**2+(self.y-p2.y)**2+(self.z-p2.z)**2)
    def __add__(self,p2):
        x1=self.x+p2.x
        y1=self.y+p2.y
        z1=self.z+p2.z
        return Point(x1,y1,z1)
p1=Point(2,1,4)
print(p1)
p2=Point(4,3,5)
p3=Point(-2,-1,4)
dist=p2.distance(p3)
print(dist)
print(p2+p3)
