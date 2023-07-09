import numpy as np
import numpy.linalg as la
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def is_counter_clockwise(triangle):
    A = triangle[1:,:] - triangle[0,:]
    return la.det(A) > 0.

def is_inside_circumcircle(point, triangle):
    A = triangle - point
    A = np.hstack([A, A[:,0:1]**2 + A[:,1:2]**2])
    return la.det(A) > 0.

def circumcircle(triangle):
    return patches.Circle

class Triangulation:

    def __init__(self, points):
        self.points          = np.asfarray(points)
        self.current_point   = 0
        self.n_points        = len(points)
        self.triangulation   = []
        self.bad_triangles   = [] 
        self.polygon         = []
        self.super_points    = []
        self.super_triangles = []
        self.complete        = False

    def create_super_triangles(self):
        x_min, y_min = self.points.min(0) - 0.2
        x_max, y_max = self.points.max(0) + 0.2

        self.super_points = np.asfarray([[x_min, y_min], 
                                         [x_max, y_min],
                                         [x_max, y_max],
                                         [x_min, y_max]])
        self.points       = np.vstack((self.super_points, self.points))

        self.super_triangles = [[0, 1, 2], [2, 3, 0]]
        self.triangulation   = [[0, 1, 2], [2, 3, 0]]
        self.current_point   = self.current_point + 4
        return
    
    def select_next_point(self):
        if self.current_point >= self.n_points + 3:
            self.complete = True
            return

        self.current_point += 1
        return
    
    def get_bad_triangles(self):
        p = self.current_point
        self.bad_triangles = []
        for i, triangle in enumerate(self.triangulation):
            if is_inside_circumcircle( self.points[p], self.points[triangle]):
                self.bad_triangles.append(i)
        return
    
    def merge_bad_triangles(self):
        polygon = []
        for i in self.bad_triangles:
            bad_triangle = self.triangulation[i]
            
            for v in range(3):
                vert1, vert2 = bad_triangle[v], bad_triangle[(v+1) % 3]

                shared_edge = False
                for j in self.bad_triangles:
                    if i == j:
                        continue

                    if vert1 in self.triangulation[j] and vert2 in self.triangulation[j]:
                        shared_edge = True
                        break

                if not shared_edge:
                    polygon.append([vert1, vert2])
        
        self.polygon = polygon
        return
    
    def remove_bad_triangles(self):
        self.triangulation = [triangle for i, triangle in enumerate(self.triangulation) if i not in self.bad_triangles]
        return
    
    def triangulate_bad_polygon(self):
        p = self.current_point
        for edge in self.polygon:
            triangle = [*edge, p]
            if not is_counter_clockwise(self.points[triangle]):
                triangle = triangle[::-1]
            self.triangulation.append(triangle)
        return
    
    def remove_super_triangles(self):
        self.points   = self.points[4:]
        triangulation = []
        for triangle in self.triangulation:
            if any( map(lambda v: v < 4, triangle) ):
                continue

            triangulation.append( list(map(lambda v: v - 4, triangle)) )

        self.triangulation = triangulation
        return
    
def animate_triangle_draw(triangle, start, stop, frame, ax):
    total_frames    = stop - start
    frames_per_edge = total_frames // 3
    elapsed_frames  = frame - start

    if elapsed_frames < 0:
        return
    
    if elapsed_frames > total_frames:
        ax.fill(triangle[:,0], triangle[:,1], fc = 'none', ec = 'black', lw = 1)
        return
    
    t = 3 * elapsed_frames / total_frames
    x0, y0 = triangle[0]
    x1, y1 = triangle[1]
    x2, y2 = triangle[2]

    t1 = min(t, 1.)
    t2 = max( min(t - 1., 1.), 0. )
    t3 = max( min(t - 2., 1.), 0. )

    ax.plot([x0, x0 + t1 * (x1 - x0)], [y0, y0 + t1 * (y1 - y0)], '-', lw = 1, color = 'black')
    ax.plot([x1, x1 + t2 * (x2 - x1)], [y1, y1 + t2 * (y2 - y1)], '-', lw = 1, color = 'black')
    ax.plot([x2, x2 + t3 * (x0 - x2)], [y2, y2 + t3 * (y0 - y2)], '-', lw = 1, color = 'black')


    
    return
    

rnd.seed(1234)
colors = ['#365162', '#9c5315', '#cdbfb3']


video_length  = 30
fps           = 24
n_frames      = int(video_length * fps)
n_points      = 10

tri = Triangulation( points = rnd.uniform(size = (n_points, 2)) )
tri.create_super_triangles()

# while not tri.complete:
#     tri.get_bad_triangles()
#     tri.merge_bad_triangles()
#     tri.remove_bad_triangles()
#     tri.triangulate_bad_polygon()
#     tri.select_next_point()
# tri.remove_super_triangles()

fig = plt.figure(figsize = [10.80/2, 19.20/2], dpi = 100)

ax1 = fig.add_axes([0.1, 0.7, 0.8, 0.2])
ax1.set_xticks([])
ax1.set_yticks([])

ax = fig.add_axes([0.1, 0.02, 0.8, 0.8])
ax.set_aspect('equal')
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
# ax.axis('off')

# for frame in range(n_frames):
#     ax1.clear()
#     ax1.text(0.5, 0.5, "frame_%d" % frame)

#     ax.clear()
#     ax.plot(tri.points[4:,0], 
#             tri.points[4:,1], 
#             'o', 
#             ms = 5, 
#             color = colors[0])
    
#     ax.plot(tri.super_points[:,0], 
#             tri.super_points[:,1], 
#             'o', 
#             ms = 5 if frame > 1*fps else 0.0001, 
#             color = colors[2])
    
#     for triangle in tri.triangulation:
#         animate_triangle_draw(tri.points[triangle], 1*fps, 1*fps + 30, frame, ax)

#     plt.pause(0.01)
#     frame += 1

#     if frame > 5*fps:
#         break


# plt.show()


# animation

# [0s-1s] points only
# [1s-5s] points, super triangle + verts
# [5s] insert point-1