
import os, subprocess
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize

# video parameters
video_length      = 10
fps               = 24
n_frames          = int(video_length * fps)
updates_per_frame = 10
width, height     = 10.80, 19.20


soft = 0.1    # softening parameter
G    = 39.478 # gravitational constant in au^3 yr^-2 Mo^-1

@dataclass
class History:
    path: list = None
    kinetic_energy  : list = None
    potential_energy: list = None

    def push_state(self, pos: list, ke: float, pe: float):

        if self.path is None:
            self.path = np.asfarray([pos])
            self.kinetic_energy   = np.asfarray([ke])
            self.potential_energy = np.asfarray([pe])
            return

        self.path = np.append(self.path, [pos], axis = 0)
        self.kinetic_energy   = np.append(self.kinetic_energy,   ke)
        self.potential_energy = np.append(self.potential_energy, pe)
        return

@dataclass
class Body:
    pos : list
    vel : list
    mass: float = 1.0
    acc : list  = None

    history: History = None

    kinetic_energy  : float = 0.
    potential_energy: float = 0.

    def push_state(self):

        if self.history is None:
            self.history = History()

        self.history.push_state(self.pos, self.kinetic_energy, self.potential_energy)
        return
    
    def get_acceleration(self, others: list):

        acc = np.asfarray([0., 0., 0.])
        pe  = 0.
        for other in others:
            dist_vec = other.pos - self.pos # distance vector
            dist     = np.sqrt( dist_vec[0]**2 + dist_vec[1]**2 + dist_vec[2]**2 + soft**2 ) # distance
            acc      = acc + G * other.mass * dist_vec / dist**3
            pe       = pe  + G * other.mass / dist 

        self.acc              = acc
        self.kinetic_energy   = 0.5 * self.mass * (self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)
        self.potential_energy = -self.mass * pe
        return
    
class System:

    def __init__(self) -> None:
        self.bodies = []

        self.cm_pos = None
        self.cm_vel = None
        self.ready  = False

    @property
    def nbodies(self):
        return len(self.bodies)

    def add_body(self, pos: list, vel: list, mass: float):

        if self.ready:
            raise ValueError("cannot add bodies after finalizing")
        
        self.bodies.append( Body(pos = np.asfarray(pos), vel = np.asfarray(vel), mass = mass) )
        return
    
    def _unlock(self):
        self.ready = False
        return
    
    def finalize(self):

        self.to_center_of_mass_coords()
        for i, body in enumerate(self.bodies):
            body.get_acceleration( self.bodies[:i] + self.bodies[i+1:] ) # acceleration
            body.push_state()
        self.ready = True
        return

    def to_center_of_mass_coords(self):

        total_mass, pos_cm, vel_cm = 0.0, np.array([0., 0., 0.]), np.array([0., 0., 0.])
        for body in self.bodies:
            total_mass = total_mass + body.mass
            pos_cm     = pos_cm + body.mass * body.pos
            vel_cm     = vel_cm + body.mass * body.vel

        pos_cm = pos_cm / total_mass
        vel_cm = vel_cm / total_mass
            
        for body in self.bodies:
            body.pos = body.pos - pos_cm
            body.vel = body.vel - vel_cm

        self.cm_pos = pos_cm
        self.cm_vel = vel_cm
        return
    
    def update(self, dt: float = 0.01):

        for body in self.bodies:
            body.vel = body.vel + 0.5 * body.acc * dt # half-step velocity
            body.pos = body.pos + body.vel * dt       # next position

        for i, body in enumerate(self.bodies):
            body.get_acceleration( self.bodies[:i] + self.bodies[i+1:] ) # acceleration
            body.vel = body.vel + 0.5 * body.acc * dt # half-step velocity == next velocity

        for body in self.bodies:
            body.push_state()

        self.to_center_of_mass_coords()
        return

def lagrange_points(b1: Body, b2: Body, grid: list = None):

    def force_magnitude_squared(x: float, y: float, b1: Body, b2: Body, ang_vel: float):
        dx1, dy1, dx2, dy2 = x - b1.pos[0], y - b1.pos[1], x - b2.pos[0], y - b2.pos[1]
        dist3_1, dist3_2 = (dx1**2 + dy1**2)**1.5, (dx2**2 + dy2**2)**1.5
        fx = b1.mass * dx1 / dist3_1 + b2.mass * dx2 / dist3_2 - ang_vel * x 
        fy = b1.mass * dy1 / dist3_1 + b2.mass * dy2 / dist3_2 - ang_vel * y 
        return fx**2 + fy**2

    def f1(r: float, angle: float, b1: Body, b2: Body, ang_vel: float):
        x, y = r * np.cos(angle), r * np.sin(angle)
        return force_magnitude_squared(x, y, b1, b2, ang_vel)
    
    def f2(rt: list, b1: Body, b2: Body, ang_vel: float):
        r, angle = rt
        return f1(r, angle, b1, b2, ang_vel)
    

    r1      = np.sqrt( b1.pos[0]**2 + b1.pos[1]**2 )
    ang_vel = (b1.mass + b2.mass) / ((b1.pos[0] - b2.pos[0])**2 + (b1.pos[1] - b2.pos[1])**2)**1.5
    angle   = np.arctan2(b1.pos[1], b1.pos[0])


    opts = [[(1.001 * r1, 2.000 * r1), angle + np.pi                          ],
            [(0.000,      0.999 * r1), angle                                  ],
            [(1.001 * r1, 2.000 * r1), angle                                  ],
            [(1.001 * r1, 2.000 * r1), (angle +   np.pi/6, angle +  5*np.pi/6)],
            [(1.001 * r1, 2.000 * r1), (angle + 7*np.pi/6, angle + 11*np.pi/6)]]

    # logrange points
    points = []
    for i in range(5):
        if i < 3:
            r_bound, t = opts[i]
            r0         = 0.5 * (r_bound[0] + r_bound[1])
            r          = minimize(f1, x0 = r0, args = (t, b1, b2, ang_vel), bounds = [r_bound]).x
        else:
            r_bound, t_bound = opts[i]
            r0, t0           = 0.5 * (r_bound[0] + r_bound[1]), 0.5 * (t_bound[0] + t_bound[1])
            r, t             = minimize(f2, x0 = [r0, t0], args = (b1, b2, ang_vel), bounds = [r_bound, t_bound]).x
        points.append([r*np.cos(t), r*np.sin(t), 0.])

    if grid is None:
        f = None
    else:
        f = force_magnitude_squared(grid[0], grid[1], b1, b2, ang_vel) # force map    
        f = 0.5 * np.log10(f)
    return np.asfarray(points), f

def rotate(vec: list, phi_z: float, phi_y: float = 0., phi_x: float = 0.):

    x, y, z = vec

    if phi_z:
        c, s = np.cos(phi_z), np.sin(phi_z)
        x, y = c*x - s*y, s*x + c*y
    
    if phi_y:
        c, s = np.cos(phi_y), np.sin(phi_y)
        x, z = c*x - s*z, s*x + c*z
    
    if phi_x:
        c, s = np.cos(phi_x), np.sin(phi_x)
        y, z = c*y - s*z, s*y + c*z

    return np.asfarray([x, y, z])


#################################################################################################
# video generator functions
#################################################################################################

def create_lagrange_points_video():

    system = System()
    system.add_body(pos = [0.0, 1.0, 0.0], vel = [6.0, 0.0, 0.0], mass = 0.05) 
    system.add_body(pos = [0.0, 0.0, 0.0], vel = [0.0, 0.0, 0.0], mass = 1.00) 
    # system.add_body(pos = [0.0, 1.016], vel = [6.2826, 0.0], mass = 3.0034e-6) # earth
    # system.add_body(pos = [0.0, 0.000], vel = [0.0000, 0.0], mass = 1.0000e+0) # sun
    # system.add_body(pos = [0.0, 1.6], vel = [0.1, 0.4], mass = 0.80)
    system.finalize()


    aspect = height / width
    x_lim = np.array([-1.5, 1.5])
    y_lim = aspect * x_lim

    xgrid, ygrid = np.meshgrid(np.linspace(x_lim[0] * 1.05, x_lim[1] * 1.05, 201), 
                               np.linspace(y_lim[0] * 1.05, y_lim[1] * 1.05, 201) )

    colors = np.array(['#365162', '#9c5315', '#cdbfb3'])

    fig = plt.figure(figsize = [width, height], dpi = 100)
    ax1 = fig.add_axes([0.,0.,1.,1.])
    ax1.axis('off')
    ax1.set_aspect('equal')

    if not os.path.exists('tmp'):
            os.mkdir('tmp')

    for frame in range(n_frames):

        # for _ in range(updates_per_frame):
        system.update(dt = 0.005)

        if frame < 0:
            continue

        print("generating frame %04d/%04d\r" % (frame + 1, n_frames), end = '')

        ax1.clear()
        ax1.set(xlim = x_lim, ylim = y_lim)
        for i, body in enumerate( system.bodies ):
            ax1.plot(body.history.path[-100:,0], body.history.path[-100:,1], '-', lw = 3, alpha = 0.8, color = colors[i])
            ax1.plot([body.pos[0]], [body.pos[1]], 'o', ms = 10 * np.exp(body.mass), color = colors[i])

        lps, fmap = lagrange_points(system.bodies[0], system.bodies[1], [xgrid, ygrid])
        # fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
        ax1.contourf(xgrid, ygrid, fmap, levels = np.linspace(-2, 2, 21), cmap = 'Blues')
        for i, (Lx, Ly, Lz) in enumerate( lps ):
            ax1.plot(Lx, Ly, 'o', ms = 5, color = 'black')
            ax1.text(Lx + 0.12, 
                     Ly + 0.12, 
                     '$L_%d$' % (i+1), 
                     fontsize = 28,
                     horizontalalignment = "right",
                     verticalalignment   = "top",)
            
        fig.savefig('tmp/frame_%04d.png' % frame)

    #     plt.pause(0.01)
    # plt.show()
    # return

    subprocess.run(['ffmpeg', 
                    '-framerate', 
                    '24',
                    '-i',
                    'tmp/frame_%04d.png',
                    # '-i', 'audio_file',
                    '-shortest',
                    '-c:v', 
                    'libx264',
                    '-r', 
                    '24',
                    '-metadata', 
                    'Title=Lagrange Points for 2-body system',
                    '-metadata', 
                    'Year=2023',
                    'lagrange_points.mp4'],
                    check = True)

    for file in os.listdir('tmp'):
        os.remove(os.path.join('tmp', file))

    os.rmdir('tmp')
    return

def three_body_system_video():

    system = System()
    system.add_body(pos = [0.0, 1.016, 0.0], vel = [6.2826, 0.0, 0.0], mass = 3.0034e-6) # bhoomi 
    system.add_body(pos = [0.0, 0.000, 0.0], vel = [0.0000, 0.0, 0.0], mass = 1.0000e+0) # suryan
    system.finalize()

    aspect = height / width
    x_lim = np.array([-1.5, 1.5])
    y_lim = x_lim

    colors = np.array(['#365162', '#9c5315', '#cdbfb3'])

    fig = plt.figure(figsize = [width/2, height/2], dpi = 100)
    ax1 = fig.add_axes([0.1,0.2,0.8,0.8])
    ax1.set_aspect('equal')

    # ax2 = fig.add_axes([0.55, 0.65, 0.4, 0.4])
    # ax2.set_aspect('equal')

    # ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.2])

    # if not os.path.exists('tmp'):
    #         os.mkdir('tmp')

    for frame in range(0, n_frames):

        # for _ in range(updates_per_frame):
        system.update(dt = 0.005)

        if frame < 0:
            continue

        # print("generating frame %04d/%04d\r" % (frame + 1, n_frames), end = '')

        ax1.clear()
        # ax2.clear()
        # ax3.clear()
        ax1.set(xlim = x_lim, ylim = y_lim, xticks = [], yticks = [])
        # ax1.axis('off')
        # ax2.set(xlim = [-0.02, 0.02], ylim = [-0.02, 0.02], xticks = [], yticks = [])
        for i, body in enumerate( system.bodies ):
            ax1.plot(body.history.path[-100:,0], 
                     body.history.path[-100:,1], 
                     '-', 
                     lw = 3, 
                     alpha = 0.8, 
                     color = colors[i])
            ax1.plot([body.pos[0]], 
                     [body.pos[1]], 
                     'o', 
                     ms = 10 * np.exp(body.mass), 
                     color = colors[i])
            
            # ax3.plot(body.history.potential_energy + body.history.kinetic_energy, '-', lw = 2, color = colors[i])

            # if i > 0:
            #     ax2.plot(body.history.path[-100:,0] - system.bodies[1].history.path[-100:,0], 
            #              body.history.path[-100:,1] - system.bodies[1].history.path[-100:,1], 
            #              '-', 
            #              lw = 2, 
            #              alpha = 0.8, 
            #              color = colors[i])
            #     ax2.plot([body.pos[0] - system.bodies[1].pos[0]], 
            #              [body.pos[1] - system.bodies[1].pos[1]], 
            #              'o', 
            #              ms = 10 * np.exp(body.mass), 
            #              color = colors[i])
            
        # fig.savefig('tmp/frame_%04d.png' % frame)

        plt.pause(0.01)
    plt.show()
    return

    subprocess.run(['ffmpeg', 
                    '-framerate', 
                    '24',
                    '-i',
                    'tmp/frame_%04d.png',
                    # '-i', 'audio_file',
                    '-shortest',
                    '-c:v', 
                    'libx264',
                    '-r', 
                    '24',
                    '-metadata', 
                    'Title=Object at L2 for 2-body system',
                    '-metadata', 
                    'Year=2023',
                    'object_at_L2.mp4'],
                    check = True)

    for file in os.listdir('tmp'):
        os.remove(os.path.join('tmp', file))

    os.rmdir('tmp')
    return


if __name__ == '__main__':

    # create_lagrange_points_video()
    three_body_system_video()