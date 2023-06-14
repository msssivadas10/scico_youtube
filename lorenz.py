#!/usr/bin/python3
#
# @file lorenz.py Simulation of the lorenz attractor
# @author ms3
#

import os, subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import LinearSegmentedColormap



# lorentz system ode
def lorenz(t, x, sigma, rho, beta):
    dxdt = sigma * (x[1] - x[0])
    dydt = x[0] * (rho - x[2]) - x[1]
    dzdt = x[0] * x[1] - beta * x[2]
    return dxdt, dydt, dzdt

# solving the system
def solve_lorenz_ode(sigma = 10., beta = 8./3, rho = 28., n_points = 1000, n = 10): 
    t = np.linspace(0., 100., n_points)
    x = []
    for i in range(n):
        xi = odeint(lorenz, 
                    y0 = np.random.uniform(0.8, 1.2, 3), 
                    t = t, 
                    args = (sigma, rho, beta), 
                    tfirst = True)
        x.append(xi)
    return x

# create the video!
def create_video():

    video_length  = 30
    fps           = 24
    n_frames      = int(video_length * fps)
    pts_per_frame = 100
    n_points      = n_frames * pts_per_frame
    n             = 10

    # solving the system
    sigma, beta, rho = 10., 8./3, 28. # parameters
    x = solve_lorenz_ode(sigma, beta, rho, n_points, n)

    # create a colourmap
    colors = []
    for color in ['#365162', '#9c5315', '#cdbfb3']:
        r = int('0x' + color[1:3], 16) / 255
        g = int('0x' + color[3:5], 16) / 255
        b = int('0x' + color[5:7], 16) / 255
        colors.append([r,g,b])
    cmap = LinearSegmentedColormap.from_list('cmap', colors, 100)
    colors = cmap( np.linspace(0, 1, n) )

    # making the video
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    fig = plt.figure(figsize = [10.80, 19.20], dpi = 100)
    fig.set_facecolor('black')
    ax1 = fig.add_axes([0, 0, 1, 1])
    ax1.set_aspect('equal')
    ax1.axis('off')

    start = 0
    for frame in range(n_frames):
        print("generating frame %04d/%04d\r" % (frame + 1, n_frames), end = '')

        stop = start + pts_per_frame
        for xi, color_i in zip(x, colors):
            ax1.plot(xi[start:stop,0], 
                    xi[start:stop,2], 
                    lw = 1, 
                    color = color_i, 
                    alpha = 0.2)
        start = stop

        ax1.set(xlim = [-20, 20], ylim = [0, 50])
        fig.savefig('tmp/frame_%04d.png' % frame)

        # plt.pause(0.01)
    # plt.show()

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
                    'Title=Lorenz System',
                    '-metadata', 
                    'Year=2023',
                    'lorenz_system.mp4'],
                    check = True)

    for file in os.listdir('tmp'):
        os.remove(os.path.join('tmp', file))

    os.rmdir('tmp')

if __name__ == '__main__':
    create_video()