#!/usr/bin/python3
#
# @file pendulum.py simulation of damped driven pendulum
# @author ms3
#

import os, subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

video_length = 30
fps          = 24
n_frames     = int(video_length * fps)
pts_per_frame = 100
n_points     = n_frames * pts_per_frame

# pendulum differential equation
def xdot(t, x, A = 1., B = 0.4, C = 0., f = 0.):
    return x[1], -A*np.sin(x[0]) - B*x[1] + C*np.cos(f*t)

# solving the eqn
t = np.linspace(0., 10*np.pi, n_points)
x = []
n = 2
args = [(1., 0.1, 0., 0.), (1., 0.1, 0.7, 4.)]
for i in range(n):
    xi = odeint(xdot, 
                y0 = [np.pi/6 + np.random.uniform(-0.01, 0.01), 0.1 + np.random.uniform(-0.01, 0.01)], 
                t = t, 
                args = args[i],
                tfirst = True)
    x.append(xi)

colors = ['#365162', '#9c5315', '#cdbfb3']

# create phase-space axis
fig = plt.figure(figsize = [10.80, 19.20], dpi = 100)
# fig.set_facecolor('black')s
ax1 = fig.add_axes([0.1, 0.05, 0.8, 0.4])
ax1.spines[['left', 'bottom']].set_position('zero')
ax1.spines[['right', 'top']].set_visible(0)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-1.5, 1.5)
ax1.tick_params(axis = 'both', which = 'major', labelsize = 22)
# ax1.set_aspect('equal')
# ax1.axis('off')

# create pendulum axis
ax2 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
ax2.spines[['left', 'bottom']].set_position('zero')
ax2.spines[['right', 'top']].set_visible(0)
ax2.tick_params(direction = 'out', pad = -25)
ax2.set_aspect('equal')
ax2.set_xlim(-3, 3)
ax2.set_ylim(-6, 0)
ax2.set_yticks([])
ax2.tick_params(axis = 'both', which = 'major', labelsize = 22)

pendulums = []
for i in range(n):
    color_i = colors[i]
    l1, = ax2.plot([], [], '-', lw = 3 , color = color_i)
    l2, = ax2.plot([], [], 'o', ms = 20, color = color_i)
    d1, = ax1.plot([], [],      lw = 3 , color = color_i)
    pendulums.append([l1, l2, d1])


if not os.path.exists('tmp'):
    os.mkdir('tmp')

start = 0
for frame in range(n_frames):
    print("generating frame %04d/%04d\r" % (frame + 1, n_frames), end = '')

    stop = start + pts_per_frame
    for i in range(n):
        x_pos = 5 * np.sin(x[i][:stop,0])
        y_pos = 5 * np.cos(x[i][:stop,0])
        x_vel = np.cos(x[i][:stop,0]) * x[i][:stop,1]
        
        pendulums[i][0].set_xdata([0.,  x_pos[-1]])
        pendulums[i][0].set_ydata([0., -y_pos[-1]])
        pendulums[i][1].set_xdata([ x_pos[-1]])
        pendulums[i][1].set_ydata([-y_pos[-1]])
        pendulums[i][2].set_xdata(x_pos)
        pendulums[i][2].set_ydata(x_vel)

    start = stop
    fig.savefig('tmp/frame_%04d.png' % frame)

    # if frame > 5:
    #     break
    # ax1.set(xlim = [-10, 10], ylim = [-10, 10])
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
                'Title=Pendulum',
                '-metadata', 
                'Year=2023',
                'pendulum.mp4'],
                check = True)

for file in os.listdir('tmp'):
    os.remove(os.path.join('tmp', file))

os.rmdir('tmp')