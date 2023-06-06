import os, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree

colors = np.array(['#365162', '#9c5315', '#cdbfb3']) #np.array(['#c9dda6', '#a67178', '#a5ccac'])

def create_flow_chart(fig, beta, gamma):
    ax0 = fig.add_axes([0., 0.75, 1., 0.2])
    ax0.set_aspect('equal')
    ax0.axis('off')

    texts = 'S,I,R'.split(',')
    rates = [('beta', beta), ('gamma', gamma)]

    x, y, width, height, x_space = 0., 0., 3., 3., 4.
    x_shift, y_shift = width + x_space, 0.
    for i in range(3):
        ax0.add_patch(patches.Rectangle([x, y], 
                                        width, 
                                        height,
                                        edgecolor = 'black',
                                        facecolor = colors[i] + 'aa',
                                        lw = 2,
                                        )
                                    )
        
        ax0.text(x + 0.5*width, 
                y + 0.5*height, 
                '$%s$' % texts[i],
                fontsize            = 32,
                horizontalalignment = "center",
                verticalalignment   = "center",
                )
        
        if i != 2:
            ax0.add_patch(patches.FancyArrowPatch([x + width,   y + 0.5*height],
                                                  [x + x_shift, y + 0.5*height],
                                                  arrowstyle = patches.ArrowStyle("Fancy", 
                                                                                  head_length = 20, 
                                                                                  head_width  = 20, 
                                                                                  tail_width  = 0.2,
                                                                                  ),
                                                  facecolor = 'black',
                                                  )
                                                )
            ax0.text(x + width + 0.5*x_space, 
                     y + 0.8*height, 
                     '$\\%s = %.2g$' % rates[i],
                     fontsize            = 24,
                     horizontalalignment = "center",
                     verticalalignment   = "center",
                     )
        
        x, y = x + x_shift, y + y_shift
        
    ax0.set(xlim = [-x_space, x], ylim = [-0.5, 3.5])
    return ax0

def create_simulation_space(fig):
    sim_width = 0.8
    ax1 = fig.add_axes([0.1, 0.15, sim_width, sim_width])
    ax1.set_aspect('equal')
    return ax1

def create_graph(fig):
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 22)
    for i in ['top', 'bottom', 'left', 'right']:
        ax2.spines[i].set_linewidth(2)
    return ax2

def lerp_color(c1, c2, values, v1, v2):
    r1, g1, b1 = int('0x' + c1[1:3], 16), int('0x' + c1[3:5], 16), int('0x' + c1[5:7], 16)
    r2, g2, b2 = int('0x' + c2[1:3], 16), int('0x' + c2[3:5], 16), int('0x' + c2[5:7], 16)
    
    values = (values - v1) / (v2 - v1)
    r = r1 + values * (r2 - r1)
    g = g1 + values * (g2 - g1)
    b = b1 + values * (b2 - b1)
    c = np.array(list(map(lambda r,g,b: '#' + '%02x' % int(r) + '%02x' % int(g) + '%02x' % int(b),
                          r, g, b)
                        )
                    )
    return c

def draw_agents(ax1, position, state, t_minus, t_recover):
    ax1.clear()
    ax1.add_patch(patches.Rectangle([0, 0], 
                                    1., 
                                    1.,
                                    edgecolor = 'black',
                                    facecolor = 'white',
                                    lw = 4,
                                    )
                                )
    
    agent_colors = colors[state]
    j = (state == 1) & (t_minus > 0)
    agent_colors[j] = lerp_color(colors[1], colors[2], t_minus[j], 0, t_recover)

    ax1.scatter(position[:,0], position[:,1], s = 50, c = agent_colors)
    ax1.set(xlim = [0, 1], ylim = [0, 1])
    
    ax1.set_xticks([])
    ax1.set_yticks([])

def draw_graph(ax2, counts, n_agents):
    ax2.clear()
    t = len(counts) - 1
    for i in range(3):
        ax2.plot(np.arange(t+1), counts[:,i], '-', lw = 4, color = colors[i])
        ax2.plot(t, counts[-1,i], 's', ms = 14, color = colors[i])
    ax2.set_ylim([-1, n_agents + 1])
    ax2.set_xlim([-1, max(10, t+1)])
    ax2.set_xlabel('time', fontsize = 24)
    return

def create_frame(beta, gamma):
    fig = plt.figure(figsize = [10.80, 19.20], dpi = 100)
    ax0 = create_flow_chart(fig, beta, gamma) 
    ax1 = create_simulation_space(fig)
    ax2 = create_graph(fig)
    return fig, (ax1, ax2)

def initialize_simulation(n_agents, t_recover):
    position = np.random.uniform(low = 0., high = 1., size = [n_agents, 2])
    state    = np.zeros(n_agents, dtype = 'int') 
    t_minus  = np.zeros(n_agents, dtype = 'int')

    infected          = np.random.randint(0, n_agents, size = 5)
    state[infected]   = 1
    t_minus[infected] = t_recover

    counts = np.array([[n_agents - 5, 5, 0]], dtype = 'int')

    return position, state, t_minus, counts

def update_agents(position, state, t_minus, counts, n_agents, beta, t_recover, transmission_radius):
    tree = KDTree(position, leafsize = 20)
    infected,  = np.where(state == 1)
    if len(infected):
        neighbours = np.unique(np.hstack(tree.query_ball_point(position[infected,:], transmission_radius)))
        neighbours = neighbours[state[neighbours] == 0]

        transmitted = neighbours[np.random.uniform(low = 0., high = 1., size = len(neighbours)) < beta]
        state[transmitted]   = 1
        t_minus[transmitted] = t_recover + 1

    step = np.random.uniform(low = -0.05, high = 0.05, size = [n_agents, 2])
    position = (position + step) % 1.0

    t_minus[t_minus > 0] -= 1
    
    recovered = (t_minus <= 0) & (state == 1)
    t_minus[recovered] = 0
    state[recovered]   = 2

    count = np.histogram(state, [-0.5, 0.5, 1.5, 2.5])[0]
    counts = np.append(counts, count[None,:], axis = 0)

    return position, state, t_minus, counts

def create_simulation_video(n_agents, beta, t_recover, transmission_radius, t_max, video_length, fps):
    gamma    = 1 / t_recover
    n_frames = video_length * fps
    fpu      = n_frames // t_max

    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    
    fig, (ax1, ax2) = create_frame(beta, gamma)
    position, state, t_minus, counts = initialize_simulation(n_agents, t_recover)
    for frame in range(n_frames):
        print("generating frame %04d/%04d\r" % (frame + 1, n_frames), end = '')

        draw_agents(ax1, position, state, t_minus, t_recover)
        draw_graph(ax2, counts, n_agents) 

        fig.savefig('tmp/sir_sim_%04d.png' % frame)

        if frame % fpu:
            continue
     
        position, state, t_minus, counts = update_agents(position, 
                                                         state, 
                                                         t_minus, 
                                                         counts, 
                                                         n_agents, 
                                                         beta, 
                                                         t_recover,
                                                         transmission_radius)
    #     plt.pause(0.01)
    # plt.show()

    subprocess.run(['ffmpeg', 
                    '-framerate', 
                    '24',
                    '-i',
                    'tmp/sir_sim_%04d.png',
                    # '-i', 'audio_file',
                    '-shortest',
                    '-c:v', 
                    'libx264',
                    '-r', 
                    '24',
                    '-metadata', 
                    'Title=sir simulation',
                    '-metadata', 
                    'Year=2023',
                    'sir_simulation.mp4'],
                   check = True)
    
    for file in os.listdir('tmp'):
        os.remove(os.path.join('tmp', file))

    os.rmdir('tmp')
    return

if __name__ == '__main__':
    create_simulation_video(n_agents = 500, 
                            beta = 0.1, 
                            t_recover = 20, 
                            transmission_radius = 0.1, 
                            t_max = 100, 
                            video_length = 30, 
                            fps = 24)