#!/usr/bin/python3
#
# @file fourier_demo.py fourier series animations
# @author ms3
#

import os, subprocess
import numpy as np
import matplotlib.pyplot as plt

aspect = 19.2 / 10.8
width  = 10.8

#
# convert from signal coordinates to figure coordinates
#
def signal_to_main_coords(x, y):
    x, y = np.asfarray(x), np.asfarray(y)
    return 0.8*x + 0.5, 0.1*y + 0.8

#
# draw the signal axis on the main axis / figure
#
def draw_signal_axis(ax):
    # 
    # draw axis lines
    #
    ax.hlines(0.8, 0.0, 1.0, colors = ['black'], lw = 2)
    ax.vlines(0.5, 0.6, 1.0, colors = ['black'], lw = 2)

    # 
    # draw ticks
    #
    x_labs, y_labs   = ['-T/2', '-T/4', '', 'T/4', 'T/2'], ['-2', '-1', '', '+1', '+2']
    x_ticks, y_ticks = signal_to_main_coords([-0.5, -0.25, 0., 0.25, 0.5], [-2, -1., 0., 1., 2. ])
    x_zero, y_zero   = signal_to_main_coords(0., 0.)
    ax.hlines(y_ticks, x_zero - 0.010, x_zero + 0.010, colors = 'black', lw = 2)
    ax.vlines(x_ticks, y_zero - 0.005, y_zero + 0.005, colors = 'black', lw = 2)
    for xt, xl in zip(x_ticks, x_labs):
        ax.text(xt, y_zero - 0.05 / aspect, xl, 
                horizontalalignment = 'center',
                verticalalignment   = 'center', 
                transform           = ax.transAxes,
                fontsize            = 22)
    for yt, yl in zip(y_ticks, y_labs):
        ax.text(x_zero - 0.05, yt, yl, 
                horizontalalignment = 'center',
                verticalalignment   = 'center', 
                transform           = ax.transAxes,
                fontsize            = 22)
    return

#
# convert from fourier coordinates to figure coordinates
#
def fourier_to_main_coords(x, y, pos_only = True):
    x, y = np.asfarray(x), np.asfarray(y)
    if not pos_only:
        return 0.1 * x, 2.*y / 15. + 0.3
    return 0.1 * x, 8.*y / 30. + 0.1

#
# draw the fourier axis on the main axis / figure
#
def draw_fourier_axis(ax, pos_only = True):
    # 
    # draw axis lines
    #
    if pos_only:
        ax.hlines(0.1, 0.0, 1.0, colors = ['black'], lw = 2)
    else:
        ax.hlines(0.3, 0.0, 1.0, colors = ['black'], lw = 2)
    ax.vlines(0.0, 0.1, 0.5, colors = ['black'], lw = 2)

    # 
    # draw ticks
    #
    if pos_only:
        y_ticks = [0, 0.5, 1.0, 1.5]
    else:
        y_ticks = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
    x_labs, y_labs   = ['0'] + ['%df'%i for i in range(1, 11)], list(map(str, y_ticks))
    x_ticks, y_ticks = fourier_to_main_coords(np.arange(11), y_ticks, pos_only)
    x_zero, y_zero   = fourier_to_main_coords(0., 0., pos_only)
    ax.hlines(y_ticks, x_zero - 0.010, x_zero + 0.010, colors = 'black', lw = 2)
    ax.vlines(x_ticks, y_zero - 0.005, y_zero + 0.005, colors = 'black', lw = 2)
    for xt, xl in zip(x_ticks, x_labs):
        ax.text(xt, y_zero - 0.05 / aspect, xl, 
                horizontalalignment = 'center',
                verticalalignment   = 'center', 
                transform           = ax.transAxes,
                fontsize            = 22)
    for yt, yl in zip(y_ticks, y_labs):
        ax.text(x_zero - 0.05, yt, yl, 
                horizontalalignment = 'center',
                verticalalignment   = 'center', 
                transform           = ax.transAxes,
                fontsize            = 22)
    return

#
# generate a square wave with frequency f (default = 1)
#
def square_wave(t, f = 1.0):
    t = np.asfarray(t)
    return 1. + 2. * (2. * np.floor(f*t) - np.floor(2*f*t))

#
# generate a triangle wave with frequency f (default = 1)
#
def triangle_wave(t, f = 1.0):
    t = np.asfarray(t)
    return 2*np.abs(2*(f*t - np.floor(f*t + 0.5))) - 1

#
# generate a sawtooth wave with frequency f (default = 1)
#
def swatooth_wave(t, f = 1.0):
    t = np.asfarray(t)
    return 2*(f*t - np.floor(0.5 + f*t))

#
# generate a sine wave with frequency f (default = 1)
#
def sine_wave(t, f = 1.0):
    t = np.asfarray(t)
    return np.sin(2*np.pi*f*t)

#
# calculate the fourier sine series coefficients using the orthogonality of sine functions
#
def get_fourier_coeffs(ys, t, f_max = 10):
    dt = t[1] - t[0]
    a, y_sine = [], []
    for f in range(1, f_max + 1):
        yh = sine_wave(t, f)
        af = 2 * np.sum(ys * yh) * dt
        a.append(af)
        y_sine.append(yh)
    return a, y_sine

#
# linear interpolation from value xa to xb
#
def lerp(t, xa, xb):
    return xa + (xb - xa) * t

#
# create the fourier series animation video for a given signal with frequency f = 1 and 
# save with given title 
# 
def create_fourier_series_video(signal, title, pos_only = True):
    #
    # generating data
    #
    n_samples         = 500
    f_max             = 10
    t, f              = np.linspace(-0.5, 0.5, n_samples, endpoint = False), np.arange(1, f_max + 1)
    y_sig             = signal(t, f = 1.0)
    y_for, sine_waves = get_fourier_coeffs(y_sig, t, f_max)

    t_axis, ys_axis   = signal_to_main_coords(t, y_sig)
    f_axis, yf_axis   = fourier_to_main_coords(f, y_for, pos_only)
    sw_axis1 = []
    sw_axis2 =[]
    for __sine_wave in sine_waves:
        _, __sine_wave1 = signal_to_main_coords(0., __sine_wave)
        sw_axis1.append(__sine_wave1)
        _, __sine_wave2 = signal_to_main_coords(0., __sine_wave * y_sig)
        sw_axis2.append(__sine_wave2)

    #
    # create drawing area
    #
    fig = plt.figure(figsize = [width, aspect * width], dpi = 100)
    ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')
    ax.set(xlim = [-0.02, 1.02], ylim = [0, 1])
    draw_signal_axis(ax)            # axis to draw signal
    draw_fourier_axis(ax, pos_only) # axis to draw fourier coefficents

    sig_line, = ax.plot([], [], '-', lw = 4, color = '#9c5315')
    sin_line, = ax.plot([], [], '-', lw = 4, color = '#365162')
    for_line, = ax.plot([], [], 'o', ms = 10, color = '#365162')
    for_lines = []
    for _ in yf_axis:
        __line, = ax.plot([], [], '-', lw = 4, color = '#365162')
        for_lines.append(__line)

    video_length = 30
    fps          = 24
    n_frames     = int(video_length * fps)


    ff_animate_signal     = int(1.00 * fps)
    df_padding            = int(1.00 * fps)
    df_animate_sinewave   = int(1.00 * fps)
    df_wave_transition    = int(0.75 * fps)
    df_fourier_transition = int(0.50 * fps)
    df_fourier_show_lines = int(0.50 * fps)
    ff_animate_sinewave   = ff_animate_signal + df_padding + df_animate_sinewave
    ff_wave_transition    = ff_animate_sinewave + df_wave_transition
    ff_fourier_transition = ff_wave_transition + df_fourier_transition
    ff_fourier_show_lines = ff_fourier_transition + df_fourier_show_lines
    sine_index            = 0
    finalize              = False

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    #
    # rendering frames
    # 
    for frame in range(n_frames):
        # break
        print("generating frame %04d/%04d\r" % (frame + 1, n_frames), end = '')
        fig.savefig('tmp/frame_%04d.png' % frame)
        # plt.pause(0.01)

        # 
        # animate signal wave
        #  
        max_signal_sample = n_samples
        if frame < ff_animate_signal:
            max_signal_sample = int( max_signal_sample * frame / ff_animate_signal )

        sig_line.set_xdata( t_axis[:max_signal_sample])
        sig_line.set_ydata(ys_axis[:max_signal_sample])

        if (frame < ff_animate_signal + df_padding) or finalize:
            continue

        #    
        # animate fourier series
        #

        if frame < ff_animate_sinewave:
            n = int(n_samples * (frame - ff_animate_sinewave + df_animate_sinewave) / df_animate_sinewave)
            sin_line.set_xdata(t_axis[:n])
            sin_line.set_ydata(sw_axis1[sine_index][:n] )
            for_line.set_xdata( f_axis[:sine_index])
            for_line.set_ydata(yf_axis[:sine_index])
            continue

        if frame < ff_wave_transition:
            frac = 1. - (ff_wave_transition - frame) / df_wave_transition
            __y  = lerp(frac, sw_axis1[sine_index], sw_axis2[sine_index])
            sin_line.set_xdata(t_axis)
            sin_line.set_ydata(__y)
            for_line.set_xdata( f_axis[:sine_index])
            for_line.set_ydata(yf_axis[:sine_index])
            continue

        if frame < ff_fourier_transition:
            frac = 1. - (ff_fourier_transition - frame) / df_fourier_transition
            __x  = lerp(frac, t_axis, f_axis[sine_index])
            __y  = lerp(frac, sw_axis2[sine_index], yf_axis[sine_index])
            sin_line.set_xdata(__x)
            sin_line.set_ydata(__y)
            for_line.set_xdata( f_axis[:sine_index])
            for_line.set_ydata(yf_axis[:sine_index])
            continue

        if frame < ff_fourier_show_lines:
            scale = (ff_fourier_show_lines - frame - 1) / df_fourier_show_lines
            for_lines[sine_index].set_xdata([ f_axis[sine_index], f_axis[sine_index]  ]) 
            for_lines[sine_index].set_ydata([ fourier_to_main_coords(0., y_for[sine_index] * scale, pos_only)[1], yf_axis[sine_index] ])
            sin_line.set_xdata([])
            sin_line.set_ydata([])
            for_line.set_xdata( f_axis[:sine_index+1])
            for_line.set_ydata(yf_axis[:sine_index+1])
            continue
        
        if frame == ff_fourier_show_lines:
            if sine_index < f_max-1:
                sine_index += 1
            else:
                finalize = True
            ff_animate_sinewave   = ff_fourier_show_lines + df_animate_sinewave
            ff_wave_transition    = ff_animate_sinewave + df_wave_transition
            ff_fourier_transition = ff_wave_transition + df_fourier_transition
            ff_fourier_show_lines = ff_fourier_transition + df_fourier_show_lines
            continue

    # plt.show()
    # return

    #
    # generating video from frames
    #
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
                    'Title=%s' % title,
                    '-metadata', 
                    'Year=2023',
                    '%s.mp4' % ('_'.join(title.split(' ')))],
                    check = True)

    for file in os.listdir('tmp'):
        os.remove(os.path.join('tmp', file))

    os.rmdir('tmp')
    return

def main():

    # series for a square wave 
    create_fourier_series_video(signal = square_wave, title = 'Fourier series of square wave', pos_only = True)

    # series for a sawtooth wave
    create_fourier_series_video(signal = swatooth_wave, title = 'Fourier series of sawtooth wave', pos_only = False)

    return

if __name__ == '__main__':
    main()