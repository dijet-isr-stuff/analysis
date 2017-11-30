#!/usr/bin/env python3

"""
Simple script to draw data distributions
"""

from argparse import ArgumentParser
from os.path import isdir, join
import os
from h5py import File
from itertools import combinations
import numpy as np
from canvas import Canvas

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output-dir', default='.')
    return parser.parse_args()

AXES = [r'$m_{12}$', r'$m_{13}$', r'$m_{23}$']
FNAME_AXES = ['m12', 'm13', 'm23']

def make_2d(mass_cube, edges, x_ax, y_ax, out_path):
    axes = set(range(len(mass_cube.shape)))
    proj_ax = next(iter(axes - set((x_ax, y_ax))))
    projection = mass_cube.sum(axis=proj_ax)
    with Canvas(out_path) as can:
        xlow, xhigh = edges[x_ax][0], edges[x_ax][-1]
        ylow, yhigh = edges[y_ax][0], edges[y_ax][-1]
        im = can.ax.imshow(projection[1:-1,1:-1].T,
                           extent=(xlow, xhigh, ylow, yhigh),
                           origin='lower',
                           norm=mpl.colors.LogNorm())
        divider = make_axes_locatable(can.ax)
        cax = divider.append_axes("right", "5%", pad="1.5%")
        can.fig.colorbar(im, cax=cax)
        can.ax.set_xlabel(AXES[x_ax], x=0.98, ha='right')
        can.ax.set_ylabel(AXES[y_ax], y=0.98, ha='right')
    return projection

def make_1d(mass_cube, edges, ax, out_path):
    axes = set(range(len(mass_cube.shape)))
    for sumax in axes - set((ax,)):
        mass_cube = mass_cube.sum(sumax, keepdims=True)
    with Canvas(out_path) as can:
        x_centers = (edges[ax][:-1] + edges[ax][1:]) / 2
        y_vals = mass_cube.squeeze()[1:-1]
        can.ax.step(x_centers, y_vals, where='mid')
        can.ax.set_yscale('log')
        can.ax.set_xlabel(AXES[ax])


def run():
    args = parse_args()
    with File(args.input_file, 'r') as input_file:
        mass = input_file['mass']
        edges = [mass[x][1:-1] for x in mass if x.startswith('axis')]
        mass_cube = np.asarray(mass['values'])

    if not isdir(args.output_dir):
        os.mkdir(args.output_dir)

    all_pro = []
    for x_ax, y_ax in combinations(range(3), 2):
        name = '{}Vs{}.pdf'.format(FNAME_AXES[y_ax], FNAME_AXES[x_ax])
        out_path = join(args.output_dir, name)
        pro = make_2d(mass_cube, edges, x_ax, y_ax, out_path)
        logged = np.log1p(pro)
        all_pro.append(logged / logged.max())

    colors = np.stack(all_pro, 2).swapaxes(0,1)
    with Canvas(f'{args.output_dir}/colorz.pdf') as can:
        can.ax.imshow(colors, origin='lower')

    for ax in range(3):
        out_path = join(args.output_dir, '{}.pdf'.format(FNAME_AXES[ax]))
        make_1d(mass_cube, edges, ax, out_path)

if __name__ == '__main__':
    run()
