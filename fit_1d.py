#!/usr/bin/env python3

"""
Try a 1d fit of the histograms
"""

from argparse import ArgumentParser
from h5py import File
from canvas import Canvas, RatioCanvas
import numpy as np
import os

from scipy.optimize import minimize
from george.kernels import (
    ExpSquaredKernel, MyDijetKernelSimp, PolynomialKernel)
from george import GP

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output-directory', default='fit-plots')
    return parser.parse_args()

def get_xy_pts(group, x_range=None):
    assert 'hist_type' in group.attrs
    vals = np.asarray(group['values']).sum(axis=0)
    edges = np.asarray(group['axis_1'])[1:-1]
    errors = np.sqrt((np.asarray(group['errors'])**2).sum(axis=0))
    center = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    if x_range is not None:
        low, high = x_range
        ok = (center > low) & (center < high)
    else:
        ok = np.full(center.shape, True)

    return center[ok], vals[1:-1][ok], widths[ok], errors[1:-1][ok]

def run():
    args = get_args()
    with File(args.input_file,'r') as hist_file:
        x, y, xerr, yerr = get_xy_pts(hist_file['mass_1323'], (100, np.inf))

    pltdir = args.output_directory
    if not os.path.isdir(pltdir):
        os.mkdir(pltdir)
    with Canvas(f'{pltdir}/nothings.pdf') as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')

    ly, lyerr = np.log(y), np.log1p(yerr/y)
    with Canvas(f'{pltdir}/nothings_log.pdf') as can:
        can.ax.errorbar(x, ly, yerr=lyerr, fmt='.')

    get_gp = get_fancy_gp
    initial = [30, 0.56, 100, 1, 1]
    get_gp = get_linear_gp
    # get_gp = get_simple_gp
    initial = [0.1, 10]
    bounds = [(1e-5, 200), (1, 100)]
    # get_gp = get_composite_gp
    # initial = [30.0, 1.0, 0.001, 2000.0]
    # bounds = [(0.1, 100), (-100, 100), (0.001, 10), (200, 8000)]

    ln_prob = LogLike(x, ly, lyerr, get_gp)
    result = minimize(ln_prob, initial, #method = 'Nelder-Mead',
                      # jac=ln_prob.grad
    )
    print(result)
    best_pars = result.x
    gp = get_gp(best_pars)
    gp.compute(x, lyerr)
    t = np.linspace(np.min(x), np.max(x), 500)
    mu, cov = gp.predict(ly, t)
    std = np.sqrt(np.diag(cov))

    mu_x, cov_x = gp.predict(ly, x)
    signif = (ly - mu_x) / np.sqrt(np.diag(cov_x) + lyerr**2)

    with RatioCanvas(f'{pltdir}/fits_log.pdf') as can:
        can.ax.errorbar(x, ly, yerr=lyerr, fmt='.')
        can.ax.plot(t, mu, '-r')
        can.ax.fill_between(t, mu - std, mu + std,
                            facecolor=(0, 1, 0, 0.5),
                            zorder=5, label=r'GP error = $1\sigma$')
        can.ratio.stem(x, signif, markerfmt='.', basefmt=' ')


def get_fancy_gp(pars):
    amp, a, b, c, d = pars
    kernel = amp * MyDijetKernelSimp(a, b, c, d)
    return GP(kernel)

def get_simple_gp(pars):
    amp, length = pars
    kernel = amp * ExpSquaredKernel(length)
    return GP(kernel)

def get_linear_gp(pars):
    amp, sigma = pars
    kernel = (amp * PolynomialKernel(sigma, order=2))
    return GP(kernel)

def get_composite_gp(pars):
    amp, sigma, amp2, length = pars
    kernel = (amp * PolynomialKernel(sigma, order=1)
              + amp2 * ExpSquaredKernel(length) )
    return GP(kernel)


class LogLike:
    def __init__(self, x, y, yerr, get_gp):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.get_gp = get_gp
    def __call__(self, pars):
        gp = self.get_gp(pars)
        gp.compute(self.x, self.yerr)
        err = -gp.log_likelihood(self.y)
        print(err, pars)
        return err
    def grad(self, pars):
        gp = self.get_gp(pars)
        gp.compute(self.x, self.yerr)
        return -gp.grad_lnlikelihood(self.y)



if __name__ == '__main__':
    run()
