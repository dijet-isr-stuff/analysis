#!/usr/bin/env python3

"""
Make all the histograms
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_files', nargs='+')
    return parser.parse_args()

def get_mass(jets, j1, j2):
    eta, phi, pt = [jets['jet_' + x] for x in ['eta','phi','pt']]
    m2 = 2*pt[:,j1]*pt[:,j2]*(np.cosh(eta[:,j1] - eta[:,j2]) -
                              np.cos(phi[:,j1] - phi[:,j2]))
    return np.sqrt(m2)

def make_hists(grp, slice_size=100000):
    event_ds = grp['1d']
    jets_ds = grp['2d']
    for start in range(0, event_ds.shape[0], slice_size):
        sl = slice(start, start+slice_size)
        weights = event_ds['weight', sl]
        jets = jets_ds[sl,:]

        pass_jvt = jets['jet_JvtPass_Medium'] == 1
        good_jets = pass_jvt & ~np.isnan(jets['jet_eta'])
        good_jet_number = np.add.accumulate(good_jets, axis=1)
        n_jets = good_jet_number[:,-1]
        good_jet_number[~good_jets] = 0
        good_events = n_jets >= 3

        weights = weights[good_events]
        jets = jets[good_events,:]
        good_jet_number = good_jet_number[good_events,:]

        sel_jets = np.stack(
            [jets[good_jet_number == x] for x in [1,2,3]], axis=1)

        mass23 = get_mass(sel_jets, 1, 2)
        mass13 = get_mass(sel_jets, 0, 2)
        print(sel_jets, np.stack((mass23, mass13), axis=1))


def run():
    args = get_args()

    for fname in args.input_files:
        with File(fname) as h5file:
            grp = h5file['outTree']
            make_hists(grp, 10)

if __name__ == "__main__":
    run()
