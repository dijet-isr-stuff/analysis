#!/usr/bin/env python3

"""
Make all the histograms
"""

from argparse import ArgumentParser
from h5py import File
from collections import defaultdict
import numpy as np


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_files', nargs='+')
    parser.add_argument('-o','--output-name', required=True)
    return parser.parse_args()

def get_mass(jets, j1, j2):
    eta, phi, pt = [jets['jet_' + x] for x in ['eta','phi','pt']]
    m2 = 2*pt[:,j1]*pt[:,j2]*(np.cosh(eta[:,j1] - eta[:,j2]) -
                              np.cos(phi[:,j1] - phi[:,j2]))
    return np.sqrt(m2)

class Histogram:
    def __init__(self, counts=0, edges=0):
        self.counts = counts
        self.edges = edges
    def __add__(self, other):
        self.counts += other.counts
        self.edges = other.edges
        return self
        # todo, check the edges
    def write_to(self, group, name):
        hist_group = group.create_group(name)
        hist_group.attrs['hist_type'] = 'n_dim'
        hist_group.create_dataset('values', data=self.counts)
        for num, edges in enumerate(self.edges):
            hist_group.create_dataset(f'axis_{num}', data=edges)

def make_hists(grp, slice_size=100000):
    event_ds = grp['1d']
    jets_ds = grp['2d']
    hists = defaultdict(Histogram)
    mass_binning = np.concatenate(
        ([-np.inf], np.linspace(0, 2e3, 200+1), [np.inf]))
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
        masses = np.stack((mass13, mass23), axis=1)
        mass_counts, edges = np.histogramdd(
            masses, bins=[mass_binning]*2, weights=weights)
        hists['mass'] += Histogram(mass_counts, edges)

    return hists

def run():
    args = get_args()
    hists = defaultdict(Histogram)
    for fname in args.input_files:
        with File(fname) as h5file:
            grp = h5file['outTree']
            n_events = grp['1d'].shape[0]
            print(f'running on {n_events:,} events')
            new_hists = make_hists(grp)
            for new_name, new_hist in new_hists.items():
                hists[new_name] += new_hist

    with File(args.output_name,'w') as out_file:
        for name, hist in hists.items():
            hist.write_to(out_file, name)

if __name__ == "__main__":
    run()
