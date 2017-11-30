#!/usr/bin/env python2

from argparse import ArgumentParser
import json, re, sys

def get_args():
    parser = ArgumentParser()
    parser.add_argument('input_files', nargs='+')
    return parser.parse_args()

def run():
    args = get_args()
    meta_counts = {}
    ds_regex = re.compile('\.([0-9]{6,8})\.')
    for fname in args.input_files:
        dsid = ds_regex.search(fname).group(1)
        counts = meta_from_file(fname)
        meta_counts[dsid] = counts
    json.dump(meta_counts, sys.stdout, indent=2)

def meta_from_file(file_name):
    from ROOT import TFile
    tfile = TFile(file_name)
    the_hist = tfile.Get("MetaData_EventCount_Test")
    the_bin_number = the_hist.GetXaxis().FindFixBin("nEvents initial")
    return the_hist.GetBinContent(the_bin_number)

if __name__ == '__main__':
    run()
