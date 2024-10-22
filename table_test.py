#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from collections import defaultdict
from astropy.table import Table
from ctapipe.io.astropy_helpers import write_table

def astropytable_test(outtablename = 'testtable.h5'):
    #list=[24345,24234,-9]
    #data["time_start"].append(chunk["time_mono"][0])
    data = defaultdict(list)
    for i in range(15):
        data["lala"].append([1,2,3,4,5])
    #
    write_table(table=Table(data),h5file=outtablename,path="/trigger/event/telescope/tel_001",overwrite=True)
    write_table(table=Table(data),h5file="/home/burmist/home2/work/CTA/ctapipe_dev/ctapipe_dbscan_sim_process/tmp/gamma_run1.r1.dl1.h5",path="/trigger/event/telescope/tel_001",overwrite=True)
    
def main():
    pass
    
if __name__ == "__main__":
    if (len(sys.argv)==3 and (str(sys.argv[1]) == "--astropytable")):
        simtelIn = str(sys.argv[2])
        astropytable_test()
    else:
        print(" --> HELP info")
        print(" ---> for map")
        print(" [1] --astropytable")
        print(" [2] simtelIn")
