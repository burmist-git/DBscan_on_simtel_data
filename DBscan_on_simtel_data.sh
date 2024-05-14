#!/bin/bash

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d             : default"
    echo " [0] --test_getters : test getters"
    echo " [0] -h             : print help"
}

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-d" ]; then
	#
	#simtelIn="/scratch/snx3000/lburmist/simtel_data/NSB386MHz/data/corsika_run1.simtel.gz"
	#headeroutpkl="/scratch/snx3000/lburmist/simtel_data/NSB386MHz/npe/corsika_run1.npe.pkl"
	#headeroutcsv="/scratch/snx3000/lburmist/simtel_data/NSB386MHz/npe/corsika_run1.npe.csv"
	#
	simtelIn="/scratch/snx3000/lburmist/simtel_data/proton/data/corsika_run1.simtel.gz"
	headeroutpkl="/scratch/snx3000/lburmist/simtel_data/proton/npe/corsika_run1.npe.pkl"
	headeroutcsv="/scratch/snx3000/lburmist/simtel_data/proton/npe/corsika_run1.npe.csv"
	#
	pixel_mapping_csv="pixel_mapping.csv"
        pixel_mapping_neighbors_csv="pixel_mapping_neighbors.csv"
	python DBscan_on_simtel_data.py $1 $simtelIn $headeroutpkl $headeroutcsv $pixel_mapping_csv $pixel_mapping_neighbors_csv
    elif [ "$1" = "--test_getters" ]; then
	pixel_mapping_csv="pixel_mapping.csv"
	simtelIn="/scratch/snx3000/lburmist/simtel_data/NSB386MHz/data/corsika_run1.simtel.gz"
	python DBscan_on_simtel_data.py $1 $simtelIn $pixel_mapping_csv
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi

#espeak "I have done"
