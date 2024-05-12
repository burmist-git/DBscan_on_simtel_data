#!/bin/bash

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d  : default"
    echo " [0] -h  : print help"
}

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-d" ]; then
	simtelIn="../simtel_data/proton/data/corsika_run1.simtel.gz"
        npecsvIn="../simtel_data/proton/npe_csv/corsika_run1.npe.csv"
        pixel_mapping_csv="pixel_mapping.csv"
        pixel_mapping_neighbors_csv="pixel_mapping_neighbors.csv"
	python DBscan_on_simtel_data.py $simtelIn $npecsvIn $pixel_mapping_csv $pixel_mapping_neighbors_csv
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi

#espeak "I have done"
