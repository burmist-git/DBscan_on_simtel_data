#!/bin/bash

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -t             : test"
    echo " [0] -h             : print help"
}

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-t" ]; then
	simtelIn=$dataOIdirPreff"/data/corsika_run1.simtel.gz"
	python3 table_test.py --astropytable $simtelIn
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
