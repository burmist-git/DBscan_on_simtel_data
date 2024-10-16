#!/bin/bash

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d             : trigger"
    echo " [0] -n             : noise"
    echo " [0] -h             : print help"
}

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-d" ]; then
	#
	#dataOIdirPreff="../scratch/simtel_data/proton_st_NSB268MHz/"
	#dataOI_npe_dirPreff="../scratch/simtel_data/proton_st_NSB268MHz/npe/"
	#No NSB
	dataOIdirPreff="../scratch/simtel_data/proton_st/"
	dataOI_npe_dirPreff="../scratch/simtel_data/proton_st/npe/"
	#
	mkdir -p $dataOI_npe_dirPreff
	simtelIn=$dataOIdirPreff"/data/corsika_run1.simtel.gz"
	outpkl=$dataOI_npe_dirPreff"/corsika_run1.npe.pkl"
	outcsv=$dataOI_npe_dirPreff"/corsika_run1.npe.csv"
	#
	pixel_mapping_csv="pixel_mapping.csv"
        isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
        isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
	all_seed_flower_csv="all_seed_flower.list"
	python DBscan_on_simtel_data_stereo.py --trg $simtelIn $outpkl $outcsv $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv
    elif [ "$1" = "-n" ]; then
	#
	dataOIdirPreff="../scratch/simtel_data/proton_st_NSB268MHz/"
	dataOI_npe_dirPreff="../scratch/simtel_data/proton_st_NSB268MHz/npe/"
	#
	mkdir -p $dataOI_npe_dirPreff
	simtelIn=$dataOIdirPreff"/data/corsika_run1.simtel.gz"
	outpkl=$dataOI_npe_dirPreff"/corsika_run1.npe.pkl"
	outcsv=$dataOI_npe_dirPreff"/corsika_run1.npe.csv"
	#
	pixel_mapping_csv="pixel_mapping.csv"
        isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
        isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
	all_seed_flower_csv="all_seed_flower.list"
	python DBscan_on_simtel_data_stereo.py --noise $simtelIn $outpkl $outcsv $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi

#espeak "I have done"
