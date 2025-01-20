#!/bin/bash

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d : trigger"
    echo " [0] -s : screen"
    echo " [0] -h : print help"
}

function run_DBscan_NSB {
    #
    particle_type=$1
    process_file_prefix=$2
    NSBDIR=$3
    #
    dataOIdirPreff="../scratch/simtel_data/NSB/$NSBDIR/$particle_type/"
    dataOI_npe_dirPreff="../scratch/simtel_data/NSB/$NSBDIR/$particle_type/npe/"	
    ctaprocIn_dirPreff="../scratch/ctapipe_data/NSB/$NSBDIR/$particle_type/data/"
    #
    mkdir -p $dataOI_npe_dirPreff
    simtelIn=$dataOIdirPreff"/data/$process_file_prefix.simtel.gz"
    ctaprocIn=$ctaprocIn_dirPreff"/$process_file_prefix.dl1.h5"
    #
    outpkl=$dataOI_npe_dirPreff"/$process_file_prefix.npe.pkl"
    outcsv=$dataOI_npe_dirPreff"/$process_file_prefix.npe.csv"
    outh5=$dataOI_npe_dirPreff"/$process_file_prefix.npe.h5"
    #
    rm -rf $outpkl
    rm -rf $outcsv
    rm -rf $outh5
    #
    pixel_mapping_csv="pixel_mapping.csv"
    isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
    isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
    all_seed_flower_csv="all_seed_flower.list"
    python3 DBscan_on_simtel_data_stereo.py --trg $simtelIn $ctaprocIn $outpkl $outcsv $outh5 $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv
}

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-d" ]; then
 	#
	# NSB150MHz NSB200MHz NSB250MHz NSB300MHz NSB350MHz NSB400MHz NSB268MHz NSB386MHz
	#
        #../../scratch/simtel_data/NSB/NSB150MHz/proton/data/corsika_run7.simtel.gz
        #../../scratch/simtel_data/NSB/NSB200MHz/proton/data/corsika_run6.simtel.gz
        #../../scratch/simtel_data/NSB/NSB250MHz/proton/data/corsika_run5.simtel.gz
        #../../scratch/simtel_data/NSB/NSB300MHz/proton/data/corsika_run4.simtel.gz
        #../../scratch/simtel_data/NSB/NSB350MHz/proton/data/corsika_run3.simtel.gz
        #../../scratch/simtel_data/NSB/NSB400MHz/proton/data/corsika_run2.simtel.gz
        #../../scratch/simtel_data/NSB/NSB268MHz/proton/data/corsika_run1.simtel.gz
        #../../scratch/simtel_data/NSB/NSB386MHz/proton/data/corsika_run307.simtel.gz
	#
	particle_type="proton"
        process_file_prefix="corsika_run7"
        NSBDIR="NSB150MHz"
	run_DBscan_NSB $particle_type $process_file_prefix $NSBDIR
        process_file_prefix="corsika_run6"
	NSBDIR="NSB200MHz"
	run_DBscan_NSB $particle_type $process_file_prefix $NSBDIR
        process_file_prefix="corsika_run5"
	NSBDIR="NSB250MHz"
	run_DBscan_NSB $particle_type $process_file_prefix $NSBDIR
        process_file_prefix="corsika_run4"
	NSBDIR="NSB300MHz"
	run_DBscan_NSB $particle_type $process_file_prefix $NSBDIR
        process_file_prefix="corsika_run3"
	NSBDIR="NSB350MHz"
	run_DBscan_NSB $particle_type $process_file_prefix $NSBDIR
	process_file_prefix="corsika_run2"
	NSBDIR="NSB400MHz"
	run_DBscan_NSB $particle_type $process_file_prefix $NSBDIR
        process_file_prefix="corsika_run1"
	NSBDIR="NSB268MHz"
	run_DBscan_NSB $particle_type $process_file_prefix $NSBDIR
	process_file_prefix="corsika_run307"
	NSBDIR="NSB386MHz"
	run_DBscan_NSB $particle_type $process_file_prefix $NSBDIR
	#
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi

#espeak "I have done"
