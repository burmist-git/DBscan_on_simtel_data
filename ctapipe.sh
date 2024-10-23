#!/bin/bash

in_def_file="ctapipe.def"
out_sif_file="ctapipe.sif"
out_log_file="ctapipe.log"

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d  : build apptainer (singularity) no modules need to be loaded"
    echo " [0] -t  : test sif file"
    echo " [0] -h  : print help"
}

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-d" ]; then
	#
	echo " "
	echo " "
	echo " "
	date
	#
	singularity --version
	#
	rm -rf $out_sif_file
	rm -rf $out_log_file
	time singularity build --build-arg SSH_AUTH_SOCK_USER=$SSH_AUTH_SOCK $out_sif_file $in_def_file | tee -a $out_log_file
	#
	du -hs $out_sif_file
	#
	date
	#
    elif [ "$1" = "-t" ]; then
	singularity run $out_sif_file ls
	singularity run $out_sif_file ls /ctapipe_dbscan_sim_process/
	singularity run $out_sif_file pwd
	singularity run $out_sif_file ctapipe-process --help
	singularity run $out_sif_file ctapipe-info
	singularity run $out_sif_file ctapipe-info --datamodel
	echo "singularity run $out_sif_file ctapipe-process --help-all"
	#
	singularity run $out_sif_file ls /DBscan_on_simtel_data/
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
