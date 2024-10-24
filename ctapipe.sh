#!/bin/bash

in_def_file="ctapipe.def"
out_sif_file="ctapipe.sif"
out_log_file="ctapipe.log"

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d         : build apptainer (singularity) no modules need to be loaded"
    echo " [0] -t         : test sif file"
    echo " [0] --testlive : live test in cscs"
    echo " [0] -h         : print help"
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
    elif [ "$1" = "--testlive" ]; then
	sif_file_in_cscs="/scratch/snx3000/lburmist/singularity/23.10.2024/ctapipe.sif"
        simtelIn="/scratch/snx3000/lburmist/simtel_data/proton_st/data/corsika_run1000.simtel.gz"
	singularity run $sif_file_in_cscs which ctapipe-info
	singularity run $sif_file_in_cscs ctapipe-info --version
	singularity run -B /scratch/snx3000/lburmist/:/scratch/snx3000/lburmist/ $sif_file_in_cscs python3 DBscan_on_simtel_data_stereo.py --astropytable $simtelIn

	#
	dataOIdirPreff="../scratch/simtel_data/gamma_st/"
	dataOI_npe_dirPreff="../scratch/simtel_data/gamma_st/npe/"
	#
	#dataOIdirPreff="../scratch/simtel_data/proton_st/"
	#dataOI_npe_dirPreff="../scratch/simtel_data/proton_st/npe/"
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
	


    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
