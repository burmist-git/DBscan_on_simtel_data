#!/bin/bash -l
#SBATCH --job-name simtel%j
#SBATCH --error /srv/beegfs/scratch/users/b/burmistr/sim_telarray/prod5/NSB_268MHz/gamma/dbscan_npe/job_error/crgen_%j.error
#SBATCH --output /srv/beegfs/scratch/users/b/burmistr/sim_telarray/prod5/NSB_268MHz/gamma/dbscan_npe/job_output/output_%j.output
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition public-cpu
#SBATCH --time 01:00:00

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d         : default"
    echo " [1]            : particletype (gamma, gamma_diffuse, electron, proton)"
    echo " [2]            : jobID (1-5000)"
    echo " [0] -h         : print help"
}

if [ $# -eq 0 ]; then
    printHelp
else
    if [ "$1" = "-d" ]; then
        if [ $# -eq 3 ]; then
	    #
	    particletype=$2
	    jobID=$3
	    #
	    #sif_file="../ctapipe_dbscan_sim_process/ctapipe.sif"
	    sif_file="./ctapipe.sif"
            scratchDir="/srv/beegfs/scratch/users/b/burmistr/"
            dataOIdir_sim_telarray_Preff=$scratchDir"/sim_telarray/prod5/NSB_268MHz/$particletype/"
            dataOIdir_sim_telarray_data=$dataOIdir_sim_telarray_Preff"/data/"
            dataOIdir_sim_telarray_dbscan_npe=$dataOIdir_sim_telarray_Preff"/dbscan_npe/data/"
            dataOIdir_ctapipe_Preff=$scratchDir"/ctapipe/prod5/NSB_268MHz/$particletype/data/"
            mkdir -p $dataOIdir_sim_telarray_dbscan_npe
            #
            simtelIn=$dataOIdir_sim_telarray_data"/corsika_run"$jobID".simtel.gz"
            dl1In=$dataOIdir_ctapipe_Preff"/corsika_run"$jobID".r1.dl1.h5"
            #
	    outpkl=$dataOIdir_sim_telarray_dbscan_npe"/corsika_run"$jobID".npe.pkl"
            outcsv=$dataOIdir_sim_telarray_dbscan_npe"/corsika_run"$jobID".npe.csv"
            outh5=$dataOIdir_sim_telarray_dbscan_npe"/corsika_run"$jobID".npe.h5"
	    #
            pixel_mapping_csv="/DBscan_on_simtel_data/pixel_mapping.csv"
            isolated_flower_seed_super_flower_csv="/DBscan_on_simtel_data/isolated_flower_seed_super_flower.list"
            isolated_flower_seed_flower_csv="/DBscan_on_simtel_data/isolated_flower_seed_flower.list"
            all_seed_flower_csv="/DBscan_on_simtel_data/all_seed_flower.list"
	    #
            echo "simtelIn                              $simtelIn"
            echo "dl1In                                 $dl1In"
            echo "outpkl                                $outpkl"
            echo "outcsv                                $outcsv"
            echo "outh5                                 $outh5"
            echo "pixel_mapping_csv                     $pixel_mapping_csv"
            echo "isolated_flower_seed_super_flower_csv $isolated_flower_seed_super_flower_csv"
            echo "isolated_flower_seed_flower_csv       $isolated_flower_seed_flower_csv"
            echo "all_seed_flower_csv                   $all_seed_flower_csv"
            #
	    srun singularity run -B $scratchDir:$scratchDir $sif_file python3 DBscan_on_simtel_data_stereo.py --trg $simtelIn $dl1In $outpkl $outcsv $outh5 $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv
	    #echo "srun singularity run -B $scratchDir:$scratchDir $sif_file python3 DBscan_on_simtel_data_stereo.py --trg $simtelIn $dl1In $outpkl $outcsv $outh5 $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv"
	else
            printHelp       
        fi      
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
