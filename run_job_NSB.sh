#!/bin/bash -l
#SBATCH --job-name simtel%j
#SBATCH --error /srv/beegfs/scratch/users/b/burmistr/sim_telarray/nsb/dbscan_npe/job_error/crgen_%j.error
#SBATCH --output /srv/beegfs/scratch/users/b/burmistr/sim_telarray/nsb/dbscan_npe/job_output/output_%j.output
#SBATCH --mem=40G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition public-cpu
#SBATCH --time 24:00:00

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d         : default"
    echo " [1]            : jobID (0-9)"
    echo " [2]            : nsbGHz (0.386, 0.268, 0.500, 0.450, 0.400, 0.350, 0.300, 0.250, 0.200, 0.150)"
    echo " [0] -h         : print help"
}

if [ $# -eq 0 ]; then
    printHelp
else
    if [ "$1" = "-d" ]; then
	if [ $# -eq 3 ]; then
	    #
            jobID=$2
            nsbGHz=$3
	    #
	    sif_file="../ctapipe_dbscan_sim_process/ctapipe.sif"
            scratchDir="/srv/beegfs/scratch/users/b/burmistr/"
            dataOIdir_sim_telarray_Preff=$scratchDir"/sim_telarray/nsb/"
            dataOIdir_sim_telarray_data=$dataOIdir_sim_telarray_Preff"/data/"
            dataOIdir_sim_telarray_dbscan_npe=$dataOIdir_sim_telarray_Preff"/dbscan_npe/data/"
            dataOIdir_ctapipe_Preff=$scratchDir"/ctapipe/nsb/data/"
            mkdir -p $dataOIdir_sim_telarray_dbscan_npe
            #
	    simtelIn=$dataOIdir_sim_telarray_data"/corsika_dummy100000_"$nsbGHz"GHz.simtel.gz"
            dl1In=$dataOIdir_ctapipe_Preff"/corsika_dummy100000_"$nsbGHz"GHz.r1.dl1.h5"
            #
	    outpkl=$dataOIdir_sim_telarray_dbscan_npe"/corsika_dummy100000_"$nsbGHz"GHz.npe.pkl"
            outcsv=$dataOIdir_sim_telarray_dbscan_npe"/corsika_dummy100000_"$nsbGHz"GHz.npe.csv"
            outh5=$dataOIdir_sim_telarray_dbscan_npe"/corsika_dummy100000_"$nsbGHz"GHz.npe.h5"
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
	    PYTHONUNBUFFERED=1 srun singularity run -B $scratchDir:$scratchDir $sif_file python3 DBscan_on_simtel_data_stereo.py --trg $simtelIn $dl1In $outpkl $outcsv $outh5 $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv
	    #
        else
            printHelp       
        fi      
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
