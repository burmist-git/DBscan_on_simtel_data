#!/bin/bash
#SBATCH --job-name=ctapipe
#SBATCH --output=/scratch/snx3000/lburmist/simtel_data/job_outlog/simtel.%j.out
#SBATCH --error=/scratch/snx3000/lburmist/simtel_data/job_error/simtel.%j.err
#SBATCH --account=cta03
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc

siffile="/scratch/snx3000/lburmist/singularity/21.10.2024/ctapipe.sif"

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d         : default"
    echo " [1]            : particle type (g,gd,e,p)"
    echo " [2]            : fileID (1-...)"
    echo " [0] -h         : print help"
}

if [ $# -eq 0 ]; then
    printHelp
else
    if [ "$1" = "-d" ]; then
        if [ $# -eq 3 ]; then
            if [ "$2" = "g" ]; then
		particletype="gamma"
	    elif [ "$2" = "gd" ]; then
		particletype="gamma_diffuse"
            elif [ "$2" = "e" ]; then
		particletype="electron"
            elif [ "$2" = "p" ]; then
		particletype="proton"
            fi
	    #
	    fileID=$3
	    #
	    inFilePref="/scratch/snx3000/lburmist/simtel_data/$particletype/data/"
	    outFilePref="/scratch/snx3000/lburmist/simtel_data/$particletype/npe/"
	    dataCtapipeOIdirPreff="/scratch/snx3000/lburmist/ctapipe_data/$particletype/data/"
	    mkdir -p $outFilePref
	    echo "  fileID      = $fileID"
	    echo "  inFilePref  = $inFilePref"
	    echo "  outFilePref = $outFilePref"		    
	    #
	    in_simtel_file="$inFilePref/corsika_run$fileID.simtel.gz"
	    dl1In=$dataCtapipeOIdirPreff/$particletype"_run"$fileID".dl1.h5"
	    #
	    if [ -f "$in_simtel_file" ]; then
		if [ -f "$dl1In" ]; then
		    out_pkl_file="$outFilePref/corsika_run$fileID.npe.pkl"
		    out_csv_file="$outFilePref/corsika_run$fileID.npe.csv"
		    out_h5_file="$outFilePref/corsika_run$fileID.npe.h5"
		    #
		    rm -rf $out_pkl_file
		    rm -rf $out_csv_file
		    rm -rf $out_h5_file
		    #
		    pixel_mapping_csv="pixel_mapping.csv"
		    isolated_flower_seed_super_flower_csv="isolated_flower_seed_super_flower.list"
		    isolated_flower_seed_flower_csv="isolated_flower_seed_flower.list"
		    all_seed_flower_csv="all_seed_flower.list"
		    #
		    cmd="singularity run -B /scratch/snx3000/lburmist/:/scratch/snx3000/lburmist/ $siffile python3 DBscan_on_simtel_data_stereo.py --trg $in_simtel_file $dl1In $out_pkl_file $out_csv_file $out_h5_file $pixel_mapping_csv $isolated_flower_seed_super_flower_csv $isolated_flower_seed_flower_csv $all_seed_flower_csv"
		    #echo "$cmd"
		    $cmd
		fi
	    fi
	else
	    printHelp   
	fi      
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
