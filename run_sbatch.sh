#!/bin/sh

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d      : default"
    echo " [1]         : particletype (gamma, gamma_diffuse, electron, proton)"
    echo " [2]         : jobs_ID_start"
    echo " [3]         : jobs_ID_stop"
    echo " [0] --nsb   : nsb"
    echo " [0] --info  : print info"
    echo " [0] --kill  : kill all jobs"
    echo " [0] --clean : clean"
    echo " [0] -h      : print help"
}

function clean {
    rm -rf *.sh~
}

counter=0

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-d" ]; then
        if [ $# -eq 4 ]; then
	    #
	    particletype=$2
	    run_i_start=$3
            run_i_stop=$4	    
            for run_i in `seq $run_i_start $run_i_stop`; do
		echo "$run_i"
		sbatch ./run_job_$particletype.sh -d $particletype $run_i
            done
        else
            printHelp
        fi
    elif [ "$1" = "--nsb" ]; then
	sbatch ./run_job_NSB.sh -d 0 0.386
	sbatch ./run_job_NSB.sh -d 1 0.268
	sbatch ./run_job_NSB.sh -d 2 0.500
	sbatch ./run_job_NSB.sh -d 3 0.450
	sbatch ./run_job_NSB.sh -d 4 0.400
	sbatch ./run_job_NSB.sh -d 5 0.350
	sbatch ./run_job_NSB.sh -d 6 0.300
	sbatch ./run_job_NSB.sh -d 7 0.250
	sbatch ./run_job_NSB.sh -d 8 0.200
	sbatch ./run_job_NSB.sh -d 9 0.150
    elif [ "$1" = "--info" ]; then
        squeue | head -n 1
        squeue | grep burmistr
	#sacct
    elif [ "$1" = "--kill" ]; then
        scancel --user=burmistr --state=pending
        scancel --user=burmistr --state=CG
        scancel --user=burmistr --state=R
    elif [ "$1" = "--clean" ]; then
	clean
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
