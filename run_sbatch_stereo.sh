#!/bin/sh

#n_jobs_start=214
#n_jobs_stop=217
n_jobs_start=1
n_jobs_stop=71
username_whoami=$(whoami)

function printHelp {
    echo " --> ERROR in input arguments "
    echo " [0] -d    : default"
    echo " [1]       : particle type (g,gd,e,p)"
    echo " [0] -info : print info"
    echo " [0] -kill : kill all jobs"
    echo " [0] -h    : print help"
}

if [ $# -eq 0 ] 
then    
    printHelp
else
    if [ "$1" = "-d" ]; then
        if [ $# -eq 2 ]; then
	    particletype="$2"
            for i in $(seq $n_jobs_start $n_jobs_stop); do
		#echo "$i"
		sbatch run_job_stereo.sh -d $particletype $i
		#source run_job_stereo.sh -d $particletype $i
            done
	fi
    elif [ "$1" = "-info" ]; then
        squeue | head -n 1
        squeue | grep $username_whoami
    elif [ "$1" = "-kill" ]; then
        scancel --user=$username_whoami --state=pending
        scancel --user=$username_whoami --state=CG
        scancel --user=$username_whoami --state=R
    elif [ "$1" = "-h" ]; then
        printHelp
    else
        printHelp
    fi
fi
