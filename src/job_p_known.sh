#!/bin/bash
#
#SBATCH -n 4              #number of cores
#SBATCH -t 0-12:00	   #runtime in D-HH:MM format
#SBATCH -p serial_requeue  #general
#SBATCH --mem-per-cpu=8000
#SBATCH -o log.out
#SBATCH -e log.err

cd ~/physics/research/nowak/indirect-rec/src
julia run_local_p_known.jl
