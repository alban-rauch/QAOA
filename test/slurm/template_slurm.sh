#!/bin/bash
################################################################################################
#SBATCH -J quantum_sim              # Job name               (or --job-name=quantum_sim)
#SBATCH -p 0745-1R5600-NOIB         # Partition name         (or --partition=0745-1R5600-NOIB)
#SBATCH -N 1                        # Nodes requested        (or --nodes=1)
#SBATCH -n 2                        # MPI processes in total
#SBATCH --ntasks-per-node=2         # MPI tasks per node
#SBATCH --exclusive                 # Resource not shared with other users
#SBATCH --time=00:10:00             # Time limit
#SBATCH --output=outputs/%x_%j.out  # Output file
#SBATCH --error=outputs/%x_%j.err   # Error file
################################################################################################


module load rocm python

python scripts/$1