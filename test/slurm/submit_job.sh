#!/bin/bash

#SBATCH --job-name=quantum_sim      # Job name
#SBATCH --partition=gpu             # Target cluster partition (cpu, gpu, debug)
#SBATCH --nodes=1                   # Isolate 1 physical computer node
#SBATCH --gres=gpu:1                # Allocate 1 AMD GPU (MI350P or Radeon 9700)
#SBATCH --time=00:10:00             # Time limit
#SBATCH --output=outputs/%x_%j.out  # Output file (where prints go)
#SBATCH --error=outputs/%x_%j.err   # Error log file


module load rocm python

python scripts/$1