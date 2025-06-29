#!/bin/bash
#SBATCH --job-name=run_jupyter_docker      # Job name
#SBATCH --output=tmp/%x.out                 # Standard output and error log (Job name and ID)
#SBATCH --error=tmp/%x.err                  # Standard error log (Job name and ID)
#SBATCH --exclusive                        # Use all available resources on the node
#SBATCH --nodes=2
#SBATCH --ntasks=2                         # Run a single task
#SBATCH --cpus-per-task=16
# So we use 1 node, but all available CPUs and RAM

srun --pty docker-compose up
