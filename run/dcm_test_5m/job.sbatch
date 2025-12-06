#!/bin/bash
#SBATCH --job-name=dcm_test_5m                  # Job name
#SBATCH --account=gts-dt98                      # Tracking account
#SBATCH -c 4                                    # Number of cores required
#SBATCH --mem-per-cpu=4G                        # Memory per core
#SBATCH --time=0-16:0:0                         # Duration of the job
#SBATCH --qos=inferno                           # Queue name (where job is submitted)
#SBATCH -o Report-%j.out                        # Combined output and error messages file (relative to where the job was submitted)
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=chung62@gatech.edu

cd /storage/home/hcoda1/5/chung62/projects/dynein-cell-model/build/examples

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $SLURM_CPUS_PER_TASK"

srun ./run_model_metrics ../../run/dcm_test_5m ../../run/dcm_test_5m/metrics.csv
