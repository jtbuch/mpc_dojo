#!/bin/bash

#SBATCH --account=carney-ashenhav-condo
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ivan_grahek@brown.edu
#SBATCH -J 	cartpole_sims
#SBATCH -o log/R-%x.%j.out

# Array of intervals and horizons
declare -a length_ratios=(1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)

# Loop over all parameters
for length_ratio in "${length_ratios[@]}"; do
                                    # Submit job with specific parameters
                                    sbatch \
                                        --job-name="MPC_len${length_ratio}" \
                                        --account=carney-ashenhav-condo \
                                        --time=100:00:00 \
                                        --mem=48G \
                                        --nodes=1 \
                                        -o "log/MPC_len${length_ratio}.%j.out" \
                                        run_one_model.sh "$length_ratio"
done