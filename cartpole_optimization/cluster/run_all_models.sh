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
declare -a length_ratios=(1.0 1.5 2.0 2.5 3.0)
declare -a recompute_intervals=(1 2 3 4 5 6 7 8 9 10)

# Loop over all parameters
for length_ratio in "${length_ratios[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
                                    # Submit job with specific parameters
                                    sbatch \
                                        --job-name="MPC_len${length_ratio}_recompute${recompute_interval}" \
                                        --account=carney-ashenhav-condo \
                                        --time=40:00:00 \
                                        --mem=10G \
                                        --nodes=1 \
                                        -o "log/MPC_len${length_ratio}_recompute${recompute_interval}.%j.out" \
                                        run_one_model.sh "$length_ratio" "$recompute_interval"

                                    done
done
