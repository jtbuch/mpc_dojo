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
declare -a length_ratios=(1.0 2.0 3.0 4.0)
declare -a recompute_intervals=(1 2 3 4 5 6 7 8 9)
declare -a wind_mus=(0.0 0.05 0.1 0.15)

# Loop over all combinations and submit jobs
for length_ratio in "${length_ratios[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
        for wind_mu in "${wind_mus[@]}"; do
            sbatch \
                --job-name="MPC_len${length_ratio}_recompute${recompute_interval}_wind${wind_mu}" \
                --account=carney-ashenhav-condo \
                --time=80:00:00 \
                --mem=40G \
                --nodes=1 \
                -o "log/MPC_len${length_ratio}_recompute${recompute_interval}_wind${wind_mu}.%j.out" \
                run_one_model.sh "$length_ratio" "$recompute_interval" "$wind_mu"
        done
    done
done