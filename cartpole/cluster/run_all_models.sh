#!/bin/bash

#SBATCH --account=carney-ashenhav-condo
#SBATCH --time=700:00:00
#SBATCH --mem=48G
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ivan_grahek@brown.edu
#SBATCH -J 	cartpole_sims
#SBATCH -o log/R-%x.%j.out

# Array of intervals and horizons
declare -a intervals=(1 2 3)
declare -a horizons=(1 2 3)

# Loop over just intervals and horizons
for interval in "${intervals[@]}"; do
    for horizon in "${horizons[@]}"; do
        # Submit job with specific parameters
        sbatch \
            --job-name="MPC_interval${interval}_horizon${horizon}" \
            --account=carney-ashenhav-condo \
            --time=70:00:00 \
            --mem=10G \
            --nodes=1 \
            -o "log/MPC_interval${interval}_horizon${horizon}.%j.out" \
            run_one_model.sh "$interval" "$horizon"
    done
done
