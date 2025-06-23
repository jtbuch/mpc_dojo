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
declare -a intervals=(1 3 5)
declare -a horizons=(1 10 20 30 40 50 60 70 80 90 100)
declare -a obs_noises = (0.0 0.2 0.4)
declare -a action_noises = (0.0 0.2 0.4)
declare -a obs_mus = (0.0 0.2 0.4)
declare -a action_mus = (0.0 0.2 0.4)

# Loop over all parameters
for interval in "${intervals[@]}"; do
    for horizon in "${horizons[@]}"; do
        for obs_noise in "${obs_noises[@]}"; do
            for action_noise in "${action_noises[@]}"; do
                for obs_mu in "${obs_mus[@]}"; do
                    for action_mu in "${action_mus[@]}"; do
                        # Submit job with specific parameters
                        sbatch \
                            --job-name="MPC_int${interval}_hor${horizon}_obsN${obs_noise}_actN${action_noise}_obsM${obs_mu}_actM${action_mu}" \
                            --account=carney-ashenhav-condo \
                            --time=70:00:00 \
                            --mem=10G \
                            --nodes=1 \
                            -o "log/MPC_int${interval}_hor${horizon}_obsN${obs_noise}_actN${action_noise}_obsM${obs_mu}_actM${action_mu}.%j.out" \
                            run_one_model.sh "$interval" "$horizon" "$obs_noise" "$obs_mu" "$action_noise" "$action_mu"
                    done
                done
            done
        done
    done
done
