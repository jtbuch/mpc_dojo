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
declare -a recompute_intervals=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
declare -a wind_mus=(0.0 0.05 0.1 0.15)
declare -a wind_sigmas=(0.0 0.05 0.1 0.15)

# Loop 1: length_ratio and recompute_interval combinations (wind_mu and wind_sigma fixed at first element)
for length_ratio in "${length_ratios[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
        sbatch \
            --job-name="MPC_len${length_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}" \
            --account=carney-ashenhav-condo \
            --time=80:00:00 \
            --mem=40G \
            --nodes=1 \
            -o "log/MPC_len${length_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}.%j.out" \
            run_one_model.sh "$length_ratio" "$recompute_interval" "${wind_mus[0]}" "${wind_sigmas[0]}"
    done
done

# Loop 2: wind_mu and recompute_interval combinations (length_ratio and wind_sigma fixed at first element)
for wind_mu in "${wind_mus[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
        sbatch \
            --job-name="MPC_len${length_ratios[0]}_recompute${recompute_interval}_windmu${wind_mu}_windsigma${wind_sigmas[0]}" \
            --account=carney-ashenhav-condo \
            --time=80:00:00 \
            --mem=40G \
            --nodes=1 \
            -o "log/MPC_len${length_ratios[0]}_recompute${recompute_interval}_windmu${wind_mu}_windsigma${wind_sigmas[0]}.%j.out" \
            run_one_model.sh "${length_ratios[0]}" "$recompute_interval" "$wind_mu" "${wind_sigmas[0]}"
    done
done

# # Loop 3: wind_sigma and recompute_interval combinations (length_ratio and wind_mu fixed at first element)
# for wind_sigma in "${wind_sigmas[@]}"; do
#     for recompute_interval in "${recompute_intervals[@]}"; do
#         sbatch \
#             --job-name="MPC_len${length_ratios[0]}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigma}" \
#             --account=carney-ashenhav-condo \
#             --time=80:00:00 \
#             --mem=40G \
#             --nodes=1 \
#             -o "log/MPC_len${length_ratios[0]}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigma}.%j.out" \
#             run_one_model.sh "${length_ratios[0]}" "$recompute_interval" "${wind_mus[0]}" "$wind_sigma"   
#     done
# done