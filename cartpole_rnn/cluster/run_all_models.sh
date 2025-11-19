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
declare -a ratios=(0.6 0.8 1.0 1.2 1.4)
declare -a recompute_intervals=(1 2 3 4 5 6 7 8 9 10 11)
declare -a wind_mus=(0.0 0.02 0.04 0.06)
declare -a wind_sigmas=(0.0 0.05 0.1 0.15)
declare -a world_model=('dynamics')
declare -a controller=('mpc')

# Loop 1: length_ratio and recompute_interval combinations
fixed_ratio=1.0

for length_ratio in "${ratios[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
        sbatch \
            --job-name="MPC_len${length_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${fixed_ratio}_masscart${fixed_ratio}_masspole${fixed_ratio}" \
            --account=carney-ashenhav-condo \
            --time=80:00:00 \
            --mem=40G \
            --nodes=1 \
            -o "log/MPC_len${length_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${fixed_ratio}_masscart${fixed_ratio}_masspole${fixed_ratio}.%j.out" \
            run_one_model.sh "$length_ratio" "$recompute_interval" "${wind_mus[0]}" "${wind_sigmas[0]}" "${world_model[0]}" "${controller[0]}" "$fixed_ratio" "$fixed_ratio" "$fixed_ratio"
    done
done

# Loop 2: gravity_ratio and recompute_interval combinations
for gravity_ratio in "${ratios[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
        sbatch \
            --job-name="MPC_len${fixed_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${gravity_ratio}_masscart${fixed_ratio}_masspole${fixed_ratio}" \
            --account=carney-ashenhav-condo \
            --time=80:00:00 \
            --mem=40G \
            --nodes=1 \
            -o "log/MPC_len${fixed_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${gravity_ratio}_masscart${fixed_ratio}_masspole${fixed_ratio}.%j.out" \
            run_one_model.sh "$fixed_ratio" "$recompute_interval" "${wind_mus[0]}" "${wind_sigmas[0]}" "${world_model[0]}" "${controller[0]}" "$gravity_ratio" "$fixed_ratio" "$fixed_ratio"
    done
done

# Loop 3: masscart_ratio and recompute_interval combinations
for masscart_ratio in "${ratios[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
        sbatch \
            --job-name="MPC_len${fixed_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${fixed_ratio}_masscart${masscart_ratio}_masspole${fixed_ratio}" \
            --account=carney-ashenhav-condo \
            --time=80:00:00 \
            --mem=40G \
            --nodes=1 \
            -o "log/MPC_len${fixed_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${fixed_ratio}_masscart${masscart_ratio}_masspole${fixed_ratio}.%j.out" \
            run_one_model.sh "$fixed_ratio" "$recompute_interval" "${wind_mus[0]}" "${wind_sigmas[0]}" "${world_model[0]}" "${controller[0]}" "$fixed_ratio" "$masscart_ratio" "$fixed_ratio"
    done
done

# Loop 4: masspole_ratio and recompute_interval combinations
for masspole_ratio in "${ratios[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
        sbatch \
            --job-name="MPC_len${fixed_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${fixed_ratio}_masscart${fixed_ratio}_masspole${masspole_ratio}" \
            --account=carney-ashenhav-condo \
            --time=80:00:00 \
            --mem=40G \
            --nodes=1 \
            -o "log/MPC_len${fixed_ratio}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${fixed_ratio}_masscart${fixed_ratio}_masspole${masspole_ratio}.%j.out" \
            run_one_model.sh "$fixed_ratio" "$recompute_interval" "${wind_mus[0]}" "${wind_sigmas[0]}" "${world_model[0]}" "${controller[0]}" "$fixed_ratio" "$fixed_ratio" "$masspole_ratio"
    done
done

# Loop 5: wind_mu and recompute_interval combinations (all ratios fixed at 1.0, wind_sigma fixed at first element)
for wind_mu in "${wind_mus[@]}"; do
    for recompute_interval in "${recompute_intervals[@]}"; do
        sbatch \
            --job-name="MPC_len${fixed_ratio}_recompute${recompute_interval}_windmu${wind_mu}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${fixed_ratio}_masscart${fixed_ratio}_masspole${fixed_ratio}" \
            --account=carney-ashenhav-condo \
            --time=80:00:00 \
            --mem=40G \
            --nodes=1 \
            -o "log/MPC_len${fixed_ratio}_recompute${recompute_interval}_windmu${wind_mu}_windsigma${wind_sigmas[0]}_worldmodel${world_model[0]}_controller${controller[0]}_gravity${fixed_ratio}_masscart${fixed_ratio}_masspole${fixed_ratio}.%j.out" \
            run_one_model.sh "$fixed_ratio" "$recompute_interval" "$wind_mu" "${wind_sigmas[0]}" "${world_model[0]}" "${controller[0]}" "$fixed_ratio" "$fixed_ratio" "$fixed_ratio"
    done
done

# # Loop 3: wind_sigma and recompute_interval combinations (length_ratio and wind_mu fixed at first element)
# for wind_sigma in "${wind_sigmas[@]}"; do
#     for recompute_interval in "${recompute_intervals[@]}"; do
#         sbatch \
#             --job-name="MPC_len${length_ratios[0]}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigma}_worldmodel${world_model[0]}_controller${controller[0]}" \
#             --account=carney-ashenhav-condo \
#             --time=80:00:00 \
#             --mem=40G \
#             --nodes=1 \
#             -o "log/MPC_len${length_ratios[0]}_recompute${recompute_interval}_windmu${wind_mus[0]}_windsigma${wind_sigma}_worldmodel${world_model[0]}_controller${controller[0]}.%j.out" \
#             run_one_model.sh "${length_ratios[0]}" "$recompute_interval" "${wind_mus[0]}" "$wind_sigma" "${world_model[0]}_${controller[0]}"
#     done
# done