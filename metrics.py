import os
import re
import argparse
import json
from statistics import mean, stdev

# Regular expressions to match lines in results.txt
iter_avg_pattern = re.compile(
    r"\[ITER (\d+)\] Evaluating test avg: L1 ([\d.]+) PSNR ([\d.]+) LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)
iter_std_pattern = re.compile(
    r"\[ITER (\d+)\] Evaluating test std: L1 ([\d.]+) PSNR ([\d.]+) LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)

def extract_metrics(base_dir):
    # Dictionaries to store metrics for each iteration across all runs
    iter_psnrs = {}
    iter_psnr_stds = {}
    iter_lpipss = {}
    iter_lpips_stds = {}
    iter_ssims = {}
    iter_ssim_stds = {}
    iter_mssims = {}
    iter_mssim_stds = {}
    iter_l1s = {}
    iter_l1_stds = {}

    # Gather all iterations across all runs
    all_iterations = set()
    run_metrics = []

    # Process each subfolder
    for subdir in os.listdir(base_dir):
        results_path = os.path.join(base_dir, subdir, "results.txt")
        
        # Check if results.txt exists in the current subdir
        if os.path.isfile(results_path):
            with open(results_path, "r") as file:
                content = file.read()
                
                # Find iteration lines for avg and std metrics
                avg_matches = iter_avg_pattern.findall(content)
                std_matches = iter_std_pattern.findall(content)

                # Extract metrics for this run
                run_data = {}
                for avg_match, std_match in zip(avg_matches, std_matches):
                    iter_num_avg, l1_avg, psnr_avg, lpips_avg, ssim_avg, mssim_avg = avg_match
                    iter_num_std, l1_std, psnr_std, lpips_std, ssim_std, mssim_std = std_match
                    
                    # Ensure iteration numbers match in avg and std lines
                    assert iter_num_avg == iter_num_std, "Mismatch in iteration numbers for avg and std lines."

                    iter_num = int(iter_num_avg)
                    all_iterations.add(iter_num)  # Add to global set of iterations
                    run_data[iter_num] = {
                        "L1": (float(l1_avg), float(l1_std)),
                        "PSNR": (float(psnr_avg), float(psnr_std)),
                        "LPIPS": (float(lpips_avg), float(lpips_std)),
                        "SSIM": (float(ssim_avg), float(ssim_std)),
                        "MSSIM": (float(mssim_avg), float(mssim_std)),
                    }
                
                # Store run data for backfilling later
                run_metrics.append(run_data)

    # Sort all iterations in ascending order
    all_iterations = sorted(all_iterations)

    # Process each iteration and backfill missing values for each run
    for iter_num in all_iterations:
        # Collect metric values for this iteration across all runs
        psnr_values, psnr_std_values = [], []
        lpips_values, lpips_std_values = [], []
        ssim_values, ssim_std_values = [], []
        mssim_values, mssim_std_values = [], []
        l1_values, l1_std_values = [], []

        for run_data in run_metrics:
            # Find the latest iteration <= iter_num in this run
            available_iters = sorted(k for k in run_data.keys() if k <= iter_num)
            if available_iters:
                latest_iter = available_iters[-1]  # Get latest available iteration for this run
                metrics = run_data[latest_iter]

                # Append metrics to lists for averaging
                psnr_values.append(metrics["PSNR"][0])
                psnr_std_values.append(metrics["PSNR"][1])
                lpips_values.append(metrics["LPIPS"][0])
                lpips_std_values.append(metrics["LPIPS"][1])
                ssim_values.append(metrics["SSIM"][0])
                ssim_std_values.append(metrics["SSIM"][1])
                mssim_values.append(metrics["MSSIM"][0])
                mssim_std_values.append(metrics["MSSIM"][1])
                l1_values.append(metrics["L1"][0])
                l1_std_values.append(metrics["L1"][1])

        # Calculate mean and std values for this iteration
        if psnr_values:
            iter_psnrs[iter_num] = (mean(psnr_values), stdev(psnr_values) if len(psnr_values) > 1 else 0)
            iter_psnr_stds[iter_num] = (mean(psnr_std_values), stdev(psnr_std_values) if len(psnr_std_values) > 1 else 0)
        if lpips_values:
            iter_lpipss[iter_num] = (mean(lpips_values), stdev(lpips_values) if len(lpips_values) > 1 else 0)
            iter_lpips_stds[iter_num] = (mean(lpips_std_values), stdev(lpips_std_values) if len(lpips_std_values) > 1 else 0)
        if ssim_values:
            iter_ssims[iter_num] = (mean(ssim_values), stdev(ssim_values) if len(ssim_values) > 1 else 0)
            iter_ssim_stds[iter_num] = (mean(ssim_std_values), stdev(ssim_std_values) if len(ssim_std_values) > 1 else 0)
        if mssim_values:
            iter_mssims[iter_num] = (mean(mssim_values), stdev(mssim_values) if len(mssim_values) > 1 else 0)
            iter_mssim_stds[iter_num] = (mean(mssim_std_values), stdev(mssim_std_values) if len(mssim_std_values) > 1 else 0)
        if l1_values:
            iter_l1s[iter_num] = (mean(l1_values), stdev(l1_values) if len(l1_values) > 1 else 0)
            iter_l1_stds[iter_num] = (mean(l1_std_values), stdev(l1_std_values) if len(l1_std_values) > 1 else 0)

    return iter_psnrs, iter_lpipss, iter_ssims, iter_mssims, iter_l1s, iter_psnr_stds, iter_lpips_stds, iter_ssim_stds, iter_mssim_stds, iter_l1_stds

def save_results(base_dir, iter_psnrs, iter_lpipss, iter_ssims, iter_mssims, iter_l1s, iter_psnr_stds, iter_lpips_stds, iter_ssim_stds, iter_mssim_stds, iter_l1_stds):
    # Save all values and mean/std values to mean_results.txt
    results_path = os.path.join(base_dir, "mean_results.txt")
    with open(results_path, "w") as file:
        file.write("Mean and Std Values per Iteration:\n")

        # Write results with mean and std for each metric
        for metric_name, mean_data, std_data in [
            ("PSNR", iter_psnrs, iter_psnr_stds), 
            ("LPIPS", iter_lpipss, iter_lpips_stds),
            ("SSIM", iter_ssims, iter_ssim_stds),
            ("MSSIM", iter_mssims, iter_mssim_stds),
            ("L1", iter_l1s, iter_l1_stds)
        ]:
            file.write(f"\n{metric_name} per Iteration:\n")
            for iter_num in mean_data.keys():
                mean_val, std_dev = mean_data[iter_num]
                std_mean, std_std = std_data[iter_num]
                file.write(f"Iteration {iter_num}: Mean = {mean_val:.4f}, Std Dev = {std_dev:.4f}, Std (Mean) = {std_mean:.4f}, Std (Std Dev) = {std_std:.4f}\n")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract and compute mean and std values from results.txt files.")
    parser.add_argument("base_dir", type=str, help="Base directory containing subfolders with results.txt files.")
    args = parser.parse_args()

    # Extract metrics and calculate means and stds
    iter_psnrs, iter_lpipss, iter_ssims, iter_mssims, iter_l1s, iter_psnr_stds, iter_lpips_stds, iter_ssim_stds, iter_mssim_stds, iter_l1_stds = extract_metrics(args.base_dir)

    # Save all values and mean/std results to a file
    save_results(args.base_dir, iter_psnrs, iter_lpipss, iter_ssims, iter_mssims, iter_l1s, iter_psnr_stds, iter_lpips_stds, iter_ssim_stds, iter_mssim_stds, iter_l1_stds)
