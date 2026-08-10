import glob
import os
import re

def main():
    pattern = "to_be_merged/released_full_model_*/vis/2050/none_accuracy_analysis_*.txt"
    files = glob.glob(pattern)
    
    metrics_sum = {}
    total_count = 0
    
    for f in files:
        # Extract count from filename e.g. none_accuracy_analysis_160_15.txt
        basename = os.path.basename(f)
        m = re.match(r"none_accuracy_analysis_\d+_(\d+)\.txt", basename)
        if not m:
            print(f"Skipping {f}, bad filename")
            continue
        count = int(m.group(1))
        
        with open(f, 'r') as file:
            lines = file.readlines()
            if not lines:
                continue
                
            # Line 1: acc & L/R: 0.99 & F/B: 0.96 & Bi/Sm: 0.97 & Ta/Sh: 0.99 & Stand: 1.00 & Close: 0.64 & Symm: 0.25. Total: &0.96
            line1 = lines[0].strip()
            parts = line1.replace('acc &', '').split('&')
            
            for part in parts:
                part = part.strip()
                if not part: continue
                
                # Handle 'Total: &0.96' format vs 'L/R: 0.99'
                if part.startswith('Total:'):
                    # It might be part of the last element or separated by &
                    pass
                    
            # Let's parse with regex
            matches = re.findall(r"([A-Za-z/]+):\s*(?:&)?\s*([\d\.]+)", line1)
            for key, val_str in matches:
                val = float(val_str.rstrip('.'))
                if key not in metrics_sum:
                    metrics_sum[key] = 0.0
                metrics_sum[key] += val * count
                
            # Parse 'means of mean'
            if len(lines) > 1:
                line2 = lines[1].strip()
                m2 = re.match(r"means of mean:\s*([\d\.]+)", line2)
                if m2:
                    val = float(m2.group(1))
                    if "means of mean" not in metrics_sum:
                        metrics_sum["means of mean"] = 0.0
                    metrics_sum["means of mean"] += val * count
            
            total_count += count

    if total_count == 0:
        print("No valid files found!")
        return
        
    out_dir = "to_be_merged/complete_released_full_model/vis/2050"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "none_accuracy_analysis.txt")
    
    with open(out_path, 'w') as f:
        # Reconstruct the string
        # Target format: acc & L/R: 0.99 & F/B: 0.96 & Bi/Sm: 0.97 & Ta/Sh: 0.99 & Stand: 1.00 & Close: 0.64 & Symm: 0.25. Total: &0.96
        # Order: L/R, F/B, Bi/Sm, Ta/Sh, Stand, Close, Symm, Total
        order = ["L/R", "F/B", "Bi/Sm", "Ta/Sh", "Stand", "Close", "Symm"]
        
        parts = []
        for key in order:
            if key in metrics_sum:
                avg = metrics_sum[key] / total_count
                parts.append(f"{key}: {avg:.2f}")
                
        # Total
        if "Total" in metrics_sum:
            avg_total = metrics_sum["Total"] / total_count
            line1_out = "acc & " + " & ".join(parts) + f". Total: &{avg_total:.2f}\n"
        else:
            line1_out = "acc & " + " & ".join(parts) + "\n"
            
        f.write(line1_out)
        
        if "means of mean" in metrics_sum:
            avg_mean = metrics_sum["means of mean"] / total_count
            f.write(f"means of mean: {avg_mean:.2f}\n")
            
    print(f"Calculated average over {total_count} scenes.")
    print(f"Output written to {out_path}")

if __name__ == "__main__":
    main()
