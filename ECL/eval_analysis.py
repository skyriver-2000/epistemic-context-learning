import os
import argparse
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
from pathlib import Path

from ECL.utils.eval_utils import load_eval_data, calculate_accuracies


def main(args: argparse.Namespace):
    """Main function to run the analysis."""
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Directory not found at {input_path}")

    files = sorted(list(input_path.glob("*.json")))
    models = defaultdict(lambda: {"protocols": []})
    
    # load all files into models
    for f in files:
        model_name = f.stem.rsplit('-', 1)[0]
        protocol_name = f.stem.rsplit('-', 1)[1]
        if protocol_name == "RAW":
            models[model_name]["RAW"] = f
        else:
            models[model_name]["protocols"].append((protocol_name, f))

    # analyze each model
    for model_name, model_files in models.items():
        print(f"\n{'='*20} Analysis for Model: {model_name} {'='*20}")
        
        raw_file = model_files.get('RAW', None)
        if not raw_file:
            print(f"RAW file not found for model {model_name}. Skipping.")
            continue

        raw_data = load_eval_data(raw_file)
        raw_data = raw_data.get("outputs", {})
        if not raw_data:
            print(f"Warning: 'outputs' key not found in {raw_file.name}. Cannot determine RAW accuracy.")
            continue
        
        raw_correctness = raw_data.get("is_correct", [])
        if not raw_correctness:
            print(f"Warning: 'is_correct' key not found in {raw_file.name}. Cannot determine RAW accuracy.")
            continue
        
        datasets = raw_data.get("dataset", [])
        if not datasets:
            print(f"Warning: 'dataset' key not found in {raw_file.name}. Cannot perform dataset-level analysis.")
        
        # calculate raw accuracy
        raw_overall_acc = calculate_accuracies(raw_data)
        
        # Prepare data for combined table
        acc_data = [{"Protocal": "RAW", "Acc": f"{raw_overall_acc:.2%}", "Change": f"{0.00:.2%}"}]
        
        for proto_name, proto_file in model_files["protocols"]:
            row_acc_data = {"Protocal": proto_name}
            proto_data = load_eval_data(proto_file)
            proto_data = proto_data.get("outputs", {})
            if not proto_data:
                print(f"Warning: 'outputs' key not found in {proto_file.name}. Cannot determine {proto_name} accuracy.")
                continue
            
            proto_correctness = proto_data.get("is_correct", [])
            if not proto_correctness:
                print(f"Warning: 'is_correct' key not found in {proto_file.name}. Cannot determine {proto_name} accuracy.")
                continue
            
            proto_overall_acc = calculate_accuracies(proto_data)
            
            change = proto_overall_acc - raw_overall_acc
            
            row_acc_data[f"Acc"] = f"{proto_overall_acc:.2%}"
            row_acc_data[f"Change"] = f"{change:+.2%}"
        
            acc_data.append(row_acc_data)

        # Report final analysis
        print("\nRaw vs. Protocol Acc")
        if acc_data:
            df = pd.DataFrame(acc_data)
            print(df.to_string(index=False))
        else:
            print("No other protocols found or processed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze model evaluation results.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="mas_eval",
        help="Directory containing the .json evaluation files."
    )
    args = parser.parse_args()
    main(args) 