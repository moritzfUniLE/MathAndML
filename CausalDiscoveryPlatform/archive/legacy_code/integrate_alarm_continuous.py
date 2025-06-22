#!/usr/bin/env python3
"""
Script to integrate continuous ALARM dataset from NoTearsData directory.
"""
import os
import shutil
import json
import pandas as pd

def integrate_alarm_continuous():
    """Integrate continuous ALARM dataset."""
    
    # Define paths
    source_dir = "../NoTearsData"
    datasets_dir = "datasets"
    alarm_dir = os.path.join(datasets_dir, "alarm_continuous")
    
    # Check if source files exist
    alarm_data_path = os.path.join(source_dir, "alarm_data_continuous.csv")
    alarm_bif_path = os.path.join(source_dir, "alarm_ground_truth.bif")  # Same structure
    
    if not os.path.exists(alarm_data_path):
        print(f"ERROR: ALARM continuous data file not found at {alarm_data_path}")
        return False
    
    if not os.path.exists(alarm_bif_path):
        print(f"ERROR: ALARM BIF file not found at {alarm_bif_path}")
        return False
    
    # Create datasets directory if it doesn't exist
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Create alarm_continuous dataset directory
    os.makedirs(alarm_dir, exist_ok=True)
    
    print(f"Integrating ALARM continuous dataset...")
    
    # Copy data file
    data_dest = os.path.join(alarm_dir, "alarm_continuous_data.csv")
    shutil.copy2(alarm_data_path, data_dest)
    print(f"✓ Copied data file to {data_dest}")
    
    # Copy BIF file
    bif_dest = os.path.join(alarm_dir, "alarm_continuous.bif")
    shutil.copy2(alarm_bif_path, bif_dest)
    print(f"✓ Copied BIF file to {bif_dest}")
    
    # Analyze dataset to create info.json
    try:
        df = pd.read_csv(data_dest)
        n_samples, n_vars = df.shape
        variable_names = df.columns.tolist()
        
        # Count edges from BIF file (simplified)
        edge_count = 0
        with open(bif_dest, 'r') as f:
            bif_content = f.read()
            # Count probability statements with conditioning (contains '|')
            for line in bif_content.split('\n'):
                if 'probability' in line and '|' in line:
                    edge_count += 1
        
        # Create info.json
        info = {
            "name": "alarm_continuous",
            "description": "ALARM network (continuous) - Continuous version of the ALARM medical diagnosis network with 37 variables",
            "type": "continuous",
            "nodes": n_vars,
            "variables": variable_names,
            "edges": edge_count,
            "samples": n_samples,
            "bif_file": "alarm_continuous.bif",
            "data_file": "alarm_continuous_data.csv",
            "structure_parsed": True,
            "source": "Continuous version of classic ALARM Bayesian Network",
            "domain": "Medical diagnosis network (continuous variables)"
        }
        
        info_path = os.path.join(alarm_dir, "info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Created dataset info file")
        print(f"  - Variables: {n_vars}")
        print(f"  - Samples: {n_samples}")
        print(f"  - Estimated edges: {edge_count}")
        print(f"  - Data type: continuous")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to analyze dataset: {e}")
        return False

def main():
    """Main function."""
    print("ALARM Continuous Dataset Integration")
    print("=" * 40)
    
    if integrate_alarm_continuous():
        print("\n✅ ALARM continuous dataset successfully integrated!")
        print("\nYou can now:")
        print("1. Launch the web GUI: python notears_web_gui.py")
        print("2. Select 'alarm_continuous' from the dataset dropdown")
        print("3. Run NOTEARS algorithms on the continuous ALARM network")
    else:
        print("\n❌ Failed to integrate ALARM continuous dataset")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())