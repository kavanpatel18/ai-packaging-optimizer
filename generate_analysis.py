# filename: generate_analysis.py

import pandas as pd
import numpy as np

# --- 1. Define Our Standard Box Sizes ---
# This MUST be the same list as in your prediction script.
STANDARD_BOXES = [
    (10, 8, 1),  # Flat mailer
    (6, 4, 4),
    (8, 6, 4),
    (10, 8, 6),
    (12, 12, 8),
    (18, 18, 16),
    (24, 18, 18)
]

def find_optimal_box(product_dims, available_boxes):
    """
    Finds the smallest box a product can fit in, allowing for rotation.
    """
    p_dims = sorted(product_dims, reverse=True)
    
    best_box = None
    min_volume = float('inf')

    for box in available_boxes:
        b_dims = sorted(box, reverse=True)
        
        if (p_dims[0] <= b_dims[0] and
            p_dims[1] <= b_dims[1] and
            p_dims[2] <= b_dims[2]):
            
            box_volume = b_dims[0] * b_dims[1] * b_dims[2]
            
            if box_volume < min_volume:
                min_volume = box_volume
                best_box = box
                
    return best_box

def run_analysis(input_csv_path, output_csv_path):
    """
    Calculates optimal packaging for an entire dataset and saves the analysis.
    """
    # 1. Load the prepared dataset
    print(f"Loading data from '{input_csv_path}'...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found. Please run Phase 1 script first.")
        return
        
    print(f"Analyzing {len(df)} products...")
    
    results = []
    
    # 2. Iterate through each product
    for index, row in df.iterrows():
        # Use the TRUE dimensions from the dataset for this analysis
        product_dims = [row['product_length'], row['product_width'], row['product_height']]
        
        # 3. Find the optimal box
        optimal_box = find_optimal_box(product_dims, STANDARD_BOXES)
        
        product_volume = product_dims[0] * product_dims[1] * product_dims[2]
        
        if optimal_box:
            box_volume = optimal_box[0] * optimal_box[1] * optimal_box[2]
            wasted_space = (1 - product_volume / box_volume) * 100
        else:
            # Handle cases where no box fits
            optimal_box = None
            box_volume = None
            wasted_space = None
            
        results.append({
            'image_id': row['image_id'],
            'product_length': row['product_length'],
            'product_width': row['product_width'],
            'product_height': row['product_height'],
            'unit': row.get('unit', 'inches'), 
            'optimal_box': str(optimal_box) if optimal_box else 'None',
            'product_volume': product_volume,
            'box_volume': box_volume,
            'wasted_space_percentage': wasted_space
        })

    # 4. Save the final analysis
    analysis_df = pd.DataFrame(results)
    analysis_df.to_csv(output_csv_path, index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to '{output_csv_path}'")
    print("\nAnalysis Summary:")
    print(analysis_df.head())


if __name__ == "__main__":
    INPUT_CSV = 'MASTER_product_dataset_prepared.csv'
    OUTPUT_CSV = 'packaging_optimization_analysis.csv'
    
    run_analysis(INPUT_CSV, OUTPUT_CSV)