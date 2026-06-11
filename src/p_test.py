import os
import argparse
import pandas as pd

def process_and_print_stats(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...\n")
    df = pd.read_csv(csv_path)

    # Identify the model class prediction columns dynamically (e.g., M1_Class, M2_Class...)
    model_class_cols = [col for col in df.columns if col.endswith('_Class') and col != 'Ensemble_Class']
    
    # ---------------------------------------------------------
    # Define Masks for the 3 Conditions
    # ---------------------------------------------------------
    
    # Condition 1: All individual predictions are identical
    # If the number of unique values across the model class columns is 1, they all agree.
    cond1_mask = df[model_class_cols].nunique(axis=1) == 1
    df_cond1 = df[cond1_mask]

    # Condition 2: At least 4 predictions match the ensemble result
    # Count how many models match the Ensemble_Class for each row
    match_counts = df[model_class_cols].eq(df['Ensemble_Class'], axis=0).sum(axis=1)
    cond2_mask = match_counts >= 4
    df_cond2 = df[cond2_mask]

    # Condition 3: Final ensemble probability > 0.85
    cond3_mask = df['Ensemble_Prob'] > 0.85
    df_cond3 = df[cond3_mask]

    # ---------------------------------------------------------
    # Helper function to print stats for a given filtered dataframe
    # ---------------------------------------------------------
    def print_condition_stats(filtered_df, condition_title):
        print("=" * 60)
        print(condition_title)
        print("=" * 60)
        
        # Get unique classes present in the whole dataset to ensure we check all classes (e.g., 0, 1, 2, 3, 4)
        all_classes = sorted(df['Ensemble_Class'].unique())
        
        for c in all_classes:
            # Filter by class
            class_df = filtered_df[filtered_df['Ensemble_Class'] == c]
            total_count = len(class_df)
            
            print(f"Class {c} | Total qualified images: {total_count}")
            
            if total_count > 0:
                # Sort descending by Ensemble_Prob and get top 10
                top10_df = class_df.sort_values(by='Ensemble_Prob', ascending=False).head(10)
                
                print("  Top 10 Probabilities:")
                for idx, row in enumerate(top10_df.itertuples(), 1):
                    # Using getattr to dynamically access Image_Name and Ensemble_Prob
                    img_name = getattr(row, 'Image_Name')
                    prob = getattr(row, 'Ensemble_Prob')
                    print(f"    {idx}. {prob:.4f}  ({img_name})")
            print("-" * 40)
        print("\n")

    # ---------------------------------------------------------
    # Execute and Print
    # ---------------------------------------------------------
    print_condition_stats(
        df_cond1, 
        "Condition 1: All individual predictions are identical"
    )
    
    print_condition_stats(
        df_cond2, 
        "Condition 2: At least 4 predictions match the ensemble result"
    )
    
    print_condition_stats(
        df_cond3, 
        "Condition 3: Final ensemble probability > 0.85"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pseudo labels CSV")
    parser.add_argument('--csv_path', type=str, default='./runs/pseudo/2/pseudo_labels.csv', 
                        help='Path to the pseudo_labels.csv file')
    args = parser.parse_args()
    
    process_and_print_stats(args.csv_path)