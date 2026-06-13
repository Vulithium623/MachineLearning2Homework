import argparse
import pandas as pd

def extract_standard_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Detect ID column
    id_col = None
    for col in ['Image_Path', 'Image_Name', 'ID', 'Unnamed: 0']:
        if col in df.columns:
            id_col = col
            break
    if not id_col:
        id_col = df.columns[0]
        
    # 2. Detect Prediction column
    pred_col = None
    for col in ['OOF_Pred', 'Ensemble_Class', 'Majority_Class', 'Pred', 'Prediction']:
        if col in df.columns:
            pred_col = col
            break
    if not pred_col:
        raise ValueError(f"Could not detect prediction column in {csv_path}. "
                         "Expected one of: 'OOF_Pred', 'Ensemble_Class', 'Majority_Class'.")

    # 3. Detect Probability column
    prob_series = None
    if 'Ensemble_Prob' in df.columns:
        prob_series = df['Ensemble_Prob']
    else:
        prob_cols = [c for c in df.columns if c.startswith('Prob_Class_')]
        if prob_cols:
            prob_series = df[prob_cols].max(axis=1)
        else:
            prob_series = pd.Series(["N/A"] * len(df))
            
    # Extract only what we need
    extracted_df = pd.DataFrame({
        'ID': df[id_col].astype(str),
        'Pred': df[pred_col],
        'Prob': prob_series
    })
    
    return extracted_df

def format_prob(val):
    try:
        f_val = float(val)
        return f"{f_val:.4f}"
    except ValueError:
        return str(val)

def main():
    parser = argparse.ArgumentParser(description="Compare predictions between two CSV files.")
    parser.add_argument("csv1", type=str, help="Path to the first CSV file (e.g., previous OOF)")
    parser.add_argument("csv2", type=str, help="Path to the second CSV file (e.g., current OOF)")
    args = parser.parse_args()

    print(f"Loading CSV 1: {args.csv1}")
    df1 = extract_standard_data(args.csv1)
    
    print(f"Loading CSV 2: {args.csv2}")
    df2 = extract_standard_data(args.csv2)

    # Merge on ID
    merged = pd.merge(df1, df2, on='ID', suffixes=('_1', '_2'), how='inner')
    
    if len(merged) == 0:
        print("Error: No matching IDs found between the two CSV files.")
        return

    # Find differences
    diff_df = merged[merged['Pred_1'] != merged['Pred_2']]
    total_diff = len(diff_df)
    
    print("=" * 60)
    print(f"Total samples compared: {len(merged)}")
    print(f"Total different predictions: {total_diff}")
    print("=" * 60)

    if total_diff == 0:
        print("Both CSV files have identical predictions!")
        return

    # Output top 10 differences
    print("\nTop 10 differences:")
    print("-" * 60)
    
    head_diff = diff_df.head(10)
    
    for i, (_, row) in enumerate(head_diff.iterrows(), 1):
        img_id = row['ID']
        pred1 = row['Pred_1']
        prob1 = format_prob(row['Prob_1'])
        pred2 = row['Pred_2']
        prob2 = format_prob(row['Prob_2'])
        
        print(f"{i:2d}. ID: {img_id}")
        print(f"    CSV 1 -> Pred: {pred1} (Prob: {prob1})")
        print(f"    CSV 2 -> Pred: {pred2} (Prob: {prob2})")
        print("-" * 60)

if __name__ == "__main__":
    main()