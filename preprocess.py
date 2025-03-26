"""
Module handling preprocessing mapping files for use with network propagation.

Requires the following files:
Data/Mappings/*.tsv - Mappings containing proteins, p-values, genes, and SNPs

Author: Luke Krongard
"""

#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Dict, List
import os

#-----------------------------------------------------------------------------------------------------------------------

def log_transform(x: float) -> float:
    """
    Log-transform a value to highlight the most significant results.
    """
    return -np.log10(x)

#-----------------------------------------------------------------------------------------------------------------------

def robust_sigmoid(x: float, median: float, iqr_x: float) -> float:
    """
    Calculate a robust sigmoid function for a given x value.
    """
    return 1 / (1 + np.exp(-1 * (x - median) / iqr_x))

#-----------------------------------------------------------------------------------------------------------------------

def preprocess_mappings(file_path: str) -> None:
    """
    Preprocesses a mapping file for use with network propagation.
    """
    # Load the mapping file
    mappings = pd.read_csv(file_path, sep='\t')
    
    # Extract the base name from the file path
    file_name = os.path.basename(file_path)
    base_name = file_name.replace('_mappings.tsv', '')
    
    # Log-transform the p-values
    mappings['Log_P-Value'] = mappings['P-Value'].apply(log_transform)

    # Calculate the interquartile range, median, and IQR_x
    q1 = mappings['Log_P-Value'].quantile(0.25)
    q3 = mappings['Log_P-Value'].quantile(0.75)
    median = mappings['Log_P-Value'].quantile(0.5)
    iqr_x = (q3 - q1) / (2 * np.sqrt(2) * 0.476996)

    # Calculate robust sigmoid function
    mappings['Robust_Sigmoid'] = mappings['Log_P-Value'].apply(lambda x: robust_sigmoid(x, median, iqr_x))

    # Apply min-max scaling to the robust sigmoid values
    min_val = mappings['Robust_Sigmoid'].min()
    max_val = mappings['Robust_Sigmoid'].max()
    mappings['Scaled_Robust_Sigmoid'] = (mappings['Robust_Sigmoid'] - min_val) / (max_val - min_val)

    # Drop unnecessary columns
    mappings.drop(columns=['P-Value', 'Log_P-Value', 'Robust_Sigmoid'], inplace=True)

    # Arrange columns in a specific order
    mappings = mappings[['Protein', 'Scaled_Robust_Sigmoid', 'Gene', 'SNP']]

    # Sort by score in descending order
    mappings.sort_values(by='Scaled_Robust_Sigmoid', ascending=False, inplace=True)
    
    # Save the processed mapping file
    output_path = os.path.join("Data/Processed", f"{base_name}_processed.tsv")
    mappings.to_csv(output_path, sep='\t', index=False)
    print(f"Processed file saved as {output_path}")

#-----------------------------------------------------------------------------------------------------------------------

def main():
    # Get all mapping files
    mapping_files = [f for f in os.listdir('Data/Mappings') if f.endswith('.tsv')]

    # Preprocess each mapping file
    for file in mapping_files:
        preprocess_mappings(os.path.join('Data/Mappings', file))
        print(f'Preprocessed {file}')

if __name__ == '__main__':
    main()
