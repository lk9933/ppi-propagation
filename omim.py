"""
Module which handles OMIM data processing.
"""

import pandas as pd
from typing import Dict, List
import os

def main():
    # Load OMIM data
    omim_file = "Data/OMIM/morbidmap.tsv"
    omim_df = pd.read_csv(omim_file, sep='\t')
    
    # Set list of disease terms
    epilepsy_terms = [
        "epilepsy",
        "epileptic",
        "epileptical"
    ]
    als_terms = [
        "amyotrophic lateral sclerosis",
    ]
    amd_terms = [
        "age-related macular degeneration",
        "Macular degeneration, age-related",
        "macular degeneration",
        "AMD"
    ]
    psoriasis_terms = [
        "psoriasis",
        "psoriatic arthritis",
        "psoriatic arthropathy",
        "psoriatic",
        "psoriatic disease",
        "psoriatic skin disease",
        "psoriatic skin",
        "psoriatic skin lesions",
        "psoriatic skin manifestations",
    ]
    diabetes_terms = [
        "diabetes",
        "diabetic",
        "diabetes mellitus",
        "type 1, diabetes mellitus",
        "type 2, diabetes mellitus",
    ]

    # Filter for epilepsy
    epilepsy_df = omim_df[omim_df['Phenotype'].str.contains('|'.join(epilepsy_terms), case=False, na=False)]

    # Filter for ALS
    als_df = omim_df[omim_df['Phenotype'].str.contains('|'.join(als_terms), case=False, na=False)]

    # Filter for AMD
    amd_df = omim_df[omim_df['Phenotype'].str.contains('|'.join(amd_terms), case=False, na=False)]

    # Filter for Psoriasis
    psoriasis_df = omim_df[omim_df['Phenotype'].str.contains('|'.join(psoriasis_terms), case=False, na=False)]

    # Filter for Diabetes
    diabetes_df = omim_df[omim_df['Phenotype'].str.contains('|'.join(diabetes_terms), case=False, na=False)]

    # Rename "Gene/Locus And Other Related Symbols" to "Gene"
    epilepsy_df = epilepsy_df.rename(columns={"Gene/Locus And Other Related Symbols": "Gene"})
    als_df = als_df.rename(columns={"Gene/Locus And Other Related Symbols": "Gene"})
    amd_df = amd_df.rename(columns={"Gene/Locus And Other Related Symbols": "Gene"})
    psoriasis_df = psoriasis_df.rename(columns={"Gene/Locus And Other Related Symbols": "Gene"})
    diabetes_df = diabetes_df.rename(columns={"Gene/Locus And Other Related Symbols": "Gene"})

    # Drop unnecessary columns
    epilepsy_df = epilepsy_df.drop(columns=['MIM Number', 'Cyto Location'])
    als_df = als_df.drop(columns=['MIM Number', 'Cyto Location'])
    amd_df = amd_df.drop(columns=['MIM Number', 'Cyto Location'])
    psoriasis_df = psoriasis_df.drop(columns=['MIM Number', 'Cyto Location'])
    diabetes_df = diabetes_df.drop(columns=['MIM Number', 'Cyto Location'])

    # Parse the comma separated gene symbols
    epilepsy_df['Gene'] = epilepsy_df['Gene'].apply(lambda x: [gene.strip() for gene in x.split(',')])
    als_df['Gene'] = als_df['Gene'].apply(lambda x: [gene.strip() for gene in x.split(',')])
    amd_df['Gene'] = amd_df['Gene'].apply(lambda x: [gene.strip() for gene in x.split(',')])
    psoriasis_df['Gene'] = psoriasis_df['Gene'].apply(lambda x: [gene.strip() for gene in x.split(',')])
    diabetes_df['Gene'] = diabetes_df['Gene'].apply(lambda x: [gene.strip() for gene in x.split(',')])

    # Explode the DataFrame to have one gene per row
    epilepsy_df = epilepsy_df.explode('Gene')
    als_df = als_df.explode('Gene')
    amd_df = amd_df.explode('Gene')
    psoriasis_df = psoriasis_df.explode('Gene')
    diabetes_df = diabetes_df.explode('Gene')

    # Remove duplicates
    epilepsy_df = epilepsy_df.drop_duplicates(subset=['Gene'])
    als_df = als_df.drop_duplicates(subset=['Gene'])
    amd_df = amd_df.drop_duplicates(subset=['Gene'])
    psoriasis_df = psoriasis_df.drop_duplicates(subset=['Gene'])
    diabetes_df = diabetes_df.drop_duplicates(subset=['Gene'])

    # Save as a list of genes
    epilepsy_genes = list(epilepsy_df['Gene'].unique())
    als_genes = list(als_df['Gene'].unique())
    amd_genes = list(amd_df['Gene'].unique())
    psoriasis_genes = list(psoriasis_df['Gene'].unique())
    diabetes_genes = list(diabetes_df['Gene'].unique())

    # Save the lists to files
    output_dir = "Data/OMIM/Processed"
    os.makedirs(output_dir, exist_ok=True)
    epilepsy_output_file = os.path.join(output_dir, "epilepsy_genes.txt")
    als_output_file = os.path.join(output_dir, "als_genes.txt")
    amd_output_file = os.path.join(output_dir, "amd_genes.txt")
    psoriasis_output_file = os.path.join(output_dir, "psoriasis_genes.txt")
    diabetes_output_file = os.path.join(output_dir, "diabetes_genes.txt")
    with open(epilepsy_output_file, 'w') as f:
        for gene in epilepsy_genes:
            f.write(f"{gene}\n")
    with open(als_output_file, 'w') as f:
        for gene in als_genes:
            f.write(f"{gene}\n")
    with open(amd_output_file, 'w') as f:
        for gene in amd_genes:
            f.write(f"{gene}\n")
    with open(psoriasis_output_file, 'w') as f:
        for gene in psoriasis_genes:
            f.write(f"{gene}\n")
    with open(diabetes_output_file, 'w') as f:
        for gene in diabetes_genes:
            f.write(f"{gene}\n")
    print(f"Epilepsy genes saved to {epilepsy_output_file}")
    print(f"ALS genes saved to {als_output_file}")
    print(f"AMD genes saved to {amd_output_file}")
    print(f"Psoriasis genes saved to {psoriasis_output_file}")
    print(f"Diabetes genes saved to {diabetes_output_file}")

if __name__ == "__main__":
    main()


