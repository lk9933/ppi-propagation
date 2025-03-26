"""
Module handling the combination and mapping of SNPs to genes to proteins.
"""

#-----------------------------------------------------------------------------------------------------------------------

import sqlite3
from typing import Dict, List, Tuple
from query import query_proteins_by_aliases_batch, query_pvalues_by_snps_batch

#-----------------------------------------------------------------------------------------------------------------------

DATABASE_PATH = 'Data/SQLite_DB/genomics.db'

#-----------------------------------------------------------------------------------------------------------------------

def load_magma_annotations(file_path: str) -> Dict[str, List[str]]:
    """
    Loads Magma gene annotations from a file.
    """
    annotations = {}
    with open(file_path, 'r') as file:
        # Skip the first two lines (header)
        next(file)
        next(file)
        for line in file:             
            parts = line.strip().split('\t')
            gene = parts[0]
            snps = parts[2:]
            if gene not in annotations:
                annotations[gene] = []
            annotations[gene].extend(snps)
    return annotations

#-----------------------------------------------------------------------------------------------------------------------

def map_snps_to_proteins(conn: sqlite3.Connection, annotations: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Maps SNPs to proteins using Magma gene annotations with batch processing.
    Returns a dictionary where:
        - keys are SNP ids
        - values are dictionaries with:
            - 'proteins': list of protein ids
            - 'genes': list of gene ids that connect the SNP to each protein
    """
    print('Mapping SNPs to proteins...')
    mappings = {}
    # Get all gene names at once
    all_genes = list(annotations.keys())
    
    # Create batches of genes
    batch_size = 100
    for i in range(0, len(all_genes), batch_size):
        print(f'Processing batch {i + 1} to {min(i + batch_size, len(all_genes))}...')
        gene_batch = all_genes[i:i+batch_size]
        # Query proteins for multiple genes at once
        gene_to_proteins = query_proteins_by_aliases_batch(conn, gene_batch)
        
        # Process the batch results
        for gene, proteins in gene_to_proteins.items():
            for snp in annotations[gene]:
                if snp not in mappings:
                    mappings[snp] = {'proteins': [], 'genes': []}
                
                # For each protein, store both the protein ID and the gene that connected to it
                for protein in proteins:
                    if protein not in mappings[snp]['proteins']:
                        mappings[snp]['proteins'].append(protein)
                        mappings[snp]['genes'].append(gene)
    
    print('Mapping complete.')
    return mappings

#-----------------------------------------------------------------------------------------------------------------------

def assign_pvalues_to_proteins(conn: sqlite3.Connection, mappings: Dict[str, Dict[str, List[str]]]) -> List[Tuple[str, float, str, str]]:
    """
    Assigns p-values to proteins based on the SNPs they are associated with.
    For each protein, only keeps the most significant (smallest) p-value.
    Returns a list of tuples with (protein_id, p_value, gene_id, snp_id).
    """
    print('Assigning p-values to proteins...')
    
    # First, get the most significant p-value for each SNP
    snp_best_pvalues = {}  # {snp_id: p_value}
    all_snps = list(mappings.keys())
    batch_size = 100
    
    for i in range(0, len(all_snps), batch_size):
        print(f'Processing SNP batch {i + 1} to {min(i + batch_size, len(all_snps))}...')
        snp_batch = all_snps[i:i+batch_size]
        
        # Query p-values for multiple SNPs at once
        snp_to_pvalues = query_pvalues_by_snps_batch(conn, snp_batch)
        
        # For each SNP, keep only the smallest p-value
        for snp, pvalue in snp_to_pvalues.items():
            if pvalue is not None:
                # Convert p-value to float if it's a string
                if isinstance(pvalue, str):
                    try:
                        pvalue = float(pvalue)
                    except ValueError:
                        print(f"Warning: Could not convert p-value '{pvalue}' for SNP {snp} to float")
                        continue
                
                if snp not in snp_best_pvalues or pvalue < snp_best_pvalues[snp]:
                    snp_best_pvalues[snp] = pvalue
    
    print(f'Found most significant p-values for {len(snp_best_pvalues)} SNPs')
    
    # Next, for each protein, find the SNP with the most significant p-value
    protein_best_pvalues = {}  # {protein_id: (p_value, gene_id, snp_id)}
    
    for snp in snp_best_pvalues:
        if snp in mappings:
            pvalue = snp_best_pvalues[snp]
            for idx, protein in enumerate(mappings[snp]['proteins']):
                # Ensure we don't go out of bounds
                if idx < len(mappings[snp]['genes']):
                    gene = mappings[snp]['genes'][idx]
                    
                    # Only keep the most significant p-value for each protein
                    if protein not in protein_best_pvalues or pvalue < protein_best_pvalues[protein][0]:
                        protein_best_pvalues[protein] = (pvalue, gene, snp)
    
    # Convert the dictionary to a list of tuples
    results = [(protein, pvalue, gene, snp) 
               for protein, (pvalue, gene, snp) in protein_best_pvalues.items()]
    
    print('P-value assignment complete.')
    # Sort results by p-value in ascending order
    results.sort(key=lambda x: x[1])
    return results

#-----------------------------------------------------------------------------------------------------------------------

def create_mappings(conn: sqlite3.Connection, file_path: str) -> None:
    # Load Magma gene annotations
    annotations = load_magma_annotations(file_path)

    # Map SNPs to proteins and genes
    mappings = map_snps_to_proteins(conn, annotations)
    
    # Assign p-values to proteins and get sorted results
    results = assign_pvalues_to_proteins(conn, mappings)
    
    # Save results as TSV file with all four columns
    file_name = file_path.split('/')[-1].replace('.genes.annot', '')
    with open(f'Data/Mappings/{file_name}_mappings.tsv', 'w') as file:
        file.write('Protein\tP-Value\tGene\tSNP\n')
        for protein, pvalue, gene, snp in results:
            file.write(f'{protein}\t{pvalue}\t{gene}\t{snp}\n')
    
    print(f'Results saved to {file_path}.tsv with {len(results)} entries')

#-----------------------------------------------------------------------------------------------------------------------

def main():
    # Connect to database and optimize performance
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA journal_mode = WAL")  # Use Write-Ahead Logging
    conn.execute("PRAGMA synchronous = NORMAL")  # Reduce synchronization
    conn.execute("PRAGMA cache_size = -1000000")  # Use 1GB of memory for cache

    # List of file paths to process
    file_paths = [
        'magma/alzheimer.genes.annot',
        'magma/diabetes.genes.annot',
        'magma/psoriasis.genes.annot'
    ]

    # Create mappings for each file
    for file_path in file_paths:
        create_mappings(conn, file_path)

    # Close the connection
    conn.close()

if __name__ == '__main__':
    main()