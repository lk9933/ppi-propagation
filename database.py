"""
Module handling the creation of the SQLite3 database from GWAS and STRING data.

Requires the following files:
Data/GWAS/*.tsv - GWAS data from NHGRI-EBI GWAS Catalog
Data/STRING/aliases.tsv - STRING protein aliases
Data/STRING/info.tsv - STRING protein information
Data/STRING/interactions.tsv - STRING protein interactions
Data/STRING/cluster_info.tsv - STRING cluster information
Data/STRING/cluster_proteins.tsv - STRING cluster proteins

Creates the following database:
Data/SQLite_DB/genomics.db - SQLite3 database containing GWAS and STRING data

Author: Luke Krongard
"""

#-----------------------------------------------------------------------------------------------------------------------

import sqlite3
import os
import csv

#-----------------------------------------------------------------------------------------------------------------------

def create_database(db_name: str = 'genomics.db') -> None:
    """
    Creates the SQLite3 database containing GWAS and STRING data.
    """
    # Delete the database if it already exists
    if os.path.exists(f'Data/SQLite_DB/{db_name}'):
        os.remove(f'Data/SQLite_DB/{db_name}')

    # Connect to the database
    conn = sqlite3.connect(f'Data/SQLite_DB/{db_name}')
    cursor = conn.cursor()

    # Create the GWAS Studies table
    cursor.execute('''
    CREATE TABLE Studies (
        study_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pubmed_id TEXT,
        first_author TEXT,
        publication_date TEXT,
        journal TEXT,
        link TEXT,
        title TEXT,
        trait TEXT,
        initial_sample_size TEXT,
        replication_sample_size TEXT,
        mapped_trait TEXT,
        mapped_trait_uri TEXT,
        study_accession TEXT,
        genotyping_technology TEXT,
        date_added TEXT
    )
    ''')

    # Create the GWAS SNPs table
    cursor.execute('''
    CREATE TABLE SNPs (
        snp_id INTEGER PRIMARY KEY AUTOINCREMENT,
        rs_number TEXT,
        merged_status INTEGER,
        current_rs_number TEXT,
        context TEXT,
        intergenic INTEGER
    )
    ''')

    # Create the GWAS Associations table
    cursor.execute('''
    CREATE TABLE Associations (
        association_id INTEGER PRIMARY KEY AUTOINCREMENT,
        study_id INTEGER,
        snp_id INTEGER,
        chr_id TEXT,
        chr_pos INTEGER,
        reported_genes TEXT,
        mapped_genes TEXT,
        upstream_gene_id TEXT,
        downstream_gene_id TEXT,
        snp_gene_ids TEXT,
        upstream_gene_distance REAL,
        downstream_gene_distance REAL,
        risk_allele TEXT,
        risk_allele_frequency REAL,
        p_value REAL,
        pvalue_mlog REAL,
        p_value_text TEXT,
        or_beta TEXT,
        ci_text TEXT,
        region TEXT,
        platform TEXT,
        cnv TEXT,
        FOREIGN KEY (study_id) REFERENCES Studies(study_id),
        FOREIGN KEY (snp_id) REFERENCES SNPs(snp_id)
    )
    ''')

    # Create the STRING Proteins table
    cursor.execute('''
    CREATE TABLE Proteins (
        protein_id TEXT PRIMARY KEY,
        preferred_name TEXT,
        protein_size INTEGER,
        annotation TEXT
    )
    ''')

    # Create the STRING ProteinAliases table
    cursor.execute('''
    CREATE TABLE ProteinAliases (
        alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
        protein_id TEXT,
        alias TEXT,
        source TEXT,
        FOREIGN KEY (protein_id) REFERENCES Proteins(protein_id)
    )
    ''')

    # Create the STRING Clusters table
    cursor.execute('''
    CREATE TABLE Clusters (
        cluster_id TEXT,
        string_taxon_id INTEGER,
        cluster_size INTEGER,
        best_described_by TEXT,
        PRIMARY KEY (cluster_id, string_taxon_id)
    )
    ''')

    # Create the STRING ClusterProteins table
    cursor.execute('''
    CREATE TABLE ClusterProteins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cluster_id TEXT,
        string_taxon_id INTEGER,
        protein_id TEXT,
        FOREIGN KEY (cluster_id, string_taxon_id) REFERENCES Clusters(cluster_id, string_taxon_id),
        FOREIGN KEY (protein_id) REFERENCES Proteins(protein_id)
    )
    ''')

    # Create the STRING Interactions table
    cursor.execute('''
    CREATE TABLE Interactions (
        interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        protein1_id TEXT,
        protein2_id TEXT,
        combined_score REAL,
        FOREIGN KEY (protein1_id) REFERENCES Proteins(protein_id),
        FOREIGN KEY (protein2_id) REFERENCES Proteins(protein_id)
    )
    ''')

     # GWAS indexes
    cursor.execute('CREATE INDEX idx_studies_trait ON Studies(trait)')
    cursor.execute('CREATE INDEX idx_snps_rs_number ON SNPs(rs_number)')
    cursor.execute('CREATE INDEX idx_associations_snp_id ON Associations(snp_id)')
    cursor.execute('CREATE INDEX idx_associations_study_id ON Associations(study_id)')
    
    # PPI indexes
    cursor.execute('CREATE INDEX idx_proteins_protein ON Proteins(protein_id)')
    cursor.execute('CREATE INDEX idx_protein_alias ON ProteinAliases(alias)')
    cursor.execute('CREATE INDEX idx_interactions_protein1 ON Interactions(protein1_id)')
    cursor.execute('CREATE INDEX idx_interactions_protein2 ON Interactions(protein2_id)')
    cursor.execute('CREATE INDEX idx_cluster_proteins_protein ON ClusterProteins(protein_id)')
    cursor.execute('CREATE INDEX idx_cluster_proteins_cluster ON ClusterProteins(cluster_id, string_taxon_id)')

    # Commit changes and close the connection
    conn.commit()
    conn.close()

#-----------------------------------------------------------------------------------------------------------------------

def load_gwas_files(conn: sqlite3.Connection, gwas_files: list) -> None:
    """
    Loads the GWAS data from the given files into the database.
    """
    cursor = conn.cursor()

    # Go through the files and load the data
    for file_path in gwas_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Insert into Studies table
                cursor.execute('''
                INSERT INTO Studies (
                    pubmed_id, first_author, publication_date, journal, link, title, trait,
                    initial_sample_size, replication_sample_size, mapped_trait, mapped_trait_uri,
                    study_accession, genotyping_technology, date_added
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row.get('PUBMEDID', ''),
                    row.get('FIRST AUTHOR', ''),
                    row.get('DATE', ''),
                    row.get('JOURNAL', ''),
                    row.get('LINK', ''),
                    row.get('STUDY', ''),
                    row.get('DISEASE/TRAIT', ''),
                    row.get('INITIAL SAMPLE SIZE', ''),
                    row.get('REPLICATION SAMPLE SIZE', ''),
                    row.get('MAPPED_TRAIT', ''),
                    row.get('MAPPED_TRAIT_URI', ''),
                    row.get('STUDY ACCESSION', ''),
                    row.get('GENOTYPING TECHNOLOGY', ''),
                    row.get('DATE ADDED TO CATALOG', '')
                ))
                study_id = cursor.lastrowid
                
                # Insert into SNPs table
                cursor.execute('''
                INSERT INTO SNPs (
                    rs_number, merged_status, current_rs_number, context, intergenic
                ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    row.get('SNPS', ''),
                    int(row.get('MERGED', 0)) if row.get('MERGED', '').isdigit() else 0,
                    row.get('SNP_ID_CURRENT', ''),
                    row.get('CONTEXT', ''),
                    int(row.get('INTERGENIC', 0)) if row.get('INTERGENIC', '').isdigit() else 0
                ))
                snp_id = cursor.lastrowid
                
                # Insert into Associations table
                try:
                    p_value = float(row.get('P-VALUE', 0))
                except (ValueError, TypeError):
                    p_value = 0.0
                    
                try:
                    pvalue_mlog = float(row.get('PVALUE_MLOG', 0))
                except (ValueError, TypeError):
                    pvalue_mlog = 0.0
                    
                try:
                    risk_allele_frequency = float(row.get('RISK ALLELE FREQUENCY', 0))
                except (ValueError, TypeError):
                    risk_allele_frequency = 0.0
                    
                try:
                    upstream_distance = float(row.get('UPSTREAM_GENE_DISTANCE', 0))
                except (ValueError, TypeError):
                    upstream_distance = 0.0
                    
                try:
                    downstream_distance = float(row.get('DOWNSTREAM_GENE_DISTANCE', 0))
                except (ValueError, TypeError):
                    downstream_distance = 0.0
                
                cursor.execute('''
                INSERT INTO Associations (
                    study_id, snp_id, chr_id, chr_pos, reported_genes, mapped_genes,
                    upstream_gene_id, downstream_gene_id, snp_gene_ids,
                    upstream_gene_distance, downstream_gene_distance, risk_allele,
                    risk_allele_frequency, p_value, pvalue_mlog, p_value_text,
                    or_beta, ci_text, region, platform, cnv
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    study_id,
                    snp_id,
                    row.get('CHR_ID', ''),
                    int(row.get('CHR_POS', 0)) if row.get('CHR_POS', '').isdigit() else 0,
                    row.get('REPORTED GENE(S)', ''),
                    row.get('MAPPED_GENE', ''),
                    row.get('UPSTREAM_GENE_ID', ''),
                    row.get('DOWNSTREAM_GENE_ID', ''),
                    row.get('SNP_GENE_IDS', ''),
                    upstream_distance,
                    downstream_distance,
                    row.get('STRONGEST SNP-RISK ALLELE', ''),
                    risk_allele_frequency,
                    p_value,
                    pvalue_mlog,
                    row.get('P-VALUE (TEXT)', ''),
                    row.get('OR or BETA', ''),
                    row.get('95% CI (TEXT)', ''),
                    row.get('REGION', ''),
                    row.get('PLATFORM [SNPS PASSING QC]', ''),
                    row.get('CNV', '')
                ))
    
    conn.commit()
    print("GWAS data loading complete.")

#-----------------------------------------------------------------------------------------------------------------------

def load_string_files(conn: sqlite3.Connection, ppi_files: list) -> None:
    """
    Loads the STRING data from the given files into the database.
    """
    
    cursor = conn.cursor()
    
    # Define the expected file names
    info_file = None
    aliases_file = None
    cluster_info_file = None
    cluster_proteins_file = None
    interactions_file = None
    
    for file_path in ppi_files:
        file_name = os.path.basename(file_path)
        if file_name == 'info.tsv':
            info_file = file_path
        elif file_name == 'aliases.tsv':
            aliases_file = file_path
        elif file_name == 'cluster_info.tsv':
            cluster_info_file = file_path
        elif file_name == 'cluster_proteins.tsv':
            cluster_proteins_file = file_path
        elif file_name == 'interactions.tsv':
            interactions_file = file_path
    
    # Load Proteins table
    if info_file:
        print(f"Loading protein info from {info_file}...")
        with open(info_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                try:
                    protein_size = int(row.get('protein_size', 0))
                except (ValueError, TypeError):
                    protein_size = 0
                
                cursor.execute('''
                INSERT OR IGNORE INTO Proteins (
                    protein_id, preferred_name, protein_size, annotation
                ) VALUES (?, ?, ?, ?)
                ''', (
                    row.get('protein_id', ''),
                    row.get('preferred_name', ''),
                    protein_size,
                    row.get('annotation', '')
                ))
    
    # Load ProteinAliases table
    if aliases_file:
        print(f"Loading protein aliases from {aliases_file}...")
        with open(aliases_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                cursor.execute('''
                INSERT INTO ProteinAliases (
                    protein_id, alias, source
                ) VALUES (?, ?, ?)
                ''', (
                    row.get('protein_id', ''),
                    row.get('alias', ''),
                    row.get('source', '')
                ))
    
    # Load Clusters table
    if cluster_info_file:
        print(f"Loading cluster info from {cluster_info_file}...")
        with open(cluster_info_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                try:
                    string_taxon_id = int(row.get('string_taxon_id', 0))
                except (ValueError, TypeError):
                    string_taxon_id = 0
                
                try:
                    cluster_size = int(row.get('cluster_size', 0))
                except (ValueError, TypeError):
                    cluster_size = 0
                
                cursor.execute('''
                INSERT OR IGNORE INTO Clusters (
                    string_taxon_id, cluster_id, cluster_size, best_described_by
                ) VALUES (?, ?, ?, ?)
                ''', (
                    string_taxon_id,
                    row.get('cluster_id', ''),
                    cluster_size,
                    row.get('best_described_by', '')
                ))
    
    # Load ClusterProteins table
    if cluster_proteins_file:
        print(f"Loading cluster proteins from {cluster_proteins_file}...")
        with open(cluster_proteins_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                try:
                    string_taxon_id = int(row.get('string_taxon_id', 0))
                except (ValueError, TypeError):
                    string_taxon_id = 0
                
                cursor.execute('''
                INSERT INTO ClusterProteins (
                    string_taxon_id, cluster_id, protein_id
                ) VALUES (?, ?, ?)
                ''', (
                    string_taxon_id,
                    row.get('cluster_id', ''),
                    row.get('protein_id', '')
                ))
    
    # Load Interactions table
    if interactions_file:
        print(f"Loading protein interactions from {interactions_file}...")
        with open(interactions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                try:
                    combined_score = float(row.get('combined_score', 0))
                except (ValueError, TypeError):
                    combined_score = 0.0
                
                cursor.execute('''
                INSERT INTO Interactions (
                    protein1_id, protein2_id, combined_score
                ) VALUES (?, ?, ?)
                ''', (
                    row.get('protein1', ''),
                    row.get('protein2', ''),
                    combined_score
                ))
    
    conn.commit()
    print("PPI data loading complete.")

#-----------------------------------------------------------------------------------------------------------------------

def main():
    # Create the database
    create_database()
    
    # Connect to the database
    conn = sqlite3.connect(f'Data/SQLite_DB/genomics.db')

    # Provide folder paths
    gwas_dir = 'Data/GWAS/'
    string_dir = 'Data/STRING/'

    # Provide paths to your GWAS files
    gwas_files = [
        gwas_dir + 'gwas_alzheimers.tsv',
        gwas_dir + 'gwas_diabetes.tsv',
        gwas_dir + 'gwas_psoraisis.tsv',
        gwas_dir + 'gwas_sud.tsv'
    ]
    
    # Provide paths to your PPI files
    ppi_files = [
        string_dir + 'aliases.tsv',
        string_dir + 'cluster_info.tsv',
        string_dir + 'cluster_proteins.tsv', 
        string_dir + 'info.tsv',
        string_dir + 'interactions.tsv'
    ]
    
    # Load the data
    load_gwas_files(conn, gwas_files)
    load_string_files(conn, ppi_files)
    
    # Close the connection
    conn.close()
    
    print("Database creation and data loading completed successfully.")

if __name__ == "__main__":
    main()