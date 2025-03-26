"""
Module handling the preprocessing of GWAS data and Mapgma gene mapping.

Requires the following files:
Data/SQLite_DB/genomics.db - SQLite3 database containing GWAS and STRING data

Author: Luke Krongard
"""

#-----------------------------------------------------------------------------------------------------------------------

import sqlite3
import argparse
import os
import subprocess
from query import query_snps_by_trait

#-----------------------------------------------------------------------------------------------------------------------

DATABASE_PATH = 'Data/SQLite_DB/genomics.db'

#-----------------------------------------------------------------------------------------------------------------------

def create_snploc_from_trait(conn: sqlite3.Connection, trait: str) -> None:
    """
    Creates a SNP location file for Magma gene mapping from a given trait.
    """
    snps = query_snps_by_trait(conn, trait)
    with open('magma/' + trait + '.snploc', 'w') as file:   
        for snp in snps:
            if (not snp[0].startswith('rs') or 
                snp[1] is None or snp[2] is None or 
                snp[1] == 0 or snp[2] == 0 or
                not str(snp[1]).strip() or not str(snp[2]).strip()):
                continue
            file.write('\t'.join(map(str, snp[:3])) + '\n')

#-----------------------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess GWAS data and run Magma gene mapping')
    parser.add_argument('trait', type=str, help='Trait for analysis')
    parser.add_argument('window', type=int, help='Window size for gene mapping (in kb)')
    return parser.parse_args()

#-----------------------------------------------------------------------------------------------------------------------

def main():
    # Parse arguments
    args = parse_args()
    trait = args.trait
    window = args.window

    # Create SNP location files for Magma gene mapping
    conn = sqlite3.connect(DATABASE_PATH)
    create_snploc_from_trait(conn, trait)
    conn.close()

    # Run Magma gene mapping
    os.chdir('magma')
    command = f'''
    ./magma --annotate window={window} --snp-loc {trait}.snploc --gene-loc NCBI38.gene.loc --out {trait}
    '''
    subprocess.run(command, shell=True)
    os.chdir('..')

if __name__ == '__main__':
    main()