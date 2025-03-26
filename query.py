"""
Module handling interacting with the SQLite3 database built from GWAS and STRING data.

Requires the following files:
Data/SQLite_DB/genomics.db - SQLite3 database containing GWAS and STRING data

Author: Luke Krongard
"""

#-----------------------------------------------------------------------------------------------------------------------

import sqlite3
import os
from typing import List, Tuple, Dict

#-----------------------------------------------------------------------------------------------------------------------

DATABASE_PATH = 'Data/SQLite_DB/genomics.db'

#-----------------------------------------------------------------------------------------------------------------------

def query_snps_by_trait(conn: sqlite3.Connection, trait: str) -> List[Tuple]:
    """
    Queries the SQLite3 database for all SNPs associated with a given trait."
    """
    cursor = conn.cursor()
    cursor.execute('''
    SELECT s.rs_number, a.chr_id, a.chr_pos, a.p_value
    FROM Studies st
    JOIN Associations a ON st.study_id = a.study_id
    JOIN SNPs s ON a.snp_id = s.snp_id
    WHERE st.trait LIKE ?
    ORDER BY a.p_value ASC
    ''', ('%' + trait + '%',))
    return cursor.fetchall()

#-----------------------------------------------------------------------------------------------------------------------

def query_pvalue_by_snp(conn: sqlite3.Connection, rs_number: str) -> float:
    """
    Queries the SQLite3 database for the p-value of a given SNP.
    """
    cursor = conn.cursor()
    cursor.execute('''
    SELECT a.p_value
    FROM SNPs s
    JOIN Associations a ON s.snp_id = a.snp_id
    WHERE s.rs_number = ?
    ORDER BY a.p_value ASC
    ''', (rs_number,))
    return cursor.fetchone()[0]

#-----------------------------------------------------------------------------------------------------------------------

def query_pvalues_by_snps_batch(conn: sqlite3.Connection, snps: List[str]) -> Dict[str, float]:
    """
    Query p-values for multiple SNPs at once.
    Returns the most significant (smallest) p-value for each SNP.
    """
    result = {}
    placeholders = ','.join(['?' for _ in snps])
    
    # Modified query to select the minimum p-value for each SNP
    query = f"""
    SELECT s.rs_number, MIN(a.p_value)
    FROM SNPs s
    JOIN Associations a ON s.snp_id = a.snp_id
    WHERE s.rs_number IN ({placeholders})
    GROUP BY s.rs_number
    """
    cursor = conn.cursor()
    cursor.execute(query, snps)
    rows = cursor.fetchall()
    
    for snp, pvalue in rows:
        result[snp] = pvalue
    
    return result

#-----------------------------------------------------------------------------------------------------------------------

def query_protein_by_alias(conn: sqlite3.Connection, alias: str) -> List[Tuple]:
    """
    Queries the SQLite3 database for all proteins associated with a given alias.
    """
    cursor = conn.cursor()
    cursor.execute('''
    SELECT p.protein_id, p.preferred_name, p.protein_size, p.annotation
    FROM Proteins p
    JOIN ProteinAliases pa ON p.protein_id = pa.protein_id
    WHERE pa.alias LIKE ?
    ''', ('%' + alias + '%',))
    return cursor.fetchall()

#-----------------------------------------------------------------------------------------------------------------------

def query_proteins_by_aliases_batch(conn: sqlite3.Connection, gene_aliases: List[str]) -> Dict[str, List[str]]:
    """
    Query proteins by multiple gene aliases at once.
    """
    result = {}
    placeholders = ','.join(['?' for _ in gene_aliases])
    query = f"""
    SELECT alias, protein_id 
    FROM ProteinAliases 
    WHERE alias IN ({placeholders})
    """
    cursor = conn.cursor()
    cursor.execute(query, gene_aliases)
    rows = cursor.fetchall()
    
    for alias, protein_id in rows:
        if alias not in result:
            result[alias] = []
        result[alias].append(protein_id)
    
    return result

#-----------------------------------------------------------------------------------------------------------------------

def query_interactions_by_score(conn: sqlite3.Connection, min_score: int) -> List[Tuple]:
    """
    Queries the SQLite3 database for all protein-protein interactions with a score above a given threshold.
    """
    cursor = conn.cursor()
    cursor.execute('''
    SELECT protein1_id, protein2_id, combined_score
    FROM Interactions
    WHERE combined_score >= ?
    ''', (min_score,))
    return cursor.fetchall()