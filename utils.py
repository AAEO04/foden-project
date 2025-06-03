# utils.py
import pandas as pd
import numpy as np
from typing import Union

class FeatureEngineer:
    @staticmethod
    def calculate_gas_percentages_for_duval(df: pd.DataFrame) -> pd.DataFrame:
        gas_columns: list[str] = ['Methane', 'Ethylene', 'Acetylene']
        for col in gas_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
        
        total_gas: pd.Series = df[gas_columns].sum(axis=1)
        df['CH4_pct_Duval'] = np.where(total_gas > 0, df['Methane'] / total_gas * 100, 0.0)
        df['C2H4_pct_Duval'] = np.where(total_gas > 0, df['Ethylene'] / total_gas * 100, 0.0)
        df['C2H2_pct_Duval'] = np.where(total_gas > 0, df['Acetylene'] / total_gas * 100, 0.0)
        
        return df
    
    @staticmethod
    def add_gas_ratios(df: pd.DataFrame) -> pd.DataFrame:
        required_columns: list[str] = ['Hydrogen', 'Methane', 'Ethane', 'Ethylene', 'Acetylene']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
        
        df['CH4_H2'] = np.where(df['Hydrogen'] > 0, df['Methane'] / df['Hydrogen'], 0.0)
        df['C2H6_CH4'] = np.where(df['Methane'] > 0, df['Ethane'] / df['Methane'], 0.0)
        df['C2H4_C2H6'] = np.where(df['Ethane'] > 0, df['Ethylene'] / df['Ethane'], 0.0)
        df['C2H2_C2H4'] = np.where(df['Ethylene'] > 0, df['Acetylene'] / df['Ethylene'], 0.0)
        
        return df
    
    @staticmethod
    def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
        gas_columns: list[str] = ['Hydrogen', 'Methane', 'Ethylene', 'Acetylene', 'CO', 'CO2', 'Ethane']
        for col in gas_columns:
            if col in df.columns:
                df[f'log_{col}'] = np.where(df[col] > 0, np.log1p(df[col]), 0.0)
        return df

def duval_triangle_fault_type(ch4_pct: float, c2h4_pct: float, c2h2_pct: float) -> str:
    total: float = ch4_pct + c2h4_pct + c2h2_pct
    if total == 0:
        return 'NF'
    
    ch4_pct = (ch4_pct / total) * 100
    c2h4_pct = (c2h4_pct / total) * 100
    c2h2_pct = (c2h2_pct / total) * 100
    
    if c2h2_pct > 50:
        return 'D1'
    if c2h2_pct > 15 and c2h4_pct > 20:
        return 'D2'
    if ch4_pct > 80:
        return 'PD'
    if c2h4_pct > 50:
        if c2h2_pct < 4:
            return 'T3'
        if c2h2_pct < 15:
            return 'T2'
    if ch4_pct > 50 and c2h4_pct > 20:
        return 'T1'
    if c2h2_pct > 4 and c2h4_pct > 20:
        return 'DT'
    
    return 'NF'

def adjust_fault_label(fault_type: str, health_index: Union[float, int]) -> str:
    health_index_float: float = float(health_index) if pd.notnull(health_index) else 75.0
    
    if fault_type in ['PD', 'D1', 'D2', 'T1', 'T2', 'T3', 'DT']:
        if health_index_float > 85:
            return 'NF'
        elif health_index_float < 30:
            return fault_type
        else:
            return 'NF_Undetermined'
    return fault_type