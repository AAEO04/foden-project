import pandas as pd
import numpy as np
import logging # Added for logging consistency

class FeatureEngineer:
    """Handles feature engineering for transformer fault diagnosis."""
    
    @staticmethod
    def calculate_gas_percentages_for_duval(df: pd.DataFrame) -> pd.DataFrame: # Renamed for clarity
        """
        Calculate gas percentages specifically for Duval triangle input.
        Creates columns: CH4_pct_Duval, C2H4_pct_Duval, C2H2_pct_Duval.

        Args:
            df (pd.DataFrame): Input DataFrame with Methane, Ethylene, Acethylene columns.

        Returns:
            pd.DataFrame: DataFrame with added Duval percentage columns.
        """
        gas_columns_duval = ['Methane', 'Ethylene', 'Acethylene']
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning if df is a slice

        df_copy['Total_Key_Gases_Duval'] = df_copy[gas_columns_duval].sum(axis=1)
        
        df_copy['CH4_pct_Duval'] = np.where(
            df_copy['Total_Key_Gases_Duval'] == 0, 
            0.0, 
            (df_copy['Methane'] / df_copy['Total_Key_Gases_Duval']) * 100
        )
        df_copy['C2H4_pct_Duval'] = np.where(
            df_copy['Total_Key_Gases_Duval'] == 0, 
            0.0, 
            (df_copy['Ethylene'] / df_copy['Total_Key_Gases_Duval']) * 100
        )
        df_copy['C2H2_pct_Duval'] = np.where(
            df_copy['Total_Key_Gases_Duval'] == 0, 
            0.0, 
            (df_copy['Acethylene'] / df_copy['Total_Key_Gases_Duval']) * 100
        )
        # Fill NaNs that might arise if original gas values were NaN
        duval_pct_cols = ['CH4_pct_Duval', 'C2H4_pct_Duval', 'C2H2_pct_Duval']
        df_copy[duval_pct_cols] = df_copy[duval_pct_cols].fillna(0.0)
        return df_copy

    @staticmethod
    def add_gas_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add standard DGA gas ratios.
        Ensures columns CH4_H2, C2H6_CH4, C2H4_C2H6, C2H2_C2H4 are created.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with added ratio columns.
        """
        df_copy = df.copy()
        ratios_map = {
            'CH4_H2': ('Methane', 'Hydrogen'),
            'C2H6_CH4': ('Ethane', 'Methane'),
            'C2H4_C2H6': ('Ethylene', 'Ethane'),
            'C2H2_C2H4': ('Acethylene', 'Ethylene')
        }
        for ratio_name, (num_gas, den_gas) in ratios_map.items():
            if num_gas in df_copy.columns and den_gas in df_copy.columns:
                denominator = df_copy[den_gas].replace(0, np.nan) # Avoid division by zero
                df_copy[ratio_name] = df_copy[num_gas] / denominator
            else:
                logging.warning(f"Columns for ratio {ratio_name} not found. Ratio column will be NaN or 0.")
                df_copy[ratio_name] = np.nan # Or 0.0, will be imputed later or filled
        
        # Replace inf/-inf (from x/0 if somehow not caught) with NaN, then fill NaNs for ratios with 0
        # This matches the notebook's approach. Consider if 0 is the best imputation for ratios.
        df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        ratio_cols = list(ratios_map.keys())
        df_copy[ratio_cols] = df_copy[ratio_cols].fillna(0.0)
        return df_copy

# --- Updated Diagnostic Logic Functions (from Notebook Cell 3 logic) ---
def duval_triangle_fault_type(ch4_pct: float, c2h4_pct: float, c2h2_pct: float) -> str:
    """
    Approximates fault type based on relative percentages of CH4, C2H4, C2H2.
    These percentages are assumed to be of the sum (CH4+C2H4+C2H2), normalized to 100%.
    This version aligns with the notebook's refined (though illustrative) logic.
    """
    if not ((-0.1 <= ch4_pct <= 100.1) and \
            (-0.1 <= c2h4_pct <= 100.1) and \
            (-0.1 <= c2h2_pct <= 100.1)):
        logging.warning(f"Invalid input percentages for Duval: CH4={ch4_pct}, C2H4={c2h4_pct}, C2H2={c2h2_pct}")
        return "Error_InvalidInput"
    if ch4_pct < 0.1 and c2h4_pct < 0.1 and c2h2_pct < 0.1:
        return "NF"
    if ch4_pct >= 98: return "PD"
    if c2h2_pct >= 29 and c2h4_pct < 20: return "D1"
    if c2h2_pct >= 29 and c2h4_pct >= 20 :return "D2"
    if c2h2_pct < 29 and c2h4_pct >= 50: return "T3"
    if c2h2_pct < 4 and 20 <= c2h4_pct < 50: return "T2"
    if c2h2_pct < 4 and c2h4_pct < 20: return "T1"
    if 4 <= c2h2_pct < 29 and c2h4_pct < 50: return "DT"
    return "NF_Undetermined"

def adjust_fault_label(duval_label: str, health_index: float) -> str:
    """
    Refines Duval labels using Health Index. Aligned with Notebook Cell 3 logic.
    """
    if duval_label == "Error_InvalidInput": return "Error_InvalidInput"
    if pd.isna(health_index): # Handle potential NaN in health_index
        logging.warning("Health index is NaN, cannot adjust fault label based on it.")
        return duval_label
        
    if health_index >= 85 and duval_label not in ["NF", "NF_Undetermined"]:
        return "NF"
    if health_index < 30:
        if duval_label in ["PD", "T1", "NF_Undetermined", "NF", "DT"]: return "D1"
        if duval_label == "T2": return "T3"
        return duval_label
    elif 30 <= health_index < 50:
        if duval_label in ["PD", "NF_Undetermined", "NF"]: return "T1"
        if duval_label == "T1": return "T2"
        if duval_label == "DT": return "D1"
        return duval_label
    return duval_label