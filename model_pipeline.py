import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import logging
import unittest # Keep for tests
from typing import Dict, Tuple, Optional, List 
# Ensure your utils.py has the updated functions
from utils import FeatureEngineer, duval_triangle_fault_type, adjust_fault_label
from sklearn.preprocessing import StandardScaler, LabelEncoder # For isinstance checks

# Configure logging
logging.basicConfig(
    filename='app_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w', force=True
)

# Define DEVICE globally in this module
DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Pipeline using device: {DEVICE}")

# --- Model Class (FaultClassifier - assuming it's the same as notebook's final version) ---
class FaultClassifier(nn.Module):
    def __init__(self, input_size: int, hidden1_units: int, hidden2_units: int, 
                 num_classes: int, dropout_rate: float):
        super(FaultClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1_units, hidden2_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden2_units, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# --- Load Configuration and Artifacts ---
def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

CONFIG_PATH = 'model_config_v1.json' 
config = load_config(CONFIG_PATH) 

def load_artifacts(model_path: str, scaler_path: str, label_encoder_path: str, 
                   app_config: Dict) -> Tuple[Optional[FaultClassifier], Optional[StandardScaler], Optional[LabelEncoder]]: # More specific types
    model: Optional[FaultClassifier] = None # Specific type
    scaler: Optional[StandardScaler] = None # Specific type
    label_encoder: Optional[LabelEncoder] = None # Specific type
    
    try:
        model_params = app_config['model']
        model = FaultClassifier(
            input_size=model_params['input_size'],
            hidden1_units=model_params['hidden1_units'],
            hidden2_units=model_params['hidden2_units'],
            num_classes=model_params['num_classes'], 
            dropout_rate=model_params['dropout_rate']
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True)) # Use global DEVICE
        model.to(DEVICE) 
        model.eval()
        logging.info(f"Model loaded from {model_path} and set to eval mode on {DEVICE}.")
    except FileNotFoundError:
        logging.error(f"Model file {model_path} not found")
    except KeyError as e:
        logging.error(f"KeyError in model_params: {e}. Check config file structure.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

    try:
        loaded_scaler_obj = joblib.load(scaler_path)
        if isinstance(loaded_scaler_obj, StandardScaler):
            scaler = loaded_scaler_obj
            logging.info(f"Scaler loaded from {scaler_path}")
        else:
            logging.error(f"Loaded scaler from {scaler_path} is not a StandardScaler instance.")
    except FileNotFoundError:
        logging.error(f"Scaler file {scaler_path} not found")
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")

    try:
        loaded_le_obj = joblib.load(label_encoder_path)
        if isinstance(loaded_le_obj, LabelEncoder):
            label_encoder = loaded_le_obj
            logging.info(f"Label encoder loaded from {label_encoder_path}")
        else:
            logging.error(f"Loaded label encoder from {label_encoder_path} is not a LabelEncoder instance.")
    except FileNotFoundError:
        logging.error(f"Label encoder file {label_encoder_path} not found")
    except Exception as e:
        logging.error(f"Error loading label encoder: {e}")

    return model, scaler, label_encoder

MODEL_ARTIFACT_PATH = 'best_fault_analysis_model.pt' 
SCALER_ARTIFACT_PATH = 'final_model_scaler.gz'     
LABEL_ENCODER_ARTIFACT_PATH = 'label_encoder_final.gz' 

loaded_model, loaded_scaler, loaded_label_encoder = load_artifacts(
    MODEL_ARTIFACT_PATH, SCALER_ARTIFACT_PATH, LABEL_ENCODER_ARTIFACT_PATH, config
)

FEATURE_RANGES = {
    'Hydrogen': (0, 25000), 'Oxigen': (0, 50000), 'Nitrogen': (0, 150000),
    'Methane': (0, 25000), 'CO': (0, 10000), 'CO2': (0, 50000),
    'Ethylene': (0, 25000), 'Ethane': (0, 10000), 'Acethylene': (0, 10000),
    'DBDS': (0, 1000), 'Power factor': (0, 100), 'Interfacial V': (0, 100),
    'Dielectric rigidity': (0, 100), 'Water content': (0, 1000),
    'Health index': (0, 100), 'Life expectation': (0, 60)
}

def validate_input(raw_input_data: Dict, required_features: List[str]) -> Tuple[bool, str]:
    missing_features = [feature for feature in required_features if feature not in raw_input_data or pd.isna(raw_input_data[feature])]
    if missing_features:
        return False, f"Missing or NaN value for required features: {', '.join(missing_features)}"

    for feature in required_features: 
        value_str = str(raw_input_data.get(feature))
        try:
            value = float(value_str)
            if feature in FEATURE_RANGES: 
                min_val, max_val = FEATURE_RANGES[feature]
                if not (min_val <= value <= max_val):
                    return False, f"'{feature}' value {value} is outside the typical range [{min_val}-{max_val}]."
        except ValueError:
            return False, f"Invalid non-numeric value '{raw_input_data.get(feature)}' for feature '{feature}'." # Use .get()
    return True, ""

def preprocess_input_for_prediction(
    raw_input_data: Dict, 
    config_data: Dict, 
    scaler_object: Optional[StandardScaler] # More specific type
) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[float], Optional[float], Optional[float]]:
    base_features = config_data['features']['base_input_features']
    model_features_order = config_data['features']['model_features_order']

    is_valid, error_msg = validate_input(raw_input_data, base_features)
    if not is_valid:
        logging.error(f"Input validation failed: {error_msg}")
        raise ValueError(error_msg) 

    df_input = pd.DataFrame([raw_input_data])

    for col in base_features:
        if col not in df_input.columns:
            df_input[col] = 0.0 
            logging.warning(f"Base feature {col} was missing and defaulted to 0.0")

    df_with_duval_pct = FeatureEngineer.calculate_gas_percentages_for_duval(df_input.copy())

    ch4_pct_duval = df_with_duval_pct['CH4_pct_Duval'].iloc[0] if 'CH4_pct_Duval' in df_with_duval_pct else 0.0
    c2h4_pct_duval = df_with_duval_pct['C2H4_pct_Duval'].iloc[0] if 'C2H4_pct_Duval' in df_with_duval_pct else 0.0
    c2h2_pct_duval = df_with_duval_pct['C2H2_pct_Duval'].iloc[0] if 'C2H2_pct_Duval' in df_with_duval_pct else 0.0
    
    # Ensure health_index is float, default if missing for some reason (should be caught by validate_input)
    health_index_val = df_input['Health index'].iloc[0] if 'Health index' in df_input and pd.notna(df_input['Health index'].iloc[0]) else None


    df_with_ratios = FeatureEngineer.add_gas_ratios(df_input.copy())

    df_for_model = pd.DataFrame(index=df_input.index)
    for feature in model_features_order:
        if feature in df_with_ratios.columns:
            df_for_model[feature] = df_with_ratios[feature]
        elif feature in df_input.columns: 
            df_for_model[feature] = df_input[feature]
        else:
            logging.warning(f"Model feature '{feature}' not found in input or engineered features. Defaulting to 0.")
            df_for_model[feature] = 0.0
            
    df_processed = df_for_model[model_features_order].fillna(0.0) 
    
    df_scaled: Optional[pd.DataFrame] = None # Initialize to handle Optional return
    if scaler_object is not None and isinstance(scaler_object, StandardScaler): # Pylance Fix: Check type
        try:
            df_scaled_values = scaler_object.transform(df_processed)
            df_scaled = pd.DataFrame(df_scaled_values, columns=model_features_order)
        except Exception as e:
            logging.error(f"Error during scaling: {e}. Using unscaled data as fallback.")
            df_scaled = df_processed.copy() # Fallback to unscaled but processed
    elif scaler_object is None:
        logging.warning("Scaler not loaded, using unscaled data for prediction.")
        df_scaled = df_processed.copy() # Use unscaled but processed
    else: # Should not happen if load_artifacts ensures StandardScaler type
        logging.error("Scaler object is not a StandardScaler instance. Using unscaled data.")
        df_scaled = df_processed.copy()


    return df_scaled, health_index_val, ch4_pct_duval, c2h4_pct_duval, c2h2_pct_duval


# --- Prediction ---
def predict_fault(raw_input_data: Dict) -> Dict:
    global loaded_model, loaded_scaler, loaded_label_encoder, config, DEVICE # Added DEVICE

    if not loaded_model:
        logging.error("Model not loaded at prediction time.")
        return {"status": "error", "prediction_ml": None, "raw_duval_diagnosis": None, "adjusted_duval_diagnosis": None, "error": "Model not loaded. Check logs."}
    
    # Scaler can be optional if we decide to proceed with unscaled data on error
    # Label encoder is critical for meaningful output

    try:
        processed_result = preprocess_input_for_prediction(
            raw_input_data, 
            config, 
            loaded_scaler # Pass the loaded scaler
        )
        
        # Pylance Fix: Check if df_scaled (first element) is None
        if processed_result[0] is None: 
             raise ValueError("Preprocessing returned no data for scaling/prediction.")
        
        df_scaled: pd.DataFrame = processed_result[0] # Now df_scaled is known to be a DataFrame
        health_index: Optional[float] = processed_result[1]
        ch4_pct: float = processed_result[2] if processed_result[2] is not None else 0.0 # Pylance Fix: Default if None
        c2h4_pct: float = processed_result[3] if processed_result[3] is not None else 0.0 # Pylance Fix: Default if None
        c2h2_pct: float = processed_result[4] if processed_result[4] is not None else 0.0 # Pylance Fix: Default if None

        input_tensor = torch.tensor(df_scaled.values, dtype=torch.float32).to(DEVICE) # Pylance Fix: Use global DEVICE
        with torch.no_grad():
            output_logits = loaded_model(input_tensor)
            _, predicted_numerical_label = torch.max(output_logits, 1)
            predicted_numerical_label = predicted_numerical_label.item()

        predicted_fault_ml: str = f"NumericalLabel_{predicted_numerical_label}" # Default
        if loaded_label_encoder is not None and isinstance(loaded_label_encoder, LabelEncoder): # Pylance Fix
            predicted_fault_ml = loaded_label_encoder.inverse_transform([predicted_numerical_label])[0]
        elif loaded_label_encoder is None:
             logging.warning("Label encoder not loaded. Predictions are numerical.")


        raw_duval_diag = duval_triangle_fault_type(ch4_pct, c2h4_pct, c2h2_pct)
        # Pylance fix: provide default for health_index if None for adjust_fault_label
        adjusted_duval_diag = adjust_fault_label(raw_duval_diag, health_index if health_index is not None else 75.0) 

        logging.info(f"ML Prediction: {predicted_fault_ml}, Raw Duval: {raw_duval_diag}, Adjusted Duval (HI={health_index}): {adjusted_duval_diag}")
        return {
            "status": "success",
            "prediction_ml": predicted_fault_ml,
            "raw_duval_diagnosis": raw_duval_diag,
            "adjusted_duval_diagnosis": adjusted_duval_diag,
            "error": None
        }

    except ValueError as ve: 
        logging.error(f"Input or preprocessing error: {ve}", exc_info=False)
        return {"status": "error", "prediction_ml": None, "raw_duval_diagnosis": None, "adjusted_duval_diagnosis": None, "error": str(ve)}
    except Exception as e:
        logging.error(f"General prediction error: {e}", exc_info=True)
        return {"status": "error", "prediction_ml": None, "raw_duval_diagnosis": None, "adjusted_duval_diagnosis": None, "error": f"An unexpected error occurred: {str(e)}"}

# (Unit tests - content removed for brevity but should be kept in your file)
# Example usage
if __name__ == '__main__':
    if loaded_model and loaded_label_encoder: # Scaler is optional for this print test
        sample_data = {
            'Hydrogen': 100, 'Oxigen': 2000, 'Nitrogen': 50000, 'Methane': 50,
            'CO': 100, 'CO2': 2000, 'Ethylene': 20, 'Ethane': 30, 'Acethylene': 2,
            'DBDS': 10, 'Power factor': 0.5, 'Interfacial V': 30,
            'Dielectric rigidity': 60, 'Water content': 10,
            'Health index': 90, 'Life expectation': 15
        }
        for f_name in config['features']['base_input_features']:
            sample_data.setdefault(f_name, 0) # Ensure all base features exist

        result = predict_fault(sample_data)
        print(f"\nExample Prediction Result: {result}")

        sample_data_faulty = {
            'Hydrogen': 200, 'Oxigen': 1500, 'Nitrogen': 60000, 'Methane': 800,
            'CO': 300, 'CO2': 2500, 'Ethylene': 300, 'Ethane': 150, 'Acethylene': 50,
            'DBDS': 5, 'Power factor': 0.8, 'Interfacial V': 25,
            'Dielectric rigidity': 50, 'Water content': 20,
            'Health index': 65, 'Life expectation': 10
        }
        for f_name in config['features']['base_input_features']:
            sample_data_faulty.setdefault(f_name, 0)

        result_faulty = predict_fault(sample_data_faulty)
        print(f"\nExample Faulty Prediction Result: {result_faulty}")
    else:
        print("\nCannot run example prediction: Model or label encoder not loaded.")