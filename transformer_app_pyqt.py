import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QGridLayout, QScrollArea, QMessageBox
)
from PyQt6.QtCore import Qt
import model_pipeline # Your backend logic
import logging

class TransformerAppPyQt(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transformer Fault Diagnosis (PyQt)")
        self.setGeometry(100, 100, 650, 800)  # Adjusted height for more results

        self.input_fields_widgets = {}

        self.main_layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget_contents = QWidget()
        self.scroll_area.setWidget(self.scroll_area_widget_contents)
        self.input_grid_layout = QGridLayout(self.scroll_area_widget_contents)

        # Define input field details (units, placeholders)
        # This should ideally come from a more structured source or be more robust
        input_feature_details = {
            'Hydrogen': ("ppm", "e.g., 50"), 'Oxigen': ("ppm", "e.g., 10000"), 
            'Nitrogen': ("ppm", "e.g., 60000"), 'Methane': ("ppm", "e.g., 30"),
            'CO': ("ppm", "e.g., 200"), 'CO2': ("ppm", "e.g., 1500"), 
            'Ethylene': ("ppm", "e.g., 10"), 'Ethane': ("ppm", "e.g., 20"),
            'Acethylene': ("ppm", "e.g., 1"), 'DBDS': ("ppm", "e.g., 0"), 
            'Power factor': ("e.g. 0.2-1.0", "e.g., 0.3"), # Unit can be % or fraction
            'Interfacial V': ("mN/m", "e.g., 30"), 
            'Dielectric rigidity': ("kV", "e.g., 50"),
            'Water content': ("ppm", "e.g., 15"), 
            'Health index': ("0-100", "e.g., 85"), # This is an input
            'Life expectation': ("years", "e.g., 20")
        }
        
        # Use base_input_features from the loaded config in model_pipeline
        # Fallback if config is not loaded
        base_features_for_gui = []
        if hasattr(model_pipeline, 'config') and model_pipeline.config and \
           'features' in model_pipeline.config and \
           'base_input_features' in model_pipeline.config['features']:
            base_features_for_gui = model_pipeline.config['features']['base_input_features']
        else:
            # Fallback if config isn't loaded or structured as expected
            # This indicates an issue that should be fixed in model_pipeline.py loading
            logging.error("Config with base_input_features not available from model_pipeline for GUI setup.")
            # Provide a default list so the GUI can at least partially render
            base_features_for_gui = list(input_feature_details.keys())


        row_num = 0
        for feature_name in base_features_for_gui: # <--- CORRECTED
            unit, placeholder = input_feature_details.get(feature_name, ("", f"Enter {feature_name}"))
            label = QLabel(f"{feature_name} ({unit}):")
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder)
            
            self.input_grid_layout.addWidget(label, row_num, 0)
            self.input_grid_layout.addWidget(line_edit, row_num, 1)
            self.input_fields_widgets[feature_name] = line_edit
            row_num += 1
        
        self.main_layout.addWidget(self.scroll_area)

        self.predict_button = QPushButton("Diagnose Fault")
        self.predict_button.clicked.connect(self.run_diagnosis)
        self.main_layout.addWidget(self.predict_button)

        self.result_title_label = QLabel("Diagnosis Result:")
        font = self.result_title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self.result_title_label.setFont(font)
        self.main_layout.addWidget(self.result_title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.result_model_label = QLabel("ML Model Prediction: N/A")
        self.result_model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.result_model_label)
        
        self.result_raw_duval_label = QLabel("Raw Duval Diagnosis: N/A")
        self.result_raw_duval_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.result_raw_duval_label)

        self.result_adj_duval_label = QLabel("Adjusted Duval (with HI): N/A") # New Label
        self.result_adj_duval_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.result_adj_duval_label)
        
        self.main_layout.addStretch(1)
        self.setLayout(self.main_layout)

        # Check model components after pipeline has loaded them
        if not model_pipeline.loaded_model or \
           not model_pipeline.loaded_scaler or \
           not model_pipeline.loaded_label_encoder:
            QMessageBox.critical(self, "Initialization Error",
                                 "Failed to load one or more model components (model, scaler, or label encoder). "
                                 "Check 'app_pipeline.log' and paths in model_pipeline.py.")
            self.predict_button.setEnabled(False)
        else:
            logging.info("GUI: Model components seem to be loaded.")


    def run_diagnosis(self):
        raw_input_data = {}
        # Use base_features_for_gui for collecting input
        base_features_for_gui = model_pipeline.config['features']['base_input_features'] if \
                                hasattr(model_pipeline, 'config') and model_pipeline.config else \
                                list(self.input_fields_widgets.keys())

        for feature_name in base_features_for_gui:
            line_edit_widget = self.input_fields_widgets.get(feature_name)
            if not line_edit_widget:
                QMessageBox.warning(self, "Input Error", f"Input field for {feature_name} not found.")
                return

            value_str = line_edit_widget.text()
            if not value_str.strip(): # Check if empty after stripping whitespace
                QMessageBox.warning(self, "Input Error", f"Please provide a value for {feature_name}.")
                return
            try:
                raw_input_data[feature_name] = float(value_str)
            except ValueError:
                QMessageBox.warning(self, "Input Error", f"Invalid input for {feature_name} ('{value_str}'). Please enter a numeric value.")
                self.result_model_label.setText("ML Model Prediction: Invalid Input")
                self.result_raw_duval_label.setText("Raw Duval Diagnosis: Invalid Input")
                self.result_adj_duval_label.setText("Adjusted Duval: Invalid Input")
                return

        # Call the pipeline
        result = model_pipeline.predict_fault(raw_input_data)
        
        if result['status'] == 'success':
            self.result_model_label.setText(f"ML Model Prediction: {result['prediction_ml']}")
            self.result_raw_duval_label.setText(f"Raw Duval Diagnosis: {result['raw_duval_diagnosis']}")
            self.result_adj_duval_label.setText(f"Adjusted Duval (with HI): {result['adjusted_duval_diagnosis']}")
        else:
            QMessageBox.critical(self, "Prediction Error", f"Error: {result['error']}")
            self.result_model_label.setText("ML Model Prediction: Error")
            self.result_raw_duval_label.setText("Raw Duval Diagnosis: Error")
            self.result_adj_duval_label.setText("Adjusted Duval: Error")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TransformerAppPyQt()
    window.show()
    sys.exit(app.exec())