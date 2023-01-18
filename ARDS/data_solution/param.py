eicu_path = 'F:/personal/data/eICU/'
mimic3_path = 'F:/personal/data/mimic-iii-clinical-database-1.4/'
mimic4_path = 'F:/personal/data/mimic-iv-1.0/'
lab_key_names = ['bedside glucose', 'potassium', 'sodium', 'glucose', 'creatinine', 'BUN', 'calcium', 'bicarbonate',
                 'platelets x 1000', 'WBC x 1000', 'magnesium', '-eos', '-basos', 'albumin', 'AST (SGOT)', 'ALT (SGPT)',
                 'total bilirubin', 'paO2', 'paCO2', 'pH', 'PT - INR', 'HCO3', 'FiO2', 'Base Excess', 'PTT', 'lactate',
                 'Total CO2', 'ionized calcium', 'Temperature', 'PEEP', 'Methemoglobin', 'Carboxyhemoglobin',
                 'Oxyhemoglobin', 'TV', 'direct bilirubin', '-bands', 'Respiratory Rate']
nursec_key_names = ['Heart Rate', 'Respiratory Rate', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Systolic',
                    'Non-Invasive BP Mean', 'Temperature (C)', 'Temperature (F)', 'Invasive BP Diastolic',
                    'Invasive BP Systolic', 'GCS Total', 'Invasive BP Mean', 'Eyes', 'Motor', 'Verbal',
                    'Bedside Glucose', 'CVP', 'PA Diastolic', 'PA Systolic', 'PA Mean', 'End Tidal CO2']
respi_key_names = ['FiO2', 'PEEP', 'Tidal Volume(set)', 'TV / kg IBW', 'Mean Airway Pressure', 'Peak Insp. Pressure',
                   'SaO2', 'Plateau Pressure', 'FIO2( %)', 'PEEP/CPAP', 'Tidal Volume Observed(VT)', 'ETCO2',
                   'Adult Con Pt/Vent SpO2', 'NIV Pt/Vent SpO2_5', 'Tidal Volume, Delivered']
physical_key_names = ['Heart Rate', 'Respiratory Rate', 'Non-Invasive BP Diastolic', 'Non-Invasive BP Systolic',
                      'Non-Invasive BP Mean', 'Temperature (C)', 'Temperature (F)', 'Invasive BP Diastolic',
                      'Invasive BP Systolic', 'GCS Total', 'Invasive BP Mean', 'Eyes', 'Motor', 'Verbal',
                      'Bedside Glucose', 'CVP', 'PA Diastolic', 'PA Systolic', 'PA Mean', 'End Tidal CO2'
                      ]
vitalperiodic_key_names = ['cvp', 'etco2', 'heartrate', 'padiastolic', 'pamean', 'pasystolic', 'respiration', 'sao2',
                           'temperature', 'systemicdiastolic', 'systemicmean', 'systemicsystolic']
vitalaperiodic_key_names = ['noninvasivediastolic', 'noninvasivemean', 'noninvasivesystolic']
vitalaperiodic_names = ['noninvasivediastolic', 'noninvasivemean', 'noninvasivesystolic', ]
mimic_dynamic_list = ['Albumin', 'ALT', 'AST', 'Bands', 'Base Excess', 'Basos', 'Bicarbonate', 'Bilirubin', 'BUN',
                      'Bun', 'Calcium', 'Calcium non-ionized', 'CO2', 'Creatinine', 'Eos', 'FiO2', 'Glucose',
                      'Hemoglobin', 'INR', 'Ionized Calcium', 'Lactate', 'PO2', 'pO2', 'Magnesium', 'PaCO2', 'PEEP',
                      'pH', 'PH', 'Platelets', 'Potassium', 'PTT', 'Peak Insp. Pressure', 'Sodium', 'Temperature',
                      'WBC', 'Mean Airway Pressure', 'Plateau Pressure', 'SaO2', 'SpO2', 'Tidal Volume',
                      'Central Venous Pressure', 'EtCO2', 'Eye', 'gcs', 'Motor', 'Verbal', 'Heart Rate', 'Hematocrit',
                      'Arterial Blood Pressure m', 'Arterial Blood Pressure s', 'Arterial Blood Pressure d',
                      'Pulmonary Artery Pressure', 'ART Blood Pressure A']
