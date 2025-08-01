import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template

# Load the trained model and preprocessors
try:
    model = joblib.load('models/knn_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    selected_features = joblib.load('models/selected_features.joblib')
    label_encoders = joblib.load('models/label_encoders.joblib')
    original_columns = joblib.load('models/original_columns.joblib')
except FileNotFoundError:
    print("Error: Could not load model files. Please run model_setup.py first.")
    exit()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None

    if request.method == 'POST':
        # Get the comma-separated values from the textarea
        input_data_string = request.form['input_data']

        try:
            # Split the input string by commas
            input_values = [x.strip() for x in input_data_string.split(',')]

            # The original dataset has 41 features. 'num_outbound_cmds' is dropped later.
            # We expect 41 values from the user.
            if len(input_values) != 41:
                prediction_text = f"Error: Please enter exactly 41 comma-separated values. You entered {len(input_values)}."
            else:
                # Create a DataFrame with the original column names
                input_df = pd.DataFrame([input_values], columns=original_columns)

                # Apply the saved label encoders to the categorical columns
                # This must be done before dropping 'num_outbound_cmds' as it may not be the last column
                for col in input_df.columns:
                    if col in label_encoders:
                        # Convert to string to ensure consistency with the encoder's fitting
                        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                    else:
                        # Convert other columns to float for numerical operations
                        input_df[col] = input_df[col].astype(float)

                # Drop 'num_outbound_cmds' column as it's not used by the model
                if 'num_outbound_cmds' in input_df.columns:
                    input_df.drop(['num_outbound_cmds'], axis=1, inplace=True)

                # Filter the DataFrame to only include the selected features from RFE
                input_df = input_df[selected_features]

                # Apply the same StandardScaler used during training
                scaled_input = scaler.transform(input_df)

                # Make the prediction
                prediction = model.predict(scaled_input)
                
                # Map the prediction (0 or 1) back to a meaningful class
                # This mapping should be consistent with how the labels were encoded
                class_mapping = {0: 'intrusion', 1: 'normal'}
                predicted_class = class_mapping.get(prediction[0], 'Unknown')

                prediction_text = f"Predicted Class: {predicted_class}"

        except ValueError:
            prediction_text = "Error: Invalid input. Some values could not be converted to numbers."
        except KeyError as e:
            # This handles cases where a new categorical value is entered
            prediction_text = f"Error: A previously unseen categorical value was entered for a column. Details: {e}"
        except Exception as e:
            prediction_text = f"An unexpected error occurred: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
