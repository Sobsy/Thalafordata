import pandas as pd
import numpy as np
import sys
import csv

def process_csv(file_path):
    try:
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Assuming the CSV file has a header row
            header = next(csv_reader)
            #print(f"Header: {header}")
            
            # Print each row in the CSV file
            for row in csv_reader:
                print(row)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if a file path is provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file_path>")
    else:
        # Get the CSV file path from the command line argument
        csv_file_path = sys.argv[1]
        process_csv(csv_file_path)


import pandas as pd

test=pd.read_csv(csv_file_path)


df=pd.read_csv("train_datat.csv")  #Add the path of the train-data
#test=pd.read_csv('sample_test_data.csv')
output=pd.read_csv('sample_output_generated.csv')

yout=output['HR']


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['condition'] = le.fit_transform(df['condition'])
test['condition'] = le.fit_transform(test['condition'])

X = df.drop(columns=['HR', 'uuid'])
y = df['HR']

#print(df['condition'].unique())

# Perform one-hot encoding for the 'condition' column
df_encoded = pd.get_dummies(df, columns=['condition'], prefix='condition')

# Display the DataFrame after one-hot encoding
#print(df_encoded.head())


Xt = test.drop(columns=[ 'uuid'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(Xt)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rf_model = RandomForestRegressor(random_state=42)

# Fit the model on the training data
rf_model.fit(X_train_scaled, y)

y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model performance
mse = mean_squared_error(yout, y_pred)
r2 = r2_score(yout, y_pred)

#print(f"Mean Squared Error (MSE): {mse}")
#print(f"R-squared (R2): {r2}")
results_df = pd.DataFrame({'Predicted Heart Rate Values': y_pred})
results_df.to_csv('results.csv', index=False)

print("Results.csv generated successfully")