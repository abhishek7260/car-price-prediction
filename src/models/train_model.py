import yaml
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load parameters from params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Access parameters for RandomForestRegressor
n_estimators = params['rfr']['n_estimators']
max_depth = params['rfr']['max_depth']

# Load training data
train_data = pd.read_csv("E:\\car-prediction-mlflow\\car-price-prediction\\data\\processed\\process_train.csv")
x_train = train_data.drop(['quality', 'Id'], axis=1)
y_train = train_data['quality']

# Train the model
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
model.fit(x_train, y_train)

# Save the trained model
pickle.dump(model, open("results/model.pkl", "wb"))
