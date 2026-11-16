#Imports
import pickle

from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

folder = Path(__file__).parent
data_path = folder / 'data' / 'Global Oil and Gas Plant Tracker (GOGPT) - August 2026.xlsx'

#Copy the categorical and numerical features from the notebook
categorical=['chp', 'fuel']
numerical=['capacity_mw', 'start_year', 'latitude', 'longitude']

#Data loading function:
def load_data():
    df = pd.read_excel(data_path, sheet_name='Gas & Oil Units')

    #Reducing the dataset to only the columns we need:
    df = df[['GEM location ID', 'Turbine/Engine Technology', 'Capacity (MW)', 'Start year', 'Fuel classification', 'CHP', 'Latitude', 'Longitude']]
    df.columns = ['plant_id', 'technology', 'capacity_mw', 'start_year', 'fuel', 'chp', 'latitude', 'longitude']

    #Filling missing values in chp and start_year:
    df['chp'] = df['chp'].fillna('not found')
    df['start_year'] = df['start_year'].fillna(0)

    #Removing spaces from the categorical columns:
    for c in ['technology', 'fuel','chp']:
        df[c]=df[c].str.lower().str.replace(' ', '_')

    #Retaining only the four main technology types:
    df = df[df.technology.isin(['combined_cycle', 'gas_turbine', 'steam_turbine', 'internal_combustion'])]
    df = df.reset_index(drop=True)

    return df

#Training function:
def train(df_train, y_train):
    
    #Creating the DictVectorizer and transforming the training data:
    dv = DictVectorizer(sparse=False)
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    #Training the Gradient Boosting classifier:
    model = GradientBoostingClassifier(max_depth=7, random_state=42)
    model.fit(X_train, y_train)

    return dv, model

#Predict function:
def predict(df,dv, model):
    #Transforming the test data:
    test_dicts = df[categorical + numerical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)

    return model.predict(X_test)

#Main code:
df = load_data()

#Splitting the data into training and test sets, but remembering to account for plant_id:
plants = df.plant_id.unique()
plants_full_train, plants_test = train_test_split(plants, test_size=0.2, random_state=42)
df_full_train = df[df.plant_id.isin(plants_full_train)].reset_index(drop=True)
df_test = df[df.plant_id.isin(plants_test)].reset_index(drop=True)

#Training
dv, model = train(df_full_train, df_full_train.technology.values)

#Outputting the accuracy and macro F1 score:
y_pred = predict(df_test, dv, model)
print('Accuracy=', round(accuracy_score(df_test.technology.values, y_pred), 3))
print('Macro F1=', round(f1_score(df_test.technology.values, y_pred, average='macro'), 3))

#Saving the model to a bin file
with open(folder / 'model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print('Model saved to model.bin')