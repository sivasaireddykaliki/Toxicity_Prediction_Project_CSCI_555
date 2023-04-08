# Total code
import pandas as pd
import numpy as np
from rdkit import Chem
from lightgbm import LGBMClassifier
from rdkit.Chem import Descriptors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import f1_score

#Reading dataset files
train = pd.read_csv(r"/content/drive/MyDrive/train_II.csv")
test = pd.read_csv(r"/content/drive/MyDrive/test_II.csv")

# Split the Id column into two columns: chemical_id and assay_id
train[['chemical_id','assay_id']] = train['Id'].apply(lambda x: pd.Series(str(x).split(";")))
test[['chemical_id','assay_id']] = test['x'].apply(lambda x: pd.Series(str(x).split(";")))

# Convert the SMILES strings into molecule objects
train['Molecule'] = train['chemical_id'].apply(lambda x: Chem.MolFromSmiles(x))
test['Molecule'] = test['chemical_id'].apply(lambda x: Chem.MolFromSmiles(x))

# Filter out invalid SMILES strings and null molecules
train = train[train['Molecule'].notnull()]
test = test[test['Molecule'].notnull()]

i=1

# Create new features based on molecule
for name, function in Descriptors._descList:
    train[name] = train['Molecule'].apply(lambda x: function(x))
    test[name] = test['Molecule'].apply(lambda x: function(x))
    train[name].fillna(train[name].mean(), inplace=True)
    test[name].fillna(test[name].mean(), inplace=True)
    print("feature "+str(i))
    i+=1

# # Drop unnecessary columns
train = train.drop(['Id', 'chemical_id', 'Molecule'], axis=1)
test2 = test.drop(['x', 'chemical_id', 'Molecule'], axis=1)

train.to_csv("new_tr.csv",index=False)
test2.to_csv("new_te.csv",index=False)

train = pd.read_csv("new_tr.csv")
test2 = pd.read_csv("new_te.csv")

# Feature selection
#feature_set= list(train.columns)
feature_set= ['assay_id', 'MolLogP', 'MolMR', 'qed', 'BCUT2D_LOGPHI', 'Kappa3', 'VSA_EState6', 'VSA_EState4', 'MinPartialCharge', 'BCUT2D_MWHI', 'BalabanJ', 'BCUT2D_MWLOW', 'VSA_EState3', 'VSA_EState9', 'BCUT2D_MRLOW', 'VSA_EState5', 'BCUT2D_CHGLO', 'MaxAbsPartialCharge', 'MinAbsEStateIndex', 'BCUT2D_MRHI']
train_input = train[feature_set]
train_output = train['Expected']
test_input = test2[feature_set]

train_input['assay_id']= train_input['assay_id'].astype(int)
test_input['assay_id']= test_input['assay_id'].astype(int)

print(train_input.columns)
print(test_input.columns)


# Define the parameter grid to search over
param_grid = {
    'learning_rate': [0.1,0.15,0.2,0.25],
    'max_depth': [10,15,20],
    'n_estimators':[600,700,800]
}


# Create a HistGradientBoostingClassifier object
model = LGBMClassifier()
print("Model creation")
print("Tuning Parameters")
# Use RandomizedSearchCV to perform hyperparameter tuning
# Perform a grid search over the hyperparameter space
grid_search = GridSearchCV(model, param_grid, cv=5, verbose=2)
grid_search.fit(train_input, train_output)


# Train the model with the best hyperparameters
best_params = grid_search.best_params_
best_model = LGBMClassifier(**best_params)
best_model.fit(train_input, train_output)

print("Training done")

# Predict the output
pred = best_model.predict(test_input)
# Prepare the submission data frame
submission_val=pd.DataFrame({'Id': test.x, 'Predicted': pred.astype('int')})
# Save the submission data frame to a .csv file
submission_val.to_csv("submission.csv",index=False)
print("Output generated")

print("best_params=",best_params)

#Evaluate the model using cross-validation
scores = cross_val_score(best_model, train_input, train_output, cv=5, scoring='f1')
mean_f1 = np.mean(scores)
print("mean f1_score: ", mean_f1)