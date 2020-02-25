import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

titanic_train_data_path = "~/datasets/titanic/train.csv"
titanic_test_data_path = "~/datasets/titanic/test.csv"

train_data = pd.read_csv(titanic_train_data_path)
test_data = pd.read_csv(titanic_test_data_path)


# Function to numerise the embarkation details
def emb_trans(embarkation):
    if embarkation == 'S':
        return 1
    if embarkation == 'Q':
        return 2
    if embarkation == 'C':
        return 3
    else:
        return None


# Numerising and clearing the training data
train_morph = train_data.copy()
train_morph['sex_bool'] = (train_morph['Sex'] == 'male').apply(lambda x: int(x))
train_morph['emb_bool'] = train_morph['Embarked'].apply(emb_trans)
features = ['PassengerId', 'Pclass', 'Age', 'SibSp',
            'Parch', 'Fare', 'sex_bool', 'emb_bool', 'Survived']
train_morph = train_morph[features]

# Imputing the training data
my_imputer = SimpleImputer()
imputed_train_morph = pd.DataFrame(my_imputer.fit_transform(train_morph))
imputed_train_morph.columns = train_morph.columns

# Separating the training data
X = imputed_train_morph.drop('Survived', axis=1)
y = imputed_train_morph.Survived

# Splitting training data internally
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Decision Tree Model internal
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(train_X, train_y)
val_predictions = dt_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("The MAE for the dt_model is ", val_mae)
print("Predictions look like: ", val_predictions[0:5])

# Random Forest Model internal
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
val_predictions = rf_model.predict(val_X)
val_predictions = [int(round(pred)) for pred in val_predictions]  # Integerising the non-integer predictions.
val_mae = mean_absolute_error(val_predictions, val_y)
print("The MAE for the rf_model is ", val_mae)
print("Predictions look like: ", val_predictions[0:5])

# Clearing the test data
test_morph = test_data.copy()
test_morph['sex_bool'] = (test_morph['Sex'] == 'male').apply(lambda x: int(x))
test_morph['emb_bool'] = test_morph['Embarked'].apply(emb_trans)
features = ['PassengerId', 'Pclass', 'Age', 'SibSp',
            'Parch', 'Fare', 'sex_bool', 'emb_bool']
test_X = test_morph[features]

# Imputing the test data
my_imputer = SimpleImputer()
imputed_test_X = pd.DataFrame(my_imputer.fit_transform(test_X))
imputed_test_X.columns = test_X.columns

# Creating a full RFM with imputed data
rf_model_full = RandomForestRegressor(random_state=1)
rf_model_full.fit(X, y)
val_predictions = rf_model_full.predict(imputed_test_X)
val_predictions = [int(round(pred)) for pred in val_predictions]

# Creating an output CSV file
val_pred_series = pd.Series(val_predictions, name='Survived')
final_predictions = pd.concat([test_X.PassengerId, val_pred_series], axis = 1)
final_predictions = final_predictions.set_index('PassengerId')
final_predictions.to_csv('./my_predictions.csv')
