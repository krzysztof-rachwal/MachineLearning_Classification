import os 
import tarfile
import urllib.request

#fetch the housing data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "../datasets/housing/housing.tgz"
#function to fetch the data from the internet. the urllib.request function does not work.
def fetch_housing_data (housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#fetch_housing_data()
#load the data
import pandas as pd

#returns Pandas dataframe object containing all the data.
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path) 

housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())

#plot a histogram
import matplotlib.pyplot as plt 
# housing.hist(bins=50, figsize=(20,15))
#print(plt.show())

#create a test set
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print('Train set: ', len(train_set), 'Test set: ', len(test_set))

#in order to avoid generating new test set each time the program is executed
#with the above function the model eventually use the whole dataset to train and test with we want to avoid

#we can use identifiers for each data instance. 
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#and then use combination of lognitude and lattitude to set unique identifier for each house.
housing_with_id = housing.reset_index()   # adds an `index` column
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


#you can split data using sklearn function
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#we can create median income categoy which may be important when predicting median housing prices. 
#pd.cut() function create an income category attribute with 5 categories labeled from 1 to 5 
housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0.,1.5,3.0,4.5,6.,np.inf],
                                labels=[1,2,3,4,5])

# housing['income_cat'].hist()
# print(plt.show())

#stratified sampling based on income category
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]

print(start_test_set['income_cat'].value_counts() / len(start_test_set))
#remove the income_cat attribute so the data is in its original state
for set_ in (start_train_set, start_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

#EXPLORING THE DATA

#working on a copy of the set
housing = start_train_set.copy()

#scatterplot visualizing districts. Using latitude and lognitude. 
#to visualize places with high density of data points we can use alpha option
# housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
# print(plt.show())

#graph displaying housing praices in the given area
#additionaly it shows the population size of each district
#cmap specifies colour palette.
#figsize specifies the size of the graph
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                s=housing['population']/100, label='population', figsize=(10,7),
                c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

#print the above graph
# plt.legend()
# print(plt.show())

#standard correlation coefficient
corr_matrix = housing.corr()
#how each attribute correlates with the median house value
print(corr_matrix['median_house_value'].sort_values(ascending=False))

#checking correlation between attributes using scatter_matrix()
#plots every numerical attribute against every other numerical attribute 
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
# print(plt.show())

#median house value and median income correlation
housing.plot(kind="scatter", x='median_income', y='median_house_value', alpha=0.1)
# print(plt.show())

#displaying additional attributes such as total no of rooms per household,
#total no. of people per household, etc. 
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#correlation matrix for the above
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))

#PREPARE DATA FOR ML ALGORITHMS
#revert to clean trainig set
housing = start_train_set.drop('median_house_value', axis=1)
housing_labels = start_train_set['median_house_value'].copy()

#data cleaning
#how to work with missing features
#1 Get rid of the corresponding districts.
housing.dropna(subset=["total_bedrooms"])
#2 Get rid of the whole attribute.
housing.drop("total_bedrooms", axis=1)
#3 Set the values to some value (zero, the mean, the median, etc.).
median = housing["total_bedrooms"].median()  # option 3
housing["total_bedrooms"].fillna(median, inplace=True)

#Scikit-learn provide class to deal with missing values
from sklearn.impute import SimpleImputer
#strategy defines what you want to do with missing attributes. 
#here we are going to replace missing attribtes with with the median of that attribute. 
#it is safe to apply imputer to all numerical values
imputer = SimpleImputer(strategy="median")

#we need to convert string into numerical value to calculate median 
housing_num = housing.drop("ocean_proximity", axis=1)

#fit the imputer
imputer.fit(housing_num)

#the computed median is stored in imputer's statistics_ instance variable. 
print(imputer.statistics_)

#to compare calculate the median yourself
print(housing_num.median().values)

#replace missing values in training set with the imputer values
X = imputer.transform(housing_num)

#result is a NumPy array. If you want to put it into pandas DataFrame do below
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

#handling text and categorial attributes
#ocean_proximity is the only text attribute. 
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

#most ML algorithms work with numbers so we need to convert it into numbers
#it puts text into 1d array and uses indices
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])

print(ordinal_encoder.categories_)

#ML algorithms may assume that 2 nearby values are more similar than 2 distant ones
#this is not oviously not the case here so we need to create one binary attribute per category
#one-hot encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())
print(cat_encoder.categories_)

#CUSTOM TRANSFORMERS
#you need custom transformers for tasks such as custom cleanup or combinig specific attributes.
#to be able to use your transformer with sklearn you need to implement 3 methods:
#fit(), transform(), and fit_transform()

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#Transformation pipelines - they need to include estimators where all 
#must have fit_transform() method except the last one. 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

#use ColumnTransformer to have single transformer for all types of columns (text and num)
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

#SELECT AND TRAIN THE MODEL

#train linear regression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#train few instances from a training set
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))

#measure model's RMSE
#it missess the predictions by about $68000 so quite a lot - underfitting
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

#use more powerful model to fix underfitting
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
#returns 0.0 which means no errors at all. it may be overfitted  
print(tree_rmse)

#use cross-validation to evaluate the model. 
#K-fold cross-validation
#cv - no. of folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
#Scikit-Learn’s cross-validation features expect a utility function (greater is better)
#rather than a cost function (lower is better), so the scoring function is actually 
#the opposite of the MSE (i.e., a negative value), 
#which is why the preceding code computes -scores before calculating the square root.

#display the results
#from the below results it can be seen that decision tree performs even worse than linear regression
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

print(display_scores(tree_rmse_scores))

#do corss validation for linear regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                              scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(display_scores(lin_rmse_scores))

#decission tree model is overfitting so badly that it performs worse than the linreg

#use RnadomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(lin_mse)
print("forest: ", forest_rmse)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                              scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print(display_scores(forest_rmse_scores))

#save models to easily use it in a future
import joblib

# joblib.dump(my_model, "my_model.pkl")
# # and later...
# my_model_loaded = joblib.load("my_model.pkl")


#FINE TUNE MODEL 

#Grid search using cross-validation to evaluate all possible combinations of hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)

#print evaluation scores
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#analyze the best models and their errors
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

#display importance scores next to their corresponding attribute names
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))

#EVALUATE THE SYSTEM ON THE TEST SET
#DON'T CALL FIT() ON TEST SET!!! ONLY TRANSFORM()

final_model = grid_search.best_estimator_

X_test = start_test_set.drop("median_house_value", axis=1)
y_test = start_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)   # => evaluates to 47,730.2

#compute a 95% confidenc interval for the generalizaion error
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors)))

