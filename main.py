import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from numpy import isnan
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Reading Data
ds = pd.read_csv(r'E:\On Use\Data Mining\Project\water_potability.csv')

# ----------------------------------------------------------------------------------------------------------------------
#                                                  Analysis
originalData = ds
# 1. Knowing my attributes
# print(originalData.info())
# 2. Linear Relation by correlation [In the Visual Section below]
# 3. Count Null in each column
# print(originalData.isnull().sum())
# 5. Label studies [In the Visual Section below]
# print(originalData['Potability'].value_counts())

# ----------------------------------------------------------------------------------------------------------------------
#                                                Pre-Processing

# 1. Handling Missing Values:
# 1.1. Dropping rows with NAs
def drop_NA_rows(data):
    data = data.dropna()
    return data


# 1.2. Dropping rows having both pH and Sulfate NA
def drop_pHandSulfate(data):
    data = data[(data['ph'].notna()) | (data['Sulfate'].notna())]
    return data


# 1.3. Dropping pH column
def drop_pH(data):
    data = data.drop(['ph'], axis='columns')
    return data


# 1.4. Dropping Sulphate column
def drop_Sulfate(data):
    data = data.drop(['Sulfate'], axis='columns')
    return data


# 1.5. Hot Deck Imputation (special form of prediction using knn)
def hot_deck_imputation(data):
    hot_deck_imputer = KNNImputer(n_neighbors=2, weights="uniform")
    data = hot_deck_imputer.fit_transform(data)
    data = pd.DataFrame(data, columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                                       'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'])
    return data


# 1.6. Multivariate feature Imputation
def iterative_imputation(data):
    multi_imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed = multi_imputer.fit_transform(data)
    data = pd.DataFrame(imputed, columns=data.columns)
    return data


# 2. Handling Outliers
# First, find boundary values:
HA = ds.mean() + 3 * ds.std()
LA = ds.mean() - 3 * ds.std()


# 2.1. Trimming of Outliers
def trim_outliers(data):
    data = data[(data < HA) & (data > LA)]
    return data


# 2.2. Capping on Outliers
def cap_outliers(data):
    data = np.where(
        data > HA,
        HA,
        np.where(
            data < LA,
            LA,
            data
        )
    )
    data = pd.DataFrame(data, columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                                       'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'])
    return data


# Step needed for 3: Turning some features into categorical values
# Pre-3. Turning pH and Hardness into categorical attributes
def ph_into_cat(data):
    acidic = data['ph'] < 7
    basic = data['ph'] > 7

    conditions = [acidic, basic]
    values = ['acidic', 'basic']
    data['phGroup'] = np.select(conditions, values)
    return data


def hardness_into_cat(data):
    soft = data['Hardness'] <= 60
    moderately_hard = data['Hardness'].between(60, 121, inclusive='neither')
    hard = data['Hardness'].between(121, 180, inclusive='both')
    very_hard = data['Hardness'] > 180

    conditions = [soft, moderately_hard, hard, very_hard]
    values = ['soft', 'moderately_hard', 'hard', 'very_hard']
    data['HardnessGroup'] = np.select(conditions, values)
    return data


# 3. Sampling
# 3.1 Random Sampling
def random_sampling(data):
    data_sample = data.sample(frac=0.6, replace=True, random_state=1)
    return data_sample


# 3.2 Stratified Sampling
# 3.2.1 Stratum as pH
def stratified_sampling_pH(data):
    data = ph_into_cat(data)
    data_groups = data.groupby('phGroup', group_keys=False)
    data_samples = data_groups.apply(lambda x: x.sample(frac=0.6))
    labelEncoder = LabelEncoder()
    labelEncoder.fit(data_samples["phGroup"])
    data_samples["phGroup"] = labelEncoder.transform(data_samples["phGroup"])
    return data_samples


# 3.2.2 Stratum as Hardness
def stratified_sampling_Hardness(data):
    data = hardness_into_cat(data)
    data_groups = data.groupby('HardnessGroup', group_keys=False)
    data_samples = data_groups.apply(lambda x: x.sample(frac=0.6))
    labelEncoder = LabelEncoder()
    labelEncoder.fit(data_samples["HardnessGroup"])
    data_samples["HardnessGroup"] = labelEncoder.transform(data_samples["HardnessGroup"])
    return data_samples


# 4. Data Loading and Splitting
def data_load(data):
    X = data.loc[:, data.columns != 'Potability']
    Y = data['Potability']
    return X, Y


# 4.1. Normal Data Splitting
def data_split(X, Y, s, r):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=s, random_state=r)
    return x_train, x_test, y_train, y_test


# 4.2. K-Fold Cross-Validation
def KFold_data_split(X, Y):
    kf = RepeatedKFold(n_splits=20, n_repeats=2, random_state=1)
    for train_index, test_index in kf.split(ds):
        x_train, x_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]


# 5. Data Standardization
def data_standardization(x_train, x_test):
    sc = StandardScaler()
    x_train = pd.DataFrame(sc.fit_transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(sc.transform(x_test), columns=x_test.columns)
    return x_train, x_test


# Preprocessing Flow
# 1. Drop NA rows w/ random sampling:

data1 = drop_NA_rows(ds)
data1 = data1.drop_duplicates()
data1 = random_sampling(data1)
x, y = data_load(data1)
x_train1, x_test1, y_train1, y_test1 = data_split(x, y, True, False)
x_train1, x_test1 = data_standardization(x_train1, x_test1)


# 3. Drop Sulfate and hot deck w/ random sampling:
'''
data1 = hot_deck_imputation(ds)
data1 = drop_Sulfate(data1)
data1 = data1.drop_duplicates()
data1 = random_sampling(data1)
x, y = data_load(data1)
x_train1, x_test1, y_train1, y_test1 = data_split(x, y, True, 150)
x_train1, x_test1 = data_standardization(x_train1, x_test1)
'''

# 3. Hot deck w/ random sampling:
'''
data1 = hot_deck_imputation(ds)
data1 = data1.drop_duplicates()
data1 = random_sampling(data1)
x, y = data_load(data1)
x_train1, x_test1, y_train1, y_test1 = data_split(x, y, True, 150)
x_train1, x_test1 = data_standardization(x_train1, x_test1)
'''

# 4. Drop NA rows w/ stratified sampling:
'''
data1 = drop_NA_rows(ds)
data1 = data1.drop_duplicates()
# pH label
# data1 = stratified_sampling_pH(data1)
# Hardness label
# data1 = stratified_sampling_Hardness(data1)
x, y = data_load(data1)
x_train1, x_test1, y_train1, y_test1 = data_split(x, y, True, 150)
x_train1, x_test1 = data_standardization(x_train1, x_test1)
'''

# 14. Iterative w/ random sampling:
'''
data1 = iterative_imputation(ds)
data1 = data1.drop_duplicates()
data1 = random_sampling(data1)
x, y = data_load(data1)
x_train1, x_test1, y_train1, y_test1 = data_split(x, y, True, 150)
x_train1, x_test1 = data_standardization(x_train1, x_test1)
'''


# --------------------------------------------------------------------------------------------------
#                                            Classifiers

results = [] # For Visualizing
names = ["KNN", "Naive Bayes", "Decision Tree", "SVM", "LOGISTIC REGRESSION", "Random Forest"]

def Modelevaluation(Model,pred):
    print("\t\t\t ", Model)
    print("Accuracy: ", accuracy_score(y_test1, pred) * 100, "%")
    print("Mean absolute error: ", mean_absolute_error(y_test1, pred) * 100, "%")
    print("Precision score: ", precision_score(y_test1, pred) * 100, "%")
    print("Recall score: ", recall_score(y_test1, pred) * 100, "%")
    print("--------------------------------------------------------")

# (1) KNN
# Modeling

neighbors = KNeighborsClassifier(n_neighbors=2)
neighbors.fit(x_train1, y_train1)
y_predKNN = neighbors.predict(x_test1)
# print(result)
# Accuracy
accKNN = accuracy_score(y_test1, y_predKNN)
results.append(accKNN)
Modelevaluation('KNN', y_predKNN)

# ------------------------------------------------------------------
# (2) Naïve Bayes
# Modeling

gnb = GaussianNB()
y_predNB = gnb.fit(x_train1, y_train1).predict(x_test1)
# Accuracy
accNB = accuracy_score(y_test1, y_predNB)
results.append(accNB)
Modelevaluation("Naive Bayes", y_predNB)

# -----------------------------------------------------------------
# (3) Decision Tree
# Modeling

DT = DecisionTreeClassifier()
DT = DT.fit(x_train1, y_train1)
y_predDT = DT.predict(x_test1)
# Accuracy
accDT = metrics.accuracy_score(y_test1, y_predDT)
results.append(accDT)
Modelevaluation("Decision Tree", y_predDT)

# -----------------------------------------------------------------
# (4) SVM

classifier = SVC(kernel='rbf', random_state=1)
classifier.fit(x_train1, y_train1)
Y_predSVM = classifier.predict(x_test1)
# Accuracy
cm = confusion_matrix(y_test1, Y_predSVM)
accSVM = float(cm.diagonal().sum()) / len(y_test1)
results.append(accSVM)
Modelevaluation("SVM", Y_predSVM)

# -----------------------------------------------------------------
# (5) Logistic Regression
# Modeling
LG = LogisticRegression()
LG.fit(x_train1, y_train1)
y_predLG = LG.predict(x_test1)
# Accuracy
accLG = accuracy_score(y_test1, y_predLG)
results.append(accLG)
Modelevaluation("Logistic Regression", y_predLG)

# ------------------------------------------------------------------------------------------------------------------
RF = RandomForestClassifier(n_estimators=100, random_state=100,  criterion='entropy', min_samples_leaf=50)
RF.fit(x_train1, y_train1)
y_predRF = RF.predict(x_test1)
accRF = accuracy_score(y_test1, y_predRF)
results.append(accRF)
Modelevaluation("Random Forest", y_predRF)

# ------------------------------------------------------------------------------------------------------------------
#                                                  Visualization
# Linear Relation
def LinearRelation(originalData):
    Total_data = originalData.iloc[:, :]
    correlation = Total_data.corr()
    Impact = correlation.index[abs(correlation['Potability'] > 0)]
    # Plot
    plt.subplots(figsize=(12, 8))
    highCO = Total_data.corr()
    sns.heatmap(highCO, annot=True)
    plt.show()
# LinearRelation(originalData)

#
# Potability Plotting study of Label
def PlotPotability(originalData):
    plt.clf()
    plt.style.use('ggplot')
    fig1, ax1 = plt.subplots()
    ax1.pie(originalData['Potability'].value_counts(), colors=['#51C4D3', '#CD6155'], labels=['Non Potable', 'Potable'],
            autopct='%1.1f%%', startangle=0, rotatelabels=False)
    centre_circle = plt.Circle((0, 0), 0.80, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')
    plt.tight_layout()
    plt.show()
# PlotPotability(originalData)

#
# Feature Importance
def VisualFeatureImportance(X, Y):
    extra_tree_forest = ExtraTreesClassifier(n_estimators=5, criterion='entropy', max_features=2)
    extra_tree_forest.fit(X, Y)
    feature_importance = extra_tree_forest.feature_importances_
    fin = np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis=0)
    plt.bar(X.columns, fin)
    plt.xlabel('Feature Labels')
    plt.ylabel('Feature Importances')
    plt.title('Comparison of different Feature Importances')
    plt.show()

# VisualFeatureImportance(x,y)

#  Accuracy Comparison
def AccComparison():
    Results_DF = pd.DataFrame({'Classifiers': list(names), 'Accuracy': list(results)}).sort_values('Accuracy', ascending=True)
    Results_DF.plot(x='Classifiers', y='Accuracy', kind='line')
    plt.suptitle('Accuracy Comparison')
    plt.show()

# AccComparison()