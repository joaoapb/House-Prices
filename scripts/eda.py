# This script aims to explore the data in the dataset, looking for inconsistencies and other facts
# | IMPORT ----
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# | LOAD THE DATA ----
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

df = pd.concat([train, test])

# | EXPLORATORY ANALYSIS ----

# |     Check NAs
NAs = df.isna().sum()
round(100 * NAs[NAs>0] / df.shape[0],2)

# Those columns with more than 10% of missing values need to be assessed:
# PoolQC          99.66 --> if a house has no pool, then there is no quality - which is okay. 
# MiscFeature     96.40 --> again, a house might not have any MiscFeatures we might look into the relationship between MiscFeature and SalePrice
# Alley           93.22 --> same
# Fence           80.44 --> same
# FireplaceQu     48.65 --> same
# LotFrontage     16.65 --> here it is a bit tricky. A house should have a side to the street, or maybe it is only accessible by an alley. We need to check.

# |     Check missing LotFrontage
miss_lot_frontage = df.loc[df['LotFrontage'].isna()]
'{:,.2%}'.format(miss_lot_frontage.Alley.isna().sum() / miss_lot_frontage.shape[0])

# Most values are NA, so we can assume that we have an error here. We may try and estimate the correct values based on other features.

# |     Check the relationship between MiscFeature and SalePrice
with_misc_features = df.loc[df['MiscFeature'].notna(),]
'{:,.2%}'.format(with_misc_features.shape[0] / df.shape[0])
# only 3.6% of the sold houses have a MiscFeature

pd.get_dummies(with_misc_features[['MiscFeature', 'SalePrice']]).corr()
# little correlation between these features and SalePrice

df['MiscFeatureBin'] = 0
df.loc[df['MiscFeature'].notna(), 'MiscFeatureBin'] = 1
df[['MiscFeatureBin', 'SalePrice']].corr()
df[['MiscFeatureBin', 'SalePrice']].cov()
# still little correlation between MiscFeature and SalePrice

# | Check for MSSubClass values and its relationships
df.groupby(['MSSubClass']).size()
df.MSSubClass.hist()
plt.show()

sns.scatterplot(df['MSSubClass'], df['SalePrice'])
plt.show()

# from class 20 to 50, and 120 and 150 they are 1 or 1.5 story;
# from class 60 to 75, and 160 and 190 they are 2 or 2.5 story;
df['1To1.5Story'] = 0
df.loc[df['MSSubClass'].isin([20, 30, 40, 45, 50, 120, 150]), '1To1.5Story'] = 1

df['2To2.5Story'] = 0
df.loc[df['MSSubClass'].isin([60, 70, 75, 160, 190]), '2To2.5Story'] = 1

# another feature that can be extracted from this variable is the information about being a PUD (Planned Unit Development)
# These are classes 120 to 180
df['PUD'] = 0
df.loc[df['MSSubClass'].isin([120, 150, 160, 180]), 'PUD'] = 1

# | Check for MSZoning values and its relationships
df.groupby(['MSZoning']).size()

# we can get if a MSZoning is not in a residential area
df['IsResidential'] = 0
df.loc[df['MSZoning'].isin(['FV', 'RH', 'RL', 'RP', 'RM']), 'IsResidential'] = 0

# | Check for Alley values and its relationships
df.groupby(['Alley']).size()

# | Check for Public Utilities values and its relationships
df.groupby(['Utilities']).size()

# | Check for Public Utilities values and its relationships
df.groupby(['Condition1']).size()

# nearby a railroad
df['NearRailroad'] = 0
df.loc[
    (df['Condition1'].isin(['RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])) |
    (df['Condition2'].isin(['RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])), 'NearRailroad'] = 0


# | Ages 
df['Age'] = df['YrSold'] - df['YearBuild']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']




# | STRATEGY DISCUSSION ----
# Since many variables are categorical, it makes sense to transform these in dummies (one hot encoding)
# Some categories might not make sense in terms of model validation, but will be removed consequently
# 

sns.scatterplot(df['MiscFeature'], df['SalePrice'])
plt.show()


sns.histplot(df['MSSubClass'])
plt.show()



sns.histplot(df['MSZoning'])
plt.show()
