# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew

le = LabelEncoder()

def factorize(df, factor_df, column, fill_na=None):
    factor_df[column] = df[column]
    if fill_na is not None:
        factor_df[column].fillna(fill_na, inplace=True)
    le.fit(factor_df[column].unique())
    factor_df[column] = le.transform(factor_df[column])
    return factor_df

# 整合所有的数值型特征
def merge_numerical(df, lot_frontage_by_neighborhood):
    all_df = pd.DataFrame(index=df.index)

    all_df['LotFrontage'] = df['LotFrontage']
    for key, group in lot_frontage_by_neighborhood:
        idx = (df['Neighborhood'] == key) & (df['LotFrontage'].isnull())
        all_df.loc[idx, 'LotFrontage'] = group.median()

    all_df['LotArea'] = df['LotArea']

    all_df['MasVnrArea'] = df['MasVnrArea']
    all_df['MasVnrArea'].fillna(0, inplace=True)

    all_df['BsmtFinSF1'] = df['BsmtFinSF1']
    all_df['BsmtFinSF1'].fillna(0, inplace=True)

    all_df['BsmtFinSF2'] = df['BsmtFinSF2']
    all_df['BsmtFinSF2'].fillna(0, inplace=True)

    all_df['BsmtUnfSF'] = df['BsmtUnfSF']
    all_df['BsmtUnfSF'].fillna(0, inplace=True)

    all_df['TotalBsmtSF'] = df['TotalBsmtSF']
    all_df['TotalBsmtSF'].fillna(0, inplace=True)

    all_df['1stFlrSF'] = df['1stFlrSF']
    all_df['2ndFlrSF'] = df['2ndFlrSF']
    all_df['GrLivArea'] = df['GrLivArea']

    all_df['GarageArea'] = df['GarageArea']
    all_df['GarageArea'].fillna(0, inplace=True)

    all_df['WoodDeckSF'] = df['WoodDeckSF']
    all_df['OpenPorchSF'] = df['OpenPorchSF']
    all_df['EnclosedPorch'] = df['EnclosedPorch']
    all_df['3SsnPorch'] = df['3SsnPorch']
    all_df['ScreenPorch'] = df['ScreenPorch']

    all_df['BsmtFullBath'] = df['BsmtFullBath']
    all_df['BsmtFullBath'].fillna(0, inplace=True)

    all_df['BsmtHalfBath'] = df['BsmtHalfBath']
    all_df['BsmtHalfBath'].fillna(0, inplace=True)

    all_df['FullBath'] = df['FullBath']
    all_df['HalfBath'] = df['HalfBath']
    all_df['BedroomAbvGr'] = df['BedroomAbvGr']
    all_df['KitchenAbvGr'] = df['KitchenAbvGr']
    all_df['TotRmsAbvGrd'] = df['TotRmsAbvGrd']
    all_df['Fireplaces'] = df['Fireplaces']

    all_df['GarageCars'] = df['GarageCars']
    all_df['GarageCars'].fillna(0, inplace=True)

    all_df['CentralAir'] = (df['CentralAir'] == 'Y') * 1.0

    all_df['OverallQual'] = df['OverallQual']
    all_df['OverallCond'] = df['OverallCond']

    qual_dict = {None: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    all_df['ExterQual'] = df['ExterQual'].map(qual_dict).astype(int)
    all_df['ExterCond'] = df['ExterCond'].map(qual_dict).astype(int)
    all_df['BsmtQual'] = df['BsmtQual'].map(qual_dict).astype(int)
    all_df['BsmtCond'] = df['BsmtCond'].map(qual_dict).astype(int)
    all_df['HeatingQC'] = df['HeatingQC'].map(qual_dict).astype(int)
    all_df['KitchenQual'] = df['KitchenQual'].map(qual_dict).astype(int)
    all_df['FireplaceQu'] = df['FireplaceQu'].map(qual_dict).astype(int)
    all_df['GarageQual'] = df['GarageQual'].map(qual_dict).astype(int)
    all_df['GarageCond'] = df['GarageCond'].map(qual_dict).astype(int)

    all_df['BsmtExposure'] = df['BsmtExposure'].map(
        {None: 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}).astype(int)

    bsmt_fin_dict = {None: 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    all_df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_dict).astype(int)
    all_df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_dict).astype(int)

    all_df['Functional'] = df['Functional'].map(
        {None: 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4,
         'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}).astype(int)

    all_df['GarageFinish'] = df['GarageFinish'].map(
        {None: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}).astype(int)

    all_df['Fence'] = df['Fence'].map(
        {None: 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}).astype(int)

    all_df['YearBuilt'] = df['YearBuilt']
    all_df['YearRemodAdd'] = df['YearRemodAdd']

    all_df['GarageYrBlt'] = df['GarageYrBlt']
    all_df['GarageYrBlt'].fillna(0.0, inplace=True)

    all_df['MoSold'] = df['MoSold']
    all_df['YrSold'] = df['YrSold']

    all_df['LowQualFinSF'] = df['LowQualFinSF']
    all_df['MiscVal'] = df['MiscVal']

    all_df['PoolQC'] = df['PoolQC'].map(qual_dict).astype(int)

    all_df['PoolArea'] = df['PoolArea']
    all_df['PoolArea'].fillna(0, inplace=True)

    all_df = factorize(df, all_df, 'MSSubClass')
    all_df = factorize(df, all_df, 'MSZoning', 'RL')
    all_df = factorize(df, all_df, 'LotConfig')
    all_df = factorize(df, all_df, 'Neighborhood')
    all_df = factorize(df, all_df, 'Condition1')
    all_df = factorize(df, all_df, 'BldgType')
    all_df = factorize(df, all_df, 'HouseStyle')
    all_df = factorize(df, all_df, 'RoofStyle')
    all_df = factorize(df, all_df, 'Exterior1st', 'Other')
    all_df = factorize(df, all_df, 'Exterior2nd', 'Other')
    all_df = factorize(df, all_df, 'MasVnrType', 'None')
    all_df = factorize(df, all_df, 'Foundation')
    all_df = factorize(df, all_df, 'SaleType', 'Oth')
    all_df = factorize(df, all_df, 'SaleCondition')

    all_df['IsRegularLotShape'] = (df['LotShape'] == 'Reg') * 1

    all_df['IsLandLevel'] = (df['LandContour'] == 'Lvl') * 1

    all_df['IsLandSlopeGentle'] = (df['LandSlope'] == 'Gtl') * 1

    all_df['IsElectricalSBrkr'] = (df['Electrical'] == 'SBrkr') * 1

    all_df['IsGarageDetached'] = (df['GarageType'] == 'Detchd') * 1

    all_df['IsPavedDrive'] = (df['PavedDrive'] == 'Y') * 1

    all_df['HasShed'] = (df['MiscFeature'] == 'Shed') * 1.

    all_df['Remodeled'] = (all_df['YearRemodAdd'] != all_df['YearBuilt']) * 1

    all_df['RecentRemodel'] = (all_df['YearRemodAdd'] == all_df['YrSold']) * 1

    all_df['VeryNewHouse'] = (all_df['YearBuilt'] == all_df['YrSold']) * 1

    all_df['Has2ndFloor'] = (all_df['2ndFlrSF'] == 0) * 1
    all_df['HasMasVnr'] = (all_df['MasVnrArea'] == 0) * 1
    all_df['HasWoodDeck'] = (all_df['WoodDeckSF'] == 0) * 1
    all_df['HasOpenPorch'] = (all_df['OpenPorchSF'] == 0) * 1
    all_df['HasEnclosedPorch'] = (all_df['EnclosedPorch'] == 0) * 1
    all_df['Has3SsnPorch'] = (all_df['3SsnPorch'] == 0) * 1
    all_df['HasScreenPorch'] = (all_df['ScreenPorch'] == 0) * 11

    all_df['HighSeason'] = df['MoSold'].replace(
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

    all_df['NewerDwelling'] = df['MSSubClass'].replace(
        {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

    all_df.loc[df.Neighborhood == 'NridgHt', 'Neighborhood_Good'] = 1
    all_df.loc[df.Neighborhood == 'Crawfor', 'Neighborhood_Good'] = 1
    all_df.loc[df.Neighborhood == 'StoneBr', 'Neighborhood_Good'] = 1
    all_df.loc[df.Neighborhood == 'Somerst', 'Neighborhood_Good'] = 1
    all_df.loc[df.Neighborhood == 'NoRidge', 'Neighborhood_Good'] = 1
    all_df['Neighborhood_Good'].fillna(0, inplace=True)

    all_df['SaleCondition_PriceDown'] = df.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 2, 'AdjLand': 3,
         'Family': 4, 'Normal': 5, 'Partial': 0})

    all_df['BoughtOffPlan'] = df.SaleCondition.replace(
        {'Abnorml': 0, 'Alloca': 0, 'AdjLand': 0,
         'Family': 0, 'Normal': 0, 'Partial': 1})


    all_df['BadHeating'] = df.HeatingQC.replace(
        {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})

    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                 '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                 'ScreenPorch', 'LowQualFinSF', 'PoolArea']
    all_df['TotalArea'] = all_df[area_cols].sum(axis=1)

    all_df['TotalArea1st2nd'] = all_df['1stFlrSF'] + all_df['2ndFlrSF']
    # all_df['AllLivArea'] = all_df['TotalArea1st2nd'] + all_df["LowQualFinSF"] + all_df["GrLivArea"]

    all_df['Age'] = 2010 - all_df['YearBuilt']
    all_df['TimeSinceSold'] = 2010 - all_df['YrSold']

    all_df['SeasonSold'] = all_df['MoSold'].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1,
                                                 5: 1, 6: 2, 7: 2, 8: 2, 9: 3,
                                                 10: 3, 11: 3}).astype(int)

    all_df['YearsSinceRemodel'] = all_df['YrSold'] - all_df['YearRemodAdd']

    all_df['SimplOverallQual'] = all_df.OverallQual
    all_df['SimplOverallCond'] = all_df.OverallCond
    all_df['SimplPoolQC'] = all_df.PoolQC
    all_df['SimplGarageCond'] = all_df.GarageCond
    all_df['SimplGarageQual'] = all_df.GarageQual
    all_df['SimplFireplaceQu'] = all_df.FireplaceQu
    all_df['SimplFireplaceQu'] = all_df.FireplaceQu
    all_df['SimplFunctional'] = all_df.Functional
    all_df['SimplKitchenQual'] = all_df.KitchenQual
    all_df['SimplHeatingQC'] = all_df.HeatingQC
    all_df['SimplBsmtFinType1'] = all_df.BsmtFinType1
    all_df['SimplBsmtFinType2'] = all_df.BsmtFinType2
    all_df['SimplBsmtCond'] = all_df.BsmtCond
    all_df['SimplBsmtQual'] = all_df.BsmtQual
    all_df['SimplExterCond'] = all_df.ExterCond
    all_df['SimplExterQual'] = all_df.ExterQual

    neighborhood_map = {
        'MeadowV': 0,  # 88000
        'IDOTRR': 1,   # 103000
        'BrDale': 1,   # 106000
        'OldTown': 1,  # 119000
        'Edwards': 1,  # 119500
        'BrkSide': 1,  # 124300
        'Sawyer': 1,   # 135000
        'Blueste': 1,  # 137500
        'SWISU': 2,    # 139500
        'NAmes': 2,    # 140000
        'NPkVill': 2,  # 146000
        'Mitchel': 2,  # 153500
        'SawyerW': 2,  # 179900
        'Gilbert': 2,  # 181000
        'NWAmes': 2,   # 182900
        'Blmngtn': 2,  # 191000
        'CollgCr': 2,  # 197200
        'ClearCr': 3,  # 200250
        'Crawfor': 3,  # 200624
        'Veenker': 3,  # 218000
        'Somerst': 3,  # 225500
        'Timber': 3,   # 228475
        'StoneBr': 4,  # 278000
        'NoRidge': 4,  # 290000
        'NridgHt': 4,  # 315000
    }

    all_df['NeighborhoodBin'] = df['Neighborhood'].map(neighborhood_map)
    # all_df['Bath'] = all_df["FullBath"] + all_df["HalfBath"] + all_df["BsmtFullBath"] + all_df["BsmtHalfBath"]
    # all_df["ImpArea"] = all_df["GrLivArea"] + all_df["TotalBsmtSF"] + all_df["GarageArea"]

    return all_df

# one-hot编码
def onehot(onehot_df, df, column_name, fill_na, drop_name):
    onehot_df[column_name] = df[column_name]
    if fill_na is not None:
        onehot_df[column_name].fillna(fill_na, inplace=True)

    dummies = pd.get_dummies(onehot_df[column_name], prefix='_' + column_name)
    onehot_df = onehot_df.join(dummies)
    onehot_df = onehot_df.drop([column_name], axis=1)
    return onehot_df

def merge_onehot(df):
    onehot_df = pd.DataFrame(index=df.index)

    onehot_df = onehot(onehot_df, df, 'MSSubClass', None, '40')
    onehot_df = onehot(onehot_df, df, 'MSZoning', 'RL', 'RH')
    onehot_df = onehot(onehot_df, df, 'LotConfig', None, 'FR3')
    onehot_df = onehot(onehot_df, df, 'Neighborhood', None, 'OldTown')
    onehot_df = onehot(onehot_df, df, 'Condition1', None, 'RRNe')
    onehot_df = onehot(onehot_df, df, 'BldgType', None, '2fmCon')
    onehot_df = onehot(onehot_df, df, 'HouseStyle', None, '1.5Unf')
    onehot_df = onehot(onehot_df, df, 'RoofStyle', None, 'Shed')
    onehot_df = onehot(onehot_df, df, 'Exterior1st', 'VinylSd', 'CBlock')
    onehot_df = onehot(onehot_df, df, 'Exterior2nd', 'VinylSd', 'CBlock')
    onehot_df = onehot(onehot_df, df, 'Foundation', None, 'Wood')
    onehot_df = onehot(onehot_df, df, 'SaleType', 'WD', 'Oth')
    onehot_df = onehot(onehot_df, df, 'SaleCondition', 'Normal', 'AdjLand')

    temp_df = df[['MasVnrType', 'MasVnrArea']].copy()
    idx = (df['MasVnrArea'] != 0) & ((df['MasVnrType'] == 'None') | (df['MasVnrType'].isnull()))
    temp_df.loc[idx, 'MasVnrType'] = 'BrkFace'
    onehot_df = onehot(onehot_df, temp_df, 'MasVnrType', 'None', 'BrkCmn')

    onehot_df = onehot(onehot_df, df, 'LotShape', None, 'IR3')
    onehot_df = onehot(onehot_df, df, 'LandContour', None, 'Low')
    onehot_df = onehot(onehot_df, df, 'LandSlope', None, 'Sev')
    onehot_df = onehot(onehot_df, df, 'Electrical', 'SBrkr', 'FuseP')
    onehot_df = onehot(onehot_df, df, 'GarageType', 'None', 'CarPort')
    onehot_df = onehot(onehot_df, df, 'PavedDrive', None, 'P')
    onehot_df = onehot(onehot_df, df, 'MiscFeature', 'None', 'Othr')

    onehot_df = onehot(onehot_df, df, 'Street', None, 'Grvl')
    onehot_df = onehot(onehot_df, df, 'Alley', 'None', 'Grvl')
    onehot_df = onehot(onehot_df, df, 'Condition2', None, 'PosA')
    onehot_df = onehot(onehot_df, df, 'RoofMatl', None, 'WdShake')
    onehot_df = onehot(onehot_df, df, 'Heating', None, 'Wall')

    onehot_df = onehot(onehot_df, df, 'ExterQual', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'ExterCond', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'BsmtQual', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'BsmtCond', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'HeatingQC', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'KitchenQual', 'TA', 'Ex')
    onehot_df = onehot(onehot_df, df, 'FireplaceQu', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'GarageQual', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'GarageCond', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'PoolQC', 'None', 'Ex')
    onehot_df = onehot(onehot_df, df, 'BsmtExposure', 'None', 'Gd')
    onehot_df = onehot(onehot_df, df, 'BsmtFinType1', 'None', 'GLQ')
    onehot_df = onehot(onehot_df, df, 'BsmtFinType2', 'None', 'GLQ')
    onehot_df = onehot(onehot_df, df, 'Functional', 'Typ', 'Typ')
    onehot_df = onehot(onehot_df, df, 'GarageFinish', 'None', 'Fin')
    onehot_df = onehot(onehot_df, df, 'Fence', 'None', 'MnPrv')
    onehot_df = onehot(onehot_df, df, 'MoSold', None, None)

    year_map = pd.concat(pd.Series('YearBin' + str(i+1),
                                   index=range(1871+i*20, 1891+i*20)) for i in range(0, 7))

    yearbin_df = pd.DataFrame(index=df.index)
    yearbin_df['GarageYrBltBin'] = df.GarageYrBlt.map(year_map)
    yearbin_df['GarageYrBltBin'].fillna('Unknown', inplace=True)

    yearbin_df['YearBuiltBin'] = df.YearBuilt.map(year_map)
    yearbin_df['YearRemodAddBin'] = df.YearRemodAdd.map(year_map)

    onehot_df = onehot(onehot_df, yearbin_df, 'GarageYrBltBin', None, None)
    onehot_df = onehot(onehot_df, yearbin_df, 'YearBuiltBin', None, None)
    onehot_df = onehot(onehot_df, yearbin_df, 'YearRemodAddBin', None, None)

    return onehot_df



def load_data():
    # 加载数据
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # 剔除异常点
    train_df.drop(train_df[train_df['GrLivArea'] > 4000].index, inplace=True)

    test_df.loc[666, 'GarageQual'] = 'TA'
    test_df.loc[666, 'GarageCond'] = 'TA'
    test_df.loc[666, 'GarageFinish'] = 'Unf'
    test_df.loc[666, 'GarageYrBlt'] = '1980'

    test_df.loc[1116, 'GarageType'] = np.nan

    lot_frontage_by_neighborhood = train_df['LotFrontage'].groupby(train_df['Neighborhood'])

    # 整合数值型特征
    train_df_munged = merge_numerical(train_df, lot_frontage_by_neighborhood)
    test_df_munged = merge_numerical(test_df, lot_frontage_by_neighborhood)

    print('Training set size:', train_df_munged.shape)
    print('Test set size:', test_df_munged.shape)

    print('Features engineering..')

    neighborhood_bin_train = pd.DataFrame(index=train_df.index)
    neighborhood_bin_train['NeighborhoodBin'] = train_df_munged['NeighborhoodBin']
    neighborhood_bin_test = pd.DataFrame(index=test_df.index)
    neighborhood_bin_test['NeighborhoodBin'] = test_df_munged['NeighborhoodBin']

    numeric_features = train_df_munged.dtypes[train_df_munged.dtypes != 'object'].index

    skewed = train_df_munged[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
    skewed = skewed[skewed > 0.75]
    skewed = skewed.index

    train_df_munged[skewed] = np.log1p(train_df_munged[skewed])
    test_df_munged[skewed] = np.log1p(test_df_munged[skewed])

    from sklearn.preprocessing import StandardScaler, RobustScaler
    scaler = StandardScaler()
    scaler.fit(train_df_munged[numeric_features])

    scaled = scaler.transform(train_df_munged[numeric_features])
    for i, col in enumerate(numeric_features):
        train_df_munged[col] = scaled[:, i]

    scaled = scaler.transform(test_df_munged[numeric_features])
    for i, col in enumerate(numeric_features):
        test_df_munged[col] = scaled[:, i]

    onehot_df = merge_onehot(train_df)
    onehot_df = onehot(onehot_df, neighborhood_bin_train,
                       'NeighborhoodBin', None, None)
    train_df_munged = train_df_munged.join(onehot_df)

    drop_cols = [
                    '_Exterior1st_ImStucc', '_Exterior1st_Stone',
                    '_Exterior2nd_Other', '_HouseStyle_2.5Fin',

                    '_RoofMatl_Membran', '_RoofMatl_Metal', '_RoofMatl_Roll',
                    '_Condition2_RRAe', '_Condition2_RRAn', '_Condition2_RRNn',
                    '_Heating_Floor', '_Heating_OthW',

                    '_Electrical_Mix',
                    '_MiscFeature_TenC',
                    '_GarageQual_Ex', '_PoolQC_Fa'
                ]
    train_df_munged.drop(drop_cols, axis=1, inplace=True)

    onehot_df = merge_onehot(test_df)
    onehot_df = onehot(onehot_df, neighborhood_bin_test,
                       'NeighborhoodBin', None, None)
    test_df_munged = test_df_munged.join(onehot_df)

    test_df_munged.drop(['_MSSubClass_150'], axis=1, inplace=True)

    drop_cols = [
        '_Condition2_PosN',
        '_MSZoning_C (all)',
        '_MSSubClass_160',
    ]

    train_df_munged.drop(drop_cols, axis=1, inplace=True)
    test_df_munged.drop(drop_cols, axis=1, inplace=True)

    label_df = pd.DataFrame(index=train_df_munged.index, columns=['SalePrice'])
    label_df['SalePrice'] = np.log1p(train_df['SalePrice'])

    print('Training set size:', train_df_munged.shape)
    print('Test set size:', test_df_munged.shape)
    return train_df_munged, test_df_munged, label_df, test_df["Id"]
