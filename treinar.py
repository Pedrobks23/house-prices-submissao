import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from pipeline import engenharia_features

RANDOM_STATE = 42

def main():
    df = pd.read_csv('treino.csv')
    y = df['SalePrice'].values
    X = df.drop(columns=['SalePrice'])
    colunas = list(X.columns)
    num = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='Ausente')),
                    ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('num', num, selector(dtype_include=np.number)),
                              ('cat', cat, selector(dtype_include=object))])
    xgb = XGBRegressor(n_estimators=421, learning_rate=0.0357, max_depth=5, min_child_weight=10,
                       subsample=0.810, colsample_bytree=0.591, reg_alpha=0.0071, reg_lambda=0.558,
                       random_state=RANDOM_STATE, n_jobs=-1, objective='reg:squarederror')
    pipe = Pipeline([('fe', FunctionTransformer(engenharia_features, validate=False)),
                     ('prep', prep), ('est', xgb)])
    modelo = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    modelo.fit(X, y)
    joblib.dump({'modelo': modelo, 'colunas': colunas}, 'modelo.pkl')
    print('=' * 55)
    print('modelo.pkl salvo')
    print(f'  Estimador base  : {type(xgb).__name__}')
    print(f'  Transformação   : TransformedTargetRegressor (log1p / expm1)')
    print(f'  n_estimators    : {xgb.n_estimators}')
    print(f'  learning_rate   : {xgb.learning_rate}')
    print(f'  max_depth       : {xgb.max_depth}')
    print(f'  min_child_weight: {xgb.min_child_weight}')
    print(f'  subsample       : {xgb.subsample}')
    print(f'  colsample_bytree: {xgb.colsample_bytree}')
    print(f'  reg_alpha       : {xgb.reg_alpha}')
    print(f'  reg_lambda      : {xgb.reg_lambda}')
    print(f'  Colunas brutas  : {len(colunas)}')
    print('=' * 55)

if __name__ == '__main__':
    main()
