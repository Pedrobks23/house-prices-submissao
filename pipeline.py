import os
import joblib
import numpy as np
import pandas as pd

def engenharia_features(df):
    df = df.copy()
    def col(nome):
        if nome in df.columns:
            return pd.to_numeric(df[nome], errors='coerce').fillna(0)
        return pd.Series(0, index=df.index)
    df['TotalSF']      = col('TotalBsmtSF') + col('1stFlrSF') + col('2ndFlrSF')
    df['TotalBath']    = col('FullBath') + 0.5*col('HalfBath') + col('BsmtFullBath') + 0.5*col('BsmtHalfBath')
    df['HouseAge']     = (col('YrSold') - col('YearBuilt')).clip(lower=0)
    df['RemodAge']     = (col('YrSold') - col('YearRemodAdd')).clip(lower=0)
    df['TotalPorchSF'] = col('OpenPorchSF')+col('EnclosedPorch')+col('3SsnPorch')+col('ScreenPorch')+col('WoodDeckSF')
    df['HasPool']      = (col('PoolArea')>0).astype(int)
    df['HasGarage']    = (col('GarageArea')>0).astype(int)
    df['Has2ndFloor']  = (col('2ndFlrSF')>0).astype(int)
    df['HasBsmt']      = (col('TotalBsmtSF')>0).astype(int)
    df['HasFireplace'] = (col('Fireplaces')>0).astype(int)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    return df

def prever_precos(caminho_arquivo_teste):
    df_teste = pd.read_csv(caminho_arquivo_teste)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_modelo = os.path.join(base_dir, 'modelo.pkl')
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo nao encontrado: {caminho_modelo}")
    artefato = joblib.load(caminho_modelo)
    modelo = artefato['modelo']
    colunas = artefato['colunas']
    df_teste = df_teste.reindex(columns=colunas)
    predicoes = modelo.predict(df_teste)
    return np.clip(np.asarray(predicoes), 0, None)

if __name__ == '__main__':
    arquivo = 'teste_publico.csv'
    if os.path.exists(arquivo):
        p = prever_precos(arquivo)
        print('OK -', len(p), 'predicoes | min', round(float(p.min()),2), '| media', round(float(p.mean()),2))
    else:
        print('teste_publico.csv nao encontrado nesta pasta.')
