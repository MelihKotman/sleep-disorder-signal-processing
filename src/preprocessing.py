# src/preprocessing.py
# Veriyi modele hazırlamak için gerekli ön işleme fonksiyonları buraya yazılır.

# Buraya kadar sklearn kütüphanesinden 
# eğitim ve test verisi ayırma, 
# sütun dönüştürücü, 
# one-hot encoding, 
# standartlaştırma ve pipeline işlemleri 
# için gerekli modüller import edildi.

from unicodedata import numeric
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import train

from src.config import TARGET_COL, TEST_SIZE, RANDOM_STATE

# Hedefi (Sleep Disorder) ve Girdileri (Yaş, BMI, vs.) ayıran fonksiyon
def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Veri setindeki girdi özellikleri (X) ve hedef değişkeni (y) ayırır.
    Hedef değişken string olarak döner.
    Eğer hedef sütun veri setinde yoksa hata fırlatır.
    """
    # Hedef sütunu (target) eğer veri setinde yoksa hata fırlat
    if target_col not in df.columns:
        raise ValueError(f"Hedef sütun '{target_col}' veri setinde bulunmadı.") 
    
    X = df.drop(columns = [target_col]) # Girdi özellikleri
    y = df[target_col].fillna("None").astype(str) # Hedef değişkeni string olarak çevir.

    return X, y

# Sütun dönüştürücü (ColumnTransformer) oluşturma fonksiyonu
def build_preprocessor(X : pd.DataFrame) -> ColumnTransformer:
    """
    Girdi özellikleri için sütun dönüştürücü (ColumnTransformer) oluşturur.
    Kategorik sütunlar için One-Hot Encoding (Erkek -> 0, Kadın -> 1 gibi binary sütunlar oluşturma),
    Sayısal sütunlar için Standartlaştırma (StandardScaler) (Hepsini 0-1 arasına hapseder) uygular.
    """
    numeric_cols = X.select_dtypes(include = ['int64', 'float64']).columns.tolist() # Sayısal sütunlar
    categorical_cols = X.select_dtypes(include = ['object']).columns.tolist() # Kategorik sütunlar

    numeric_transformer = Pipeline(
        steps = [
            ("scaler", StandardScaler())
        ]
    ) # Sayısal sütunlar için standartlaştırma pipeline'ı oluşturuldu.

    categorical_transformer = Pipeline(
        steps = [
            ("onehot", OneHotEncoder(handle_unknown = "ignore", sparse_output=False))
        ]
    ) # Kategorik sütunlar için One-Hot Encoding pipeline'ı oluşturuldu.

    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_transformer, numeric_cols), # Sayısal sütunlar için dönüştürücü
            ("cat", categorical_transformer, categorical_cols) # Kategorik sütunlar için dönüştürücü
        ],
        remainder = "drop" # Diğer sütunlar atılacak
    )
    return preprocessor # Sütun dönüştürücü döner

# Eğitim ve test verisi ayırma fonksiyonu
def make_train_test_split(X, y):
    """
    Veriyi eğitim ve test olarak ikiye ayırır. 
    %20'si test için ayrılır.
    Hedef değişkenin (y) dağılımını korur (stratify = y).
    """
    return train_test_split(
        X, y,
        test_size = TEST_SIZE, # Test veri setinin oranı
        random_state = RANDOM_STATE, # Rastgelelik durumu sabitlendi
        stratify = y # Hedef değişkenin dağılımını koru
    )