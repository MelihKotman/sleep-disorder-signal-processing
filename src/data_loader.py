# src/data_loader.py
# Burası veriyi setini yükleme işini yapar.

import pandas as pd
from src.config import DATA_PATH

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Veri setindeki ham veriyi DataFrame olarak yükler
    Sütunlardaki isimler arasındaki boşlukları strip() fonksiyonu ile temizler.
    """
    df = pd.read_csv(path) # CSV dosyasını pandas DataFrame olarak yükle

    df.columns = [c.strip() for c in df.columns] # Sütun isimlerindeki boşlukları temizle

    return df