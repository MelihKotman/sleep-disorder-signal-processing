# src/config.py
# Projenin genel konfigürasyon ayarları buraya yazılır.

RANDOM_STATE = 42 # Kod her çalıtşığında aynı sonucu versin diye sabitlenmiş rastgelelik durumu

DATA_PATH = "data/raw-data/Sleep_health_and_lifestyle_dataset.csv" # Veri setinin yolu
TARGET_COL = "Sleep Disorder" # Hedef değişkenin adı

TEST_SIZE = 0.20 # Veri setinin %20'si test için ayrılacak kalanı eğitim için kullanılacak