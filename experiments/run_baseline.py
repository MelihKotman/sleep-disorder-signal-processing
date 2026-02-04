# experiments/run_baseline.py
# Temel modelleri çalıştırma ve değerlendirme betiği. (Filtreleme yok)

import os
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so `import src...` works when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from sklearn.pipeline import Pipeline # Pipeline oluşturma
from sklearn.preprocessing import LabelEncoder  # Hedef değişkeni sayısallaştırmak için

from src.data_loader import load_raw_data # Ham veriyi yükleme fonksiyonu
from src.preprocessing import ( # Ön işleme fonksiyonları
    split_features_target,
    build_preprocessor,
    make_train_test_split
)
from src.models.classical import get_classical_models # Klasik ML modellerini alma fonksiyonu
from src.evaluation import ( # Değerlendirme metrikleri
    make_metrics_row,
    confusion_matrix_df,
)
from src.models.deep import get_deep_models # Derin öğrenme modellerini alma fonksiyonu

def safe_predict_proba(clf, X_test): # clf: sınıflandırıcı, X_test: test verisi
    """
    Eğer sınıflandırıcı olasılık tahmini yapabiliyorsa predict_proba kullanır.
    Yapamıyorsa direkt None döner.
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)
    else:
        return None

def main():
    print("=== BASELINE ÇALIŞTIRILIYOR === (FİLTRELEME YOK)")

    # 1. Ham veriyi yükle
    df = load_raw_data()
    print(f"Veri seti yüklendi. Veri setinin şekli: {df.shape}")

    # 2. Girdi özellikleri ve hedef değişkeni ayır
    X, y = split_features_target(df)
    print(f"Girdi özellikleri ve hedef değişken ayrıldı. Girdi şekli: {X.shape}, Hedef şekli: {y.shape}")

    # 3. Label Encoding (XGBoost ve Derin Öğrenme modelleri için)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y) # Etiketleri sayısallaştır
    label_names = le.classes_ # ['None', 'Insomnia', 'Sleep Apnea']
    labels = list(range(len(label_names))) # [0, 1, 2]
    print(f"Hedef değişken etiketleri: {label_names}")
    print(f"Sınıflar Sayısallaştırıldı: {dict(zip(labels,label_names))}")

    # 4. Eğitim ve test verisi ayır
    X_train, X_test, y_train, y_test = make_train_test_split(X, y_encoded)
    print(f"Eğitim ve test verisi ayrıldı. Eğitim şekli: {X_train.shape}, Test şekli: {X_test.shape}")

    # 6. Ön işleme adımlarını oluştur
    preprocessor = build_preprocessor(X_train)

    # 7. Klasik modelleri  ve derin öğrenme modellerini al
    models = get_classical_models()
    print(f"Klasik modeller alındı: {list(models.keys())}")
    deep_models = get_deep_models()
    print(f"Derin öğrenme modelleri alındı: {list(deep_models.keys())}")

    models.update(deep_models) # Klasik ve derin öğrenme modellerini birleştir

    # Eğitim ve değerlendirme sonuçlarını saklamak için listeler
    os.makedirs("results/tables", exist_ok = True) # Sonuç tablosu klasörü oluştur
    os.makedirs("results/figures", exist_ok = True) # Sonuç figür klasörü oluştur
    os.makedirs("results/roc_raw", exist_ok=True)

    rows = [] # Değerlendirme metrikleri için satırlar
    roc_rows = []

    for name, model in models.items():
        print(f"\n--- Model: {name} eğitiliyor ve değerlendiriliyor ---")

        clf = Pipeline(
            steps= [
                ("preprocess", preprocessor), # Ön işleme adımları
                ("model", model) # Sınıflandırıcı model
            ]
        ) # Pipeline oluştur ve modeli ekle

        clf.fit(X_train, y_train) # Modeli eğit

        y_pred = clf.predict(X_test) # Test verisi üzerinde tahmin yap

        #AUC için olasılık tahminleri al (Pipeline'da predict_proba vardır)
        y_proba = safe_predict_proba(clf, X_test)

        # =====================================================
        # ROC CURVE için ham veriyi kaydet
        # =====================================================
        if y_proba is not None:
            for i in range(len(y_test)):
                roc_row = {
                    "model": name,
                    "filter_method": "baseline",
                    "true_label": int(y_test[i])
                }
                for c in range(y_proba.shape[1]):
                    roc_row[f"proba_class_{c}"] = float(y_proba[i, c])
                roc_rows.append(roc_row)

        row = make_metrics_row(
            model_name = name,
            y_true = y_test,
            y_pred = y_pred,
            labels = labels,
            y_proba = y_proba
        ) # Değerlendirme metriklerini hesapla
        rows.append(row) # Satırı listeye ekle

        auc_val = row.get("roc_auc_ovr_macro", None) # AUC makro değeri
        auc_str = f"{auc_val:.4f}" if (auc_val is not None and pd.notna(auc_val)) else "NA" # AUC değeri stringi

        print(f"Değerlendirme Sonuçları : \n")
        print(
            f"{name} -> acc= {row['accuracy']:.4f}, " # Doğruluk
            f"f1_macro= {row['f1_macro']:.4f}, " # F1 makro
            f"spec_macro= {row['specificity_macro']:.4f}, " # Spesifiklik makro
            f"roc_auc_ovr_macro= {auc_str}" # AUC makro
        )

        # Karışıklık matrisini kaydet
        cm_df = confusion_matrix_df(y_test, y_pred, labels = labels)
        cm_df.index = label_names
        cm_df.columns = label_names
        cm_path = f"results/tables-baseline/confusion_matrix_{name}.csv"
        cm_df.to_csv(cm_path, index = True)
    
    # =====================================================
    # ROC CURVE ham verisini tek dosyada kaydet
    # =====================================================
    roc_raw_df = pd.DataFrame(roc_rows)
    roc_raw_path = "results/roc_raw/roc_raw_baseline.csv"
    roc_raw_df.to_csv(roc_raw_path, index=False)
    print(f"ROC raw verisi kaydedildi: {roc_raw_path}")

    # Sonuçları DataFrame'e dönüştür ve F1 makroya göre sırala
    results_df = pd.DataFrame(rows).sort_values(by = "f1_macro", ascending = False) 

    # Sonuçları CSV dosyasına kaydet
    out_path = "results/tables-baseline/baseline_metrics.csv"
    results_df.to_csv(out_path, index = False) 

    print("\n=== BASELINE ÇALIŞTIRMA TAMAMLANDI ===")
    print(results_df)
    print(f"\nBaseline metrikleri '{out_path}' dosyasına kaydedildi.")
    print(f"Karışıklık matrisleri 'results/tables/' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()