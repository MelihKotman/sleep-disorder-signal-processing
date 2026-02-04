# experiments/run_filtered.py
# TÜM filtrelerin (12 Adet) performansını test eden script.

import os
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from src.data_loader import load_raw_data
from src.preprocessing import build_preprocessor, make_train_test_split, split_features_target
from src.filters.filters import apply_filters
from src.models.classical import get_classical_models
from src.models.deep import get_deep_models
from src.evaluation import make_metrics_row
from src.evaluation import make_metrics_row, confusion_matrix_df

def main():
    print("=== MEGA FİLTRELEME DENEYİ BAŞLIYOR ===")

    # 1. Veri
    df = load_raw_data()
    X, y = split_features_target(df)

    # 2. Veri Sütunlarını Sayıya Çevir 
    # Filtrelemeden ÖNCE tüm string sütunları sayıya çeviriyoruz.
    print(" Kategorik veriler filtreleme için sayıya çevriliyor...")
    
    # X bir DataFrame olduğu için sütunlarını gezebiliriz
    X_numeric = X.copy()
    for col in X_numeric.columns:
        # Eğer sütun tipi 'object' (yazı) ise veya kategori ise
        if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
            le_temp = LabelEncoder()
            X_numeric[col] = le_temp.fit_transform(X_numeric[col].astype(str))
            
    # Artık X_numeric tamamen sayılardan oluşuyor. Filtreler bunu sever!
    # ----------------------------------

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = list(range(len(le.classes_)))
    label_names = le.classes_ # ['Insomnia', 'None', 'Sleep Apnea']

    # 2. Test Edilecek Modeller 
    models = get_classical_models()
    print(f"Klasik modeller alındı: {list(models.keys())}")
    deep_models = get_deep_models()
    print(f"Derin öğrenme modelleri alındı: {list(deep_models.keys())}")

    models.update(deep_models) # Klasik ve derin öğrenme modellerini birleştir

    # 3. FİLTRE LİSTESİ (Hepsi Burada!)
    filter_methods = [
        "none", "lowpass", "wavelet", "moving_average", "ema",
        "savgol", "loess", "spline", "kalman", 
        "median", "hampel", "gaussian"
    ]
    
    output_dir = "results/tables-filtered"
    rows = []

    for filter_name in filter_methods:
        print(f"\n FILTRE: {filter_name.upper()} uygulanıyor...")
        
        try:
            # Filtrele
            X_filtered = apply_filters(X_numeric, method=filter_name)
            
            # Split
            X_train, X_test, y_train, y_test = make_train_test_split(X_filtered, y_encoded)
            preprocessor = build_preprocessor(X_train)

            for model_name, model in models.items():
                print(f"  -> {model_name}...", end=" ")
                
                clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                y_proba = None
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_test)

                # --- ROC CURVE ICIN HAM VERIYI KAYDET ---
                if y_proba is not None:
                    roc_raw_df = pd.DataFrame(y_proba, columns=[f"proba_class_{i}" for i in labels])
                    roc_raw_df["true_label"] = y_test
                    roc_raw_df["base_model"] = model_name
                    roc_raw_df["filter_method"] = filter_name

                    roc_dir = "results/roc_raw"
                    os.makedirs(roc_dir, exist_ok=True)

                    roc_filename = f"roc_raw_{model_name}_{filter_name}.csv"
                    roc_raw_path = os.path.join(roc_dir, roc_filename)
                    roc_raw_df.to_csv(roc_raw_path, index=False)
                
                # Kaydet
                row = make_metrics_row(f"{model_name}_{filter_name}", y_test, y_pred, labels, y_proba)
                row["filter_method"] = filter_name
                row["base_model"] = model_name
                rows.append(row)
                print(f"F1: {row['f1_macro']:.4f}")

                # --- CONFUSION MATRIX KAYDETME ---
                cm_df = confusion_matrix_df(y_test, y_pred, labels=labels)
                cm_df.index = label_names
                cm_df.columns = label_names
                
                # Dosya ismi: confusion_matrix_LSTM_wavelet.csv gibi olacak
                cm_filename = f"confusion_matrix_{model_name}_{filter_name}.csv"
                cm_path = os.path.join(output_dir, cm_filename)
                cm_df.to_csv(cm_path, index=True)

        except Exception as e:
            print(f" HATA ({filter_name}): {e}")

    # Sonuçları Kaydet
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(by="f1_macro", ascending=False)
    
    out_path = "results/tables-filtered/filtered_metrics_all.csv"
    results_df.to_csv(out_path, index=False)

    print("\n=== DENEY SONUCU  ===")
    print(results_df[["base_model", "filter_method", "f1_macro"]].head(5))

if __name__ == "__main__":
    main()