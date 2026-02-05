import os
import pandas as pd
import sys
import numpy as np
from pathlib import Path

# Proje yolunu ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # Sklearn'den direkt alalım, garanti olsun
from sklearn.base import clone
from src.data_loader import load_raw_data
from src.preprocessing import build_preprocessor, split_features_target
from src.filters.filters import apply_filters
from src.models.classical import get_classical_models
from src.models.deep import get_deep_models
from src.evaluation import make_metrics_row, confusion_matrix_df

# === 1. RANDOMNESS (RASTGELELİK) ÇÖZÜMÜ ===
# Küresel tohumları sabitleyelim. Artık sonuçlar oynamayacak.
SEED = 42
np.random.seed(SEED)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
except ImportError:
    pass

def main():
    print("=== MEGA FİLTRELEME DENEYİ (GÜVENLİ MOD) BAŞLIYOR ===")

    # 1. Veri Yükleme
    df = load_raw_data()
    X, y = split_features_target(df)

    # 2. Kategorik -> Sayısal Dönüşüm
    # Bu kısım sızıntı yaratmaz, çünkü satır bazlıdır (Row-wise).
    print(" Kategorik veriler sayıya çevriliyor...")
    X_numeric = X.copy()
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
            le_temp = LabelEncoder()
            X_numeric[col] = le_temp.fit_transform(X_numeric[col].astype(str))

    # Hedef Değişkeni Kodla
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = list(range(len(le.classes_)))
    label_names = le.classes_

    # === 2. LEAKAGE (SIZINTI) ÇÖZÜMÜ ===
    # Filtrelemeden ÖNCE bölüyoruz. 
    # stratify=y_encoded ile sınıf dengesini koruyoruz.
    # random_state=SEED ile her seferinde AYNI hastaların seçilmesini garantiliyoruz.
    print(" Veri Eğitim ve Test olarak ayrılıyor (Önce Split)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_numeric, y_encoded, 
        test_size=0.2, 
        random_state=SEED, 
        stratify=y_encoded
    )

    # Modelleri Al
    models = get_classical_models()
    deep_models = get_deep_models()
    models.update(deep_models)

    filter_methods = [
        "none", "lowpass", "wavelet", "moving_average", "ema",
        "savgol", "loess", "spline", "kalman", 
        "median", "hampel", "gaussian"
    ]
    
    rows = []
    roc_rows = []

    for filter_name in filter_methods:
        print(f"\n FILTRE: {filter_name.upper()} uygulanıyor...")
        
        try:
            # === KRİTİK NOKTA ===
            # Train ve Test'i AYRI AYRI filtreliyoruz.
            # X_test verisi filtrelenirken X_train'i görmüyor. Sızıntı YOK.
            if filter_name != "none":
                X_train_filt = apply_filters(X_train_raw, method=filter_name)
                X_test_filt = apply_filters(X_test_raw, method=filter_name)
            else:
                X_train_filt = X_train_raw.copy()
                X_test_filt = X_test_raw.copy()
            
            # Preprocessor (Scaler) sadece Train'e fit edilir
            preprocessor = build_preprocessor(X_train_filt) # type: ignore

            for model_name, model in models.items():
                try:
                    # Her iterasyonda taze bir model kopyası oluştur
                    model_clone = clone(model)

                    # Model içi rastgeleliği sabitle
                    if hasattr(model_clone, "random_state"):
                        model_clone.random_state = SEED

                    # Pipeline
                    clf = Pipeline(
                        steps=[("preprocess", preprocessor), ("model", model_clone)]
                    )

                    # Eğit
                    clf.fit(X_train_filt, y_train)

                    # Tahmin
                    y_pred = clf.predict(X_test_filt)

                    y_proba = None
                    if hasattr(clf, "predict_proba"):
                        y_proba = clf.predict_proba(X_test_filt)

                    # ROC ham verisi
                    if y_proba is not None:
                        for i in range(len(y_test)):
                            roc_row = {
                                "base_model": model_name,
                                "filter_method": filter_name,
                                "true_label": int(y_test[i])
                            }
                            for c in range(y_proba.shape[1]):
                                roc_row[f"proba_class_{c}"] = float(y_proba[i, c])
                            roc_rows.append(roc_row)

                    # Metrikler
                    row = make_metrics_row(
                        f"{model_name}_{filter_name}",
                        y_test,
                        y_pred,
                        labels,
                        y_proba
                    )
                    row["filter_method"] = filter_name
                    row["base_model"] = model_name
                    rows.append(row)

                    # Confusion matrix
                    cm_df = confusion_matrix_df(y_test, y_pred, labels=labels)
                    cm_df.index = label_names
                    cm_df.columns = label_names
                    os.makedirs("results/tables-filtered", exist_ok=True)
                    cm_df.to_csv(
                        f"results/tables-filtered/confusion_matrix_{model_name}_{filter_name}.csv"
                    )

                except Exception as model_error:
                    print(f"   Model HATA ({model_name}): {model_error}")

        except Exception as e:
            print(f" HATA ({filter_name}): {e}")

    # Sonuçları Kaydet (Sıralama kodu)
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by=["f1_macro", "base_model", "filter_method"],
        ascending=[False, True, True]
    )
    
    out_path = "results/tables-filtered/filtered_metrics_all.csv"
    results_df.to_csv(out_path, index=False)

    # --- TEK ROC RAW DOSYASINI KAYDET ---
    roc_dir = "results/roc_raw"
    os.makedirs(roc_dir, exist_ok=True)
    roc_df = pd.DataFrame(roc_rows)
    roc_df.to_csv(os.path.join(roc_dir, "roc_raw_all.csv"), index=False)

    print("\n=== GÜVENLİ DENEY SONUCU (TOP 10) ===")
    print(results_df[["base_model", "filter_method", "f1_macro"]].head(10))

if __name__ == "__main__":
    main()