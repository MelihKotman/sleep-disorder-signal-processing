# experiments/run_baseline_smote.py
# SMOTE ile dengelenmiş veri seti üzerinde modelleri çalıştırma betiği.

import os
import pandas as pd
import sys
from pathlib import Path

# Proje kök dizinini yola ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SMOTE Kütüphanesi
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_raw_data
from src.preprocessing import (
    split_features_target,
    build_preprocessor,
    make_train_test_split
)
from src.models.classical import get_classical_models
from src.evaluation import (
    make_metrics_row,
    confusion_matrix_df,
)
from src.models.deep import get_deep_models

def safe_predict_proba(clf, X_test):
    """
    Eğer sınıflandırıcı olasılık tahmini yapabiliyorsa predict_proba kullanır.
    Yapamıyorsa direkt None döner.
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)
    else:
        return None

def main():
    print("=== BASELINE + SMOTE ÇALIŞTIRILIYOR ===")

    # 1. Ham veriyi yükle
    df = load_raw_data()
    print(f"Veri seti yüklendi. Veri setinin şekli: {df.shape}")

    # 2. Girdi özellikleri ve hedef değişkeni ayır
    X, y = split_features_target(df)
    
    # 3. Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_names = le.classes_
    labels = list(range(len(label_names)))
    print(f"Hedef değişken etiketleri: {label_names}")
    print(f"Sınıflar Sayısallaştırıldı: {dict(zip(labels,label_names))}")

    # 4. Eğitim ve test verisi ayır
    # DİKKAT: Test setine ASLA SMOTE uygulanmaz!
    X_train, X_test, y_train, y_test = make_train_test_split(X, y_encoded)
    print(f"Eğitim ve test verisi ayrıldı. Eğitim: {X_train.shape}, Test: {X_test.shape}")

    # 5. ÖN İŞLEME VE SMOTE (Pipeline yerine manuel işlem yapıyoruz)
    # Çünkü SMOTE, Pipeline içinde doğrudan çalışmaz (fit_resample gerektirir)
    
    preprocessor = build_preprocessor(X_train)
    
    # Veriyi önce sayısal hale getir (Transform)
    print("Veri ön işleniyor (One-Hot + Scaling)...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test) # Test setine sadece transform!

    # Şimdi SMOTE uygulama zamanı (Sadece Train setine)
    print("⚡ SMOTE ile veri dengeleniyor...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train) # type: ignore
    
    print(f"SMOTE Sonrası Train Şekli: {X_train_resampled.shape}")
    # Sınıf dağılımı kontrolü
    unique, counts = pd.Series(y_train_resampled).value_counts().index, pd.Series(y_train_resampled).value_counts().values # type: ignore
    print(f"Yeni Sınıf Dağılımı: {dict(zip(unique, counts))}")

    # 6. Modelleri Al
    models = get_classical_models()
    deep_models = get_deep_models()
    models.update(deep_models) # Hepsini birleştir
    
    print(f"Test edilecek modeller: {list(models.keys())}")

    
    rows = [] 

    for name, model in models.items():
        print(f"\n--- Model (SMOTE): {name} eğitiliyor... ---")

        # Burada Pipeline kullanmıyoruz çünkü veriyi yukarıda manuel işledik.
        # Direkt işlenmiş ve dengelenmiş veriyi veriyoruz.
        model.fit(X_train_resampled, y_train_resampled)

        # Tahmin (İşlenmiş test verisi ile)
        y_pred = model.predict(X_test_transformed)
        y_proba = safe_predict_proba(model, X_test_transformed)

        # İsimlerin sonuna _SMOTE ekleyelim ki karışmasın
        model_display_name = f"{name}_SMOTE"

        row = make_metrics_row(
            model_name = model_display_name,
            y_true = y_test,
            y_pred = y_pred,
            labels = labels,
            y_proba = y_proba
        )
        rows.append(row)

        auc_val = row.get("roc_auc_ovr_macro", None)
        auc_str = f"{auc_val:.4f}" if (auc_val is not None and pd.notna(auc_val)) else "NA"

        print(f"Değerlendirme Sonuçları :")
        print(f"{model_display_name} -> acc={row['accuracy']:.4f}, f1={row['f1_macro']:.4f}, auc={auc_str}")

        # Karışıklık matrisini kaydet
        cm_df = confusion_matrix_df(y_test, y_pred, labels=labels)
        cm_df.index = label_names
        cm_df.columns = label_names
        # Dosya isminde modelin orijinal adını kullanalım
        cm_path = f"results/tables-baseline-smote/confusion_matrix_{name}_SMOTE.csv"
        cm_df.to_csv(cm_path, index=True)

    # 7. Sonuçları Kaydet ve İstenen Sütun Sırasına Sok
    results_df = pd.DataFrame(rows)
    
    # SENİN İSTEDİĞİN SÜTUN SIRALAMASI
    desired_columns = [
        "model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "specificity_macro", "roc_auc_ovr_macro", "roc_auc_ovr_weighted"
    ]
    
    # Sadece bu sütunları al ve f1_macro'ya göre sırala
    results_df = results_df[desired_columns].sort_values(by="f1_macro", ascending=False)

    # Dosyayı kaydet
    out_path = "results/tables-baseline-smote/baseline_smote_metrics.csv"
    results_df.to_csv(out_path, index=False)

    print("\n=== SMOTE İŞLEMİ TAMAMLANDI ===")
    print(results_df)
    print(f"\nSMOTE metrikleri '{out_path}' dosyasına kaydedildi.")
    print(f"Karışıklık matrisleri 'results/tables/' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()