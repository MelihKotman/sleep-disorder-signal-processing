# src/ evaluation.py
# Model değerlendirme metrikleri ve fonksiyonları buraya yazılır.

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, # Doğruluk Skoru
    precision_recall_fscore_support, # Kesinlik, Duyarlılık, F1 Skoru
    confusion_matrix, # Karışıklık Matrisi
    roc_auc_score # AUC-ROC Skoru
)
from sklearn.preprocessing import LabelBinarizer

# Temel değerlendirme metriklerini hesaplayan fonksiyon
def compute_basic_metrics(y_true , y_pred): #y_true: gerçek etiketler, y_pred: tahmin edilen etiketler
    """
    Temel sınıflandırma metriklerini hesaplar:
    - Doğruluk (Accuracy)
    - Kesinlik (Precision)
    - Duyarlılık (Recall)
    - F1 Skoru (F1-Score)
    """
    acc = accuracy_score(y_true, y_pred) # Doğruluk skoru

    # Makro ortalama ile kesinlik, duyarlılık, F1 skoru burada şöyle hesaplanır:
    # Her sınıf için ayrı ayrı hesaplanır ve sonra ortalaması alınır.
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average = "macro", zero_division=0
    )

    # Ağırlıklı ortalama ile kesinlik, duyarlılık, F1 skoru burada şöyle hesaplanır:
    # Her sınıfın örnek sayısına göre ağırlıklandırılarak hesaplanır.
    p_w, r_w, f_w, _ = precision_recall_fscore_support( 
        y_true, y_pred, average = "weighted", zero_division=0
    )
    return {
        "accuracy" : acc,
        "precision_macro" : p_macro,
        "recall_macro" : r_macro,
        "f1_macro" : f_macro,
        "precision_weighted" : p_w,
        "recall_weighted" : r_w,
        "f1_weighted" : f_w
    }

# Makro ortalama ile özgüllük (specificity) hesaplayan fonksiyon
def compute_specificity_macro(y_true, y_pred, labels):
    """
    Makro ortalama ile özgüllük (specificity) hesaplar.
    Özgüllük, negatif sınıfların doğru tahmin edilme oranıdır.
    Confusion matrix : TN / (TN + FP) formülü ile hesaplanır.
    Bize her sınıf için özgüllük değerini döner ve makro ortalamasını alır.

    # TP: True Positive (Gerçek Pozitif)
    # TN: True Negative (Gerçek Negatif)
    # FP: False Positive (Yanlış Pozitif)
    # FN: False Negative (Yanlış Negatif)

    Her sınıf için (Örn: Uyku Apnesi) tek tek bakar:
    "Apnesi olmayanları ne kadar doğru ayıkladık?". 
    Sonra bunların ortalamasını alır.
    """
    cm = confusion_matrix(y_true, y_pred, labels = labels) # Karışıklık matrisi
    n_classes = cm.shape[0] # Sınıf sayısı

    specificities = [] # Her sınıf için özgüllük değerlerini tutacak liste
    # Her sınıf için özgüllük hesapla
    for i in range(n_classes):
        tp = cm[i, i] # Doğru pozitif
        fn = np.sum(cm[i, :]) - tp # Yanlış negatif
        fp = np.sum(cm[:, i]) - tp # Yanlış pozitif
        tn = np.sum(cm) - (tp + fn + fp) # Doğru negatif

        denom = tn + fp # Özgüllük paydası
        spec = tn / denom if denom != 0 else 0.0 # Özgüllük hesapla, payda 0 ise 0.0 ata
        specificities.append(spec) # Listeye ekle   
    
    return float(np.mean(specificities)) # Makro ortalama özgüllük döner

# AUC-ROC skoru hesaplayan fonksiyon
def compute_auc_ovr(y_true, y_proba, labels): # y_true: gerçek etiketler, y_proba: tahmin edilen olasılıklar
    """
    One-vs-Rest (OvR) stratejisi ile AUC-ROC skoru hesaplar.
    Çok sınıflı sınıflandırma için kullanılır.
    Bize her sınıf için weighted AUC-ROC skorunu döner ve
    bunların makro ortalamasını alır.

    y_proba (n_samples, n_classes) şeklinde olasılık tahminleri içermelidir.
    """

    lb = LabelBinarizer() # Etiketleri ikili forma çevirir
    lb.fit(labels) # Tüm sınıf etiketlerini öğren

    y_true_bin = np.asarray(lb.transform(y_true)) # Gerçek etiketleri ikili forma çevir

    # İkili form için LabelBinarizer kullanarak (n,1)'den (n, 2)'ye çevir
    if y_true_bin.shape[1] == 1: # İkili sınıf durumu
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin]) # Negatif sınıfı da ekle
    
    # AUC macro ve weighted hesapla
    auc_macro = roc_auc_score(
        y_true_bin, y_proba,
        average = "macro",
        multi_class = "ovr"
    )
    auc_weighted = roc_auc_score(
        y_true_bin, y_proba,
        average = "weighted",
        multi_class = "ovr"
    )

    return float(auc_macro), float(auc_weighted) # Makro ve weighted AUC-ROC skorlarını döner (float olarak)

# Confusion matrix'i DataFrame olarak dönen fonksiyon
def confusion_matrix_df(y_true, y_pred, labels):
    """
    Karışıklık matrisini (confusion matrix) pandas DataFrame olarak döner.
    Satırlar gerçek etiketleri, sütunlar tahmin edilen etiketleri gösterir.
    """
    cm = confusion_matrix(y_true, y_pred, labels = labels) # Karışıklık matrisi
    return pd.DataFrame(cm, index = labels, columns = labels) # DataFrame olarak döner
 
# Model metriklerini tek bir satırda toplayan fonksiyon
def make_metrics_row(model_name, y_true, y_pred, labels, y_proba = None):
    """
    Tüm metrikleri tek bir satırda toplayan fonksiyon.
    Model adı, temel metrikler, özgüllük, AUC-ROC skorları (varsa) dahil edilir.
    """
    row = {"model" : model_name} # Model adı

    # Temel metrikleri hesapla ve satıra ekle
    row.update(compute_basic_metrics(y_true, y_pred))

    # Spesifikliği hesapla ve satıra ekle
    row["specificity_macro"] = compute_specificity_macro(y_true, y_pred, labels)

    # AUC-ROC skorlarını hesapla ve satıra ekle (olasılıklar verilmişse)
    if y_proba is not None: 
        # Olasılıklar verilmişse AUC-ROC skorlarını hesapla
        try: 
            auc_macro, auc_weighted = compute_auc_ovr(y_true, y_proba, labels)
            row["roc_auc_ovr_macro"] = auc_macro 
            row["roc_auc_ovr_weighted"] = auc_weighted
        # Herhangi bir hata olursa NaN ata
        except Exception: 
            row["roc_auc_ovr_macro"] = np.nan  
            row["roc_auc_ovr_weighted"] = np.nan
    # Olasılıklar verilmemişse NaN ata
    else:
        row["roc_auc_ovr_macro"] = np.nan  
        row["roc_auc_ovr_weighted"] = np.nan
    
    return row # Metrik satırını döner