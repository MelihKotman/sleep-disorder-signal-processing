# src/models/classical.py
# Klasik makine öğrenimi modelleri buraya yazılır.

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Karar Ağacı Sınıflandırıcı
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes Sınıflandırıcı
from sklearn.ensemble import RandomForestClassifier # Rastgele Orman Sınıflandırıcı
from sklearn.svm import SVC # Destek Vektör Makineleri Sınıflandırıcı
from sklearn.neural_network import MLPClassifier # Çok Katmanlı Algılayıcı (Yapay Sinir Ağı) Sınıflandırıcı
from sklearn.ensemble import VotingClassifier # Oylama Sınıflandırıcı
from xgboost import XGBClassifier # XGBoost Sınıflandırıcı

from src.config import RANDOM_STATE # Rastgelelik durumu

# Klasik modelleri döndüren fonksiyon
def get_classical_models():
    """
    Klasik makine öğrenim modellerini oluşturur ve bir sözlük olarak geri döner.
    DT, GNB, BT/RF, SVM, LR, MLP, XGB ve VCLF modelleri oluşturulur.
    Sırasıyla:
    - Decision Tree (Karar Ağacı)
    - Gaussian Naive Bayes
    - Random Forest (Rastgele Orman)
    - Support Vector Machine (Destek Vektör Makineleri)
    - Multi-Layer Perceptron (Çok Katmanlı Algılayıcı)
    - XGBoost (Extreme Gradient Boosting)
    - Voting Classifier (Oylama Sınıflandırıcı)
    """

    models = {
        "DT" : DecisionTreeClassifier(random_state = RANDOM_STATE), # Karar Ağacı
        "GNB" : GaussianNB(), # Gaussian Naive Bayes
        "RF": RandomForestClassifier(
            n_estimators = 300, # Ağaç sayısı
            random_state = RANDOM_STATE # Rastgelelik durumu
        ), # Rastgele Orman
        "SVM" : SVC(
            kernel = "rbf", # RBF çekirdek fonksiyonu
            probability = True, # Olasılık tahmini yap
            random_state = RANDOM_STATE # Rastgelelik durumu
        ), # Destek Vektör Makineleri
        "LR" : LogisticRegression(
            max_iter = 2000, # Maksimum iterasyon sayısı
            random_state = RANDOM_STATE # Rastgelelik durumu
        ), # Lojistik Regresyon
        "MLP" : MLPClassifier(
            hidden_layer_sizes = (64, 32), # Gizli katman boyutları
            max_iter = 2000, # Maksimum iterasyon sayısı
            random_state = RANDOM_STATE # Rastgelelik durumu
        ), # Çok Katmanlı Algılayıcı
        "XGB" : XGBClassifier(
            n_estimators = 400, # Ağaç sayısı
            learning_rate = 0.05, # Öğrenme hızı
            max_depth = 4, # Maksimum derinlik
            subsample = 0.9, # Alt örnekleme oranı
            colsample_bytree = 0.8, # Özellik alt örnekleme oranı
            objective = "multi:softprob", # Çok sınıflı sınıflandırma
            eval_metric = "mlogloss", # Değerlendirme metriği
            random_state = RANDOM_STATE, # Rastgelelik durumu
            n_jobs = -1 # Tüm çekirdekleri kullan
        ),
        "VCLF" : None # Oylama Sınıflandırıcı (sonradan eklenecek)
    }
    models["VCLF"] = VotingClassifier(
        estimators = [
            ("lr", models["LR"]), # Lojistik Regresyon
            ("svm", models["SVM"]), # Destek Vektör Makineleri
            ("rf", models["RF"]), # Rastgele Orman
        ],
        voting = "soft", # Yumuşak oylama
        n_jobs = -1 # Tüm çekirdekleri kullan
    )

    return models