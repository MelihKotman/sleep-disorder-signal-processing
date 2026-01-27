# src/models/deep.py
# Derin öğrenme modelleri buraya yazılır.
# LSTM ve 1D-CNN modelleri oluşturulur.

import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin # Sklearn uyumlu sınıflar için temel sınıflar

os.environ["KERAS_BACKEND"] = "tensorflow"

# Önce ana tensorflow'u çağırıyoruz
import tensorflow as tf
# Sonra keras'ı tf üzerinden değil, direkt çağırıp tf ile bağlıyoruz
import keras
from keras.models import Sequential # Keras Sequential modeli
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout, Input # Keras katmanları
from keras.optimizers import Adam # Adam optimizasyon algoritması
from keras.utils import to_categorical # One-Hot Encoding için yardımcı fonksiyon
    

class KerasBatchClassifier(BaseEstimator, ClassifierMixin):
    """
    Tensorflow / Keras tabanlı CNN ve LSTM modelllerini 
    Scikit-Learn Pipeline yapısına entegre etmek için Wrapper Sınıfı.
    """

    def __init__(
            self,
            model_type : str = "cnn", # Model türünü cnn olarak ayarla (cnn veya lstm)
            epochs: int = 50, # Eğitim epoch sayısı epoch şu demektir: tüm eğitim verisinin modele şu kadar kez gösterilmesi
            batch_size : int = 32, # Batch ise modelin her seferinde kaç örnekle güncelleneceği biz 32 olarak ayarladık
            verbose: int = 0, # Eğitim sırasında çıktı seviyesi 0 (sessiz) olarak ayarla
            random_state: int | None = None # Rastgelelik için sabit tohum (değer)
    ):
        self.model_type = model_type # Model türü (cnn veya lstm)
        self.epochs = epochs # Eğitim epoch sayısı
        self.batch_size = batch_size #  Batch boyutu
        self.verbose = verbose # Eğitim sırasında çıktı seviyesi
        self.random_state = random_state # Rastgelelik için sabit tohum

        self.model = None # Model başlangıçta None
        self.classes_ = None # Sınıflar başlangıçta None
        self.n_classes_ = None # Sınıf sayısı başlangıçta None
    
    # Verimizi 2 boyutludan 3 boyutlu hale getiren yardımcı fonksiyon
    def _reshape_X(self, X):
        """
        Tabular veriyi CCN/LSTM için 3D formata çevirir.
        (samples, features) -> (samples, features, 1)
        CNN ve LSTM modelleri 3D veri bekler.
        """
        return X.reshape((X.shape[0], X.shape[1], 1)) # Bu veri (374, 12, 1) olur.
    
    # Modeli oluşturma fonksiyonu
    def _build_model(self, n_features: int, n_classes: int):
        """
        CNN: Conv1D ile verinin üzerinde kayan bir pencere gibi gezerek özellikler arasındaki ilişkileri öğrenir.
             MaxPooling1D ile de gereksiz detayları atıp en önemli sinyalleri tutar.
        LSTM:Burada sütunları sırayla okuyarak, özellikler arasındaki gizli bağları (hafıza temelli) öğrenir.
             Softmax katmanıyla da bize olasılık tahminleri verir. Örneğin, bir veri noktasının 'Insomnia' olma olasılığı %70, 'Sleep Apnea' olma olasılığı %20, 'None' olma olasılığı %10 gibi.
        """
        model = Sequential() # Keras Sequential modeli başlat

        if self.model_type == "cnn":
            # CNN Modeli Konfigürasyonu
            model.add(Input(shape=(n_features, 1))) # Girdi katmanı (özellik sayısı, 1 kanal)
            model.add(Conv1D(filters=64, kernel_size=2, activation="relu")) # 1D Konvolüsyon katmanı (64 filtre, kernel boyutu 2, relu aktivasyonu)
            model.add(MaxPooling1D(pool_size= 2)) # Maksimum havuzlama katmanı (özellik sayısını azaltır)
            model.add(Flatten()) # Düzleştirme katmanı (çok boyutlu veriyi tek boyuta indirger)
            model.add(Dense(50, activation= "relu")) # Yoğun katman (50 nöron, relu aktivasyonu)
            model.add(Dense(n_classes, activation= "softmax")) # Çıkış katmanı (n_classes sınıf sayısı, softmax aktivasyonu)

        elif self.model_type == "lstm":
            # LSTM Modeli Konfigürasyonu
            model.add(Input(shape=(n_features, 1))) # Girdi katmanı (özellik sayısı, 1 kanal)
            model.add(LSTM(units= 50, activation= "relu", return_sequences= False)) # LSTM katmanı (50 birim, tek yönlü, çıktı dizisi yok)
            model.add(Dropout(0.2)) # Dropout katmanı (aşırı öğrenmeyi önlemek için %20 dropout)
            model.add(Dense(n_classes, activation= "softmax")) # Çıkış katmanı (n_classes sınıf sayısı, softmax aktivasyonu)
        
        else:
            # Eğer
            raise ValueError( #
                f"Geçersiz model_type: {self.model_type}."
                f"'cnn' veya 'lstm' olmalı."
            )
        
        model.compile(
            optimizer= Adam(learning_rate=0.001), # Adam optimizasyon algoritması (öğrenme hızı 0.001) # type: ignore
            loss= "categorical_crossentropy", # Kategorik çapraz entropi kayıp fonksiyonu (çok sınıflı sınıflandırma için)
            metrics= ["accuracy"] # Doğruluk metriği
        )
        return model

    # Modeli eğitme fonksiyonu burada başlar.
    def fit(self, X, y):
        """
        İlk önce sınıf saymadan başlar np.unique ile kaç çeşit hasstalık var (None : 0, Insomnia : 1, Sleep Apnea : 2).
        Sonrasında One-Hot Coding ile 0, 1, 2 etiketlerini [1,0,0], [0,1,0], [0,0,1] formatına çevirir ve dönüşümü to_categorical ile yapar.
        Ardından modeli inşaası için verinin kaç sütun olduğunu (n_features) ile bakıp modeli o an oluşturur.
        Son olarak modeli X verisini 3D'ye çevirip fit eder.
        X: (Samples, features)
        y: (integer encoded labels (0,1,2,...))
        """

        #Sınıfları Kaydet (sklearn uyumluluğu için)
        self.classes_ = np.unique(y) # Benzersiz sınıfları bul
        self.n_classes_ = len(self.classes_) # Sınıf sayısını bul

        # One Hot Encoding (Kategorik veriyi ikili matris formatına çevir)
        y_cat = to_categorical(y, num_classes= self.n_classes_)  # Örneğin, 3 sınıf için [0] -> [1,0,0], [1] -> [0,1,0], [2] -> [0,0,1]

        # Modeli oluştur
        n_features = X.shape[1] # Özellik sayısını al
        self.model = self._build_model(n_features, self.n_classes_) # Modeli oluşturma kısmı

        # Veriyi 3D formata çevir
        X_reshaped = self._reshape_X(X) # (samples, features, 1)

        # Modeli eğit
        self.model.fit(
            X_reshaped, # 3D formatta girdi verisi
            y_cat, # One-Hot kodlanmış hedef verisi
            epochs= self.epochs, # Eğitim epoch sayısı
            batch_size= self.batch_size, # Batch boyutu
            verbose= self.verbose # Eğitim sırasında çıktı seviyesi # type: ignore
        )
        return self
    
    # Tahminlerden en yüksek olasılığa sahip sınıfı seçme fonksiyonu
    def predict(self, X):
        """
        Eğitilmiş verinin üzerinden her sınıfın ihtimallerin en büyüğünü seçer.
        Misal 0.8 0.1 0.1 ise 0. sınıfı seçer.
        X: (Samples, features)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Önce fit() metodunu çağırın.")
        
        X_reshaped = self._reshape_X(X)
        probs = self.model.predict(X_reshaped, verbose = 0) # type: ignore # Her sınıf için olasılık tahminleri
        return np.argmax(probs,axis= 1) # En yüksek olasılığa sahip sınıf etiketini döndür
    
    # Tahminler için olasılık tahminleri alma fonksiyonu
    def predict_proba(self, X):
        """
        Eğitilmiş verinin üzerinden her sınıfın olasılık tahminlerini döner.
        X: (Samples, features)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Önce fit() metodunu çağırın.")
        X_reshaped = self._reshape_X(X)
        probs = self.model.predict(X_reshaped, verbose = 0) # type: ignore # Her sınıf için olasılık tahminleri
        return probs # Olasılık tahminlerini döndür
    
def get_deep_models():
    """
    Derin Öğrenme modellerini sözlük (dict) formatında döndürür.
    run_baseline.py içinde classical modellerle aynı şekilde kullanılabilir.
    """
    return {
        "CNN": KerasBatchClassifier(
            model_type = "cnn",
            epochs = 50,
            batch_size= 32,
            verbose= 0
        ),
        "LSTM": KerasBatchClassifier(
            model_type = "lstm",
            epochs =50,
            batch_size = 32,
            verbose = 0
        )
        }