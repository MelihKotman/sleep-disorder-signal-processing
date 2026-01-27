 # src/filters/filters.py
 # Gelişmiş sinyal işleme filtreleri için fonksiyonlar
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter, lfilter
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import UnivariateSpline
import pywt
import statsmodels.api as sm


# --- YARDIMCI FONKSİYONLARI ---
# Hampel filtresi - For döngüsü ile uygulanmış versiyon
def hampel_filter_forloop(input_series, window_size=5, n_sigmas= 3):
    """
    Hampel Filtresi: Zaman serisi verilerindeki aykırı değerleri tespit etmek ve düzeltmek için kullanılır.
    Parametreler: 
    - input_series: Giriş zaman serisi verisi (Pandas Series).
    - window_size: Pencere boyutu (tek sayı olmalı).
    - n_sigmas: Aykırı değerleri tanımlamak için kullanılan standart sapma sayısı.
    """
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # Gaussian dağılım için sabit ölçek

    # Kayan Pencere (Rolling Window)
    for i in range(window_size, n - window_size):
        x0 = np.median(input_series[(i - window_size): (i + window_size)]) # Merkezi eğilim ölçüsü
        S0 = k * np.median(np.abs(input_series[(i - window_size): (i + window_size)] - x0)) # Dağılım ölçüsü (MAD)
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0 # Aykırı değeri merkezi eğilim ile değiştir

    return new_series

# Kalman Filtresi - Basit bir uygulama
def kalman_filter_1d(data, process_variance=1e-5, measurement_variance = 0.1):
    """
    Kalman Filtresi: Gürültülü zaman serisi verilerini düzeltmek için kullanılan bir algoritma.
    Parametreler:
    - data: Giriş zaman serisi verisi (1D NumPy array).
    - process_variance: Süreç gürültüsü varyansı.
    - measurement_variance: Ölçüm gürültüsü varyansı.
    """
    n_iter = len(data) # Veri uzunluğu
    sz = (n_iter,) # Durum vektörünün boyutu

    # Durum (State) ve Tahmin Başlatma
    xhat = np.zeros(sz)      # Tahmin edilen değer (state estimate)
    P = np.zeros(sz)         # Tahmin hatası kovaryansı (uncertainty)
    xhatminus = np.zeros(sz) # Önceki tahmin
    Pminus = np.zeros(sz)    # Önceki tahmin hatası kovaryansı
    K = np.zeros(sz)         # Kalman kazancı

    # Başlangıç değerleri
    xhat[0] = data[0] 
    P[0] = 1.0 # Başlangıç belirsizliği

    for k in range(1, n_iter):
        # Zaman Güncellemesi (Predict)
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k - 1] + process_variance

        # Ölçüm Güncellemesi (Update)
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance) # Kalman kazancı
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k]) # Güncellenmiş tahmin
        P[k] = (1 - K[k]) * Pminus[k] # Güncellenmiş belirsizlik

    return xhat

def apply_filters(X, method= "none"):
    """
    Verilen veri setine seilmiş gelişmiş filtreyi uygular.
    X: Pandas DataFrame veya NumPy Array
    method: Uygulanacak filtre yöntemi. Seçenekler:
        - "none": Filtre uygulanmaz, orijinal veri döndürülür.
        - "hampel_forloop": Hampel filtresi (for döngüsü ile).
        - "kalman_1d": 1D Kalman filtresi.
    """
    # Veri tipini kontrol et ve NumPy array'e çevir
    if isinstance(X,pd.DataFrame):
        data = X.values.astype(float) #Hesaplamalar için float'a çevir
        columns = X.columns
        is_df = True
    
    else:
        data = X.astype(float)
        is_df = False
    
    # Çıktı matrisi (Orijinalin kopyası)
    X_filtered = data.copy()
    rows, cols = data.shape

    # 1. LOW-PASS (Butterworth IIR)
    if method == "lowpass":
        # Order 2, Cutoff 0.1 (Normalize frekans)
        b, a = butter(2, 0.1, btype='low', analog=False) # type: ignore
        X_filtered = filtfilt(b, a, data, axis=0)

    # 2. WAVELET DENOISING
    elif method == "wavelet":
        # Her sütun için ayrı ayrı uygula
        for i in range(cols):
            signal = data[:, i]
            # Wavelet ayrıştırma (db4 dalgacığı kullan)
            coeffs = pywt.wavedec(signal, 'db4', level=2)
            # Yumuşak eşikleme (Soft Thresholding) - Gürültüyü sil
            threshold = 0.04 # Deneme yanılma ile bulunabilir
            coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            # Geri birleştirme
            X_filtered[:, i] = pywt.waverec(coeffs_thresh, 'db4')
            # Boyut uyuşmazlığı olursa (Wavelet bazen 1-2 piksell ekler) kırp
            if len(X_filtered[:, i]) > rows:
                 X_filtered[:, i] = X_filtered[:, i][:rows]

    # 3. MOVING AVERAGE (MA)
    elif method == "moving_average":
        # Pandas rolling fonksiyonu çok hızlıdır
        df_temp = pd.DataFrame(data)
        X_filtered = df_temp.rolling(window=5, center=True, min_periods=1).mean().values

    # 4. EXPONENTIAL MOVING AVERAGE (EMA)
    elif method == "ema":
        df_temp = pd.DataFrame(data)
        # alpha: Yumuşatma faktörü (düşükse daha pürüzsüz)
        X_filtered = df_temp.ewm(alpha=0.3, adjust=False).mean().values

    # 5. SAVITZKY-GOLAY
    elif method == "savgol":
        # Pencere 9, Polinom derecesi 3
        if rows > 9:
            X_filtered = savgol_filter(data, window_length=9, polyorder=3, axis=0)
        else:
            print("Veri çok kısa, SavGol atlandı.")

    # 6. LOESS (Local Regression) - DİKKAT: Yavaştır
    elif method == "loess":
        # Her sütun için ayrı ayrı
        for i in range(cols):
            # frac: Verinin ne kadarını lokal olarak kullanacağı
            # lowess fonksiyonu (x, y) döner, biz sadece y'yi ([:, 1]) alıyoruz
            smoothed = sm.nonparametric.lowess(data[:, i], np.arange(rows), frac=0.1)
            X_filtered[:, i] = smoothed[:, 1]

    # 7. SPLINE SMOOTHING
    elif method == "spline":
        x_axis = np.arange(rows)
        for i in range(cols):
            # s: Smoothing factor (daha yüksek = daha düz)
            spl = UnivariateSpline(x_axis, data[:, i], s=rows*10) 
            X_filtered[:, i] = spl(x_axis)

    # 8. KALMAN FILTER
    elif method == "kalman":
        for i in range(cols):
            X_filtered[:, i] = kalman_filter_1d(data[:, i])

    # 9. MEDIAN FILTER
    elif method == "median":
        X_filtered = median_filter(data, size=(5, 1))

    # 10. HAMPEL FILTER (Outlier Removal)
    elif method == "hampel":
        for i in range(cols):
            X_filtered[:, i] = hampel_filter_forloop(data[:, i])

    # 11. GAUSSIAN FILTER
    elif method == "gaussian":
        X_filtered = gaussian_filter1d(data, sigma=2, axis=0)

    # 12. FİLTRE YOK
    elif method == "none":
        pass

    else:
        print(f"Bilinmeyen filtre yöntemi: {method}. 'none' uygulandı.")

    # Formatı geri döndür
    if is_df:
        return pd.DataFrame(X_filtered, columns=columns) # type: ignore
    else:
        return X_filtered