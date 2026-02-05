# src/filters/filters.py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import UnivariateSpline
import pywt
import statsmodels.api as sm

# --- YARDIMCI FONKSİYONLAR ---

def hampel_filter_pandas(input_series, window_size=5, n_sigmas=3):
    """
    Hampel Filtresi (Optimize Edilmiş):
    For döngüsü yerine Pandas rolling fonksiyonları kullanılarak hızlandırılmıştır.
    """
    # Pandas serisine çevir (eğer değilse)
    if not isinstance(input_series, pd.Series):
        input_series = pd.Series(input_series)
    
    k = 1.4826  # Gaussian dağılım ölçek sabiti
    
    # Kayan medyan (Rolling Median)
    rolling_median = input_series.rolling(window=window_size, center=True).median()
    
    # MAD (Median Absolute Deviation) hesaplama
    # MAD = median(|x - median|)
    # Pandas rolling ile MAD doğrudan yok, ama şu şekilde yaklaşık ve hızlı hesaplanabilir:
    # Veya tam doğruluk için lambda kullanılır (biraz yavaştır ama for döngüsünden hızlıdır)
    # Burada hız için rolling.std() yerine, MAD mantığına sadık kalıyoruz:
    
    difference = np.abs(input_series - rolling_median)
    rolling_mad = difference.rolling(window=window_size, center=True).median()  # type: ignore
    
    threshold = n_sigmas * k * rolling_mad
    
    # Aykırı değerleri tespit et
    outlier_idx = difference > threshold
    
    # Seriyi kopyala ve değiştir
    new_series = input_series.copy()
    new_series[outlier_idx] = rolling_median[outlier_idx]
    
    # Baş ve sondaki NaN değerleri (pencere nedeniyle oluşan) orijinal veriyle doldur
    new_series.fillna(input_series, inplace=True)
    
    return new_series.values

def kalman_filter_1d(data, process_variance=1e-5, measurement_variance=0.1):
    """
    Kalman Filtresi: 1D Basit Uygulama.
    """
    n_iter = len(data)
    sz = (n_iter,)

    xhat = np.zeros(sz)      # Tahmin
    P = np.zeros(sz)         # Hata kovaryansı
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)         # Kazanç

    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # Tahmin (Predict)
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + process_variance

        # Güncelleme (Update)
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

# --- ANA FONKSİYON ---

def apply_filters(X, method="none"):
    """
    Verilen veri setine seçilen gelişmiş filtreyi uygular.
    X: Pandas DataFrame veya NumPy Array (Satırlar: Örnekler, Sütunlar: Öznitelikler)
    method: Uygulanacak filtre yöntemi.
    """
    # 1. Veri Tipini Hazırla
    if isinstance(X, pd.DataFrame):
        data = X.values.astype(float)
        columns = X.columns
        is_df = True
    else:
        data = X.astype(float)
        columns = None
        is_df = False
    
    # Çıktı matrisi (Kopyası üzerinde çalışacağız)
    X_filtered = data.copy()
    rows, cols = data.shape

    # --- FİLTRE SEÇİMİ ---

    if method == "none":
        pass # Hiçbir şey yapma

    # 1. LOW-PASS (Butterworth)
    elif method == "lowpass":
        try:
            b, a = butter(2, 0.1, btype='low', analog=False) # type: ignore
            X_filtered = filtfilt(b, a, data, axis=0)
        except Exception as e:
            print(f"Lowpass hatası: {e}")

    # 2. WAVELET (DALGACIK) - Kritik Düzeltmeler Yapıldı
    elif method == "wavelet":
        for i in range(cols):
            signal = data[:, i]
            
            # Ayrıştırma (Decomposition)
            # 'level=2' genelde yeterlidir, sinyal uzunluğuna göre artırılabilir
            coeffs = pywt.wavedec(signal, 'db4', level=2)
            
            # Eşikleme (Thresholding)
            threshold = 0.04
            coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            
            # Geri Oluşturma (Reconstruction)
            reconstructed = pywt.waverec(coeffs_thresh, 'db4')
            
            # BOYUT EŞİTLEME (Padding/Trimming)
            # Wavelet dönüşümü bazen boyutu 1-2 piksel değiştirebilir.
            if len(reconstructed) > rows:
                reconstructed = reconstructed[:rows]
            elif len(reconstructed) < rows:
                reconstructed = np.pad(reconstructed, (0, rows - len(reconstructed)), mode='edge')
            
            X_filtered[:, i] = reconstructed

    # 3. MOVING AVERAGE
    elif method == "moving_average":
        df_temp = pd.DataFrame(data)
        # min_periods=1 kenarlarda NaN oluşmasını engeller
        X_filtered = df_temp.rolling(window=5, center=True, min_periods=1).mean().values

    # 4. EMA
    elif method == "ema":
        df_temp = pd.DataFrame(data)
        X_filtered = df_temp.ewm(alpha=0.3, adjust=False).mean().values

    # 5. SAVITZKY-GOLAY
    elif method == "savgol":
        if rows > 9:
            X_filtered = savgol_filter(data, window_length=9, polyorder=3, axis=0)
        else:
            # Veri çok kısaysa filtreyi pas geç
            pass

    # 6. LOESS (Dikkat: Büyük veride yavaştır)
    elif method == "loess":
        # Sadece ilk 1000 örnek için falan yapmak gerekebilir çok büyükse
        for i in range(cols):
            # frac=0.1 verinin %10'unu lokal pencere olarak alır
            smoothed = sm.nonparametric.lowess(data[:, i], np.arange(rows), frac=0.1)
            X_filtered[:, i] = smoothed[:, 1]

    # 7. SPLINE
    elif method == "spline":
        x_axis = np.arange(rows)
        for i in range(cols):
            try:
                spl = UnivariateSpline(x_axis, data[:, i], s=rows*10)
                X_filtered[:, i] = spl(x_axis)
            except:
                pass

    # 8. KALMAN
    elif method == "kalman":
        for i in range(cols):
            X_filtered[:, i] = kalman_filter_1d(data[:, i])

    # 9. MEDIAN
    elif method == "median":
        # size=(5,1) demek dikey (satırlar boyu) filtrele demek
        X_filtered = median_filter(data, size=(5, 1))

    # 10. HAMPEL (Optimize Edilmiş)
    elif method == "hampel":
        for i in range(cols):
            X_filtered[:, i] = hampel_filter_pandas(data[:, i])

    # 11. GAUSSIAN
    elif method == "gaussian":
        X_filtered = gaussian_filter1d(data, sigma=2, axis=0)

    else:
        print(f"Uyarı: Bilinmeyen filtre '{method}'. Orijinal veri döndürülüyor.")

    # 3. Çıktı Formatı
    if is_df:
        return pd.DataFrame(X_filtered, columns=columns)
    else:
        return X_filtered