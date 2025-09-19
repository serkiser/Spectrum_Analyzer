"""
Utilidades varias para procesamiento de espectros
"""

import numpy as np

def try_savgol(y, window, poly, moving_avg_window=35):
    """Intenta aplicar filtro Savitzky-Golay, falla a media móvil"""
    try:
        from scipy.signal import savgol_filter
        window = max(3, int(window) | 1)
        
        # Asegurar que window_length no exceda el tamaño de y
        if window > len(y):
            window = len(y) - 1 if len(y) % 2 == 0 else len(y)
            window = max(3, window)
        
        if window < poly + 2:
            window = poly + 3
            
        return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
    
    except Exception as e:
        print(f"Error con Savitzky-Golay: {e}. Usando media móvil...")
        
        w = max(3, int(moving_avg_window))
        if w % 2 == 0:
            w += 1
            
        # Asegurar que la ventana no sea mayor que los datos
        if w > len(y):
            w = len(y) - 1 if len(y) % 2 == 0 else len(y)
            w = max(3, w)
            
        k = np.ones(w) / w
        
        # Usar mode='constant' en lugar de 'edge' para arrays pequeños
        if len(y) < w:
            y_pad = np.pad(y, (w//2, w//2), mode='constant', constant_values=np.median(y))
        else:
            y_pad = np.pad(y, (w//2, w//2), mode='edge')
            
        return np.convolve(y_pad, k, mode="valid")

def running_percentile(y, win=301, q=90):
    """Calcula un percentil móvil para estimar el continuo"""
    win = max(51, int(win) | 1)
    if win >= len(y):
        return np.full_like(y, np.nanmedian(y))
    half = win // 2
    cont = np.empty_like(y)
    for i in range(len(y)):
        a = max(0, i - half)
        b = min(len(y), i + half + 1)
        cont[i] = np.nanpercentile(y[a:b], q)
    return cont

def enhance_line_detection(flux, enhancement_factor=1.5):
    """Realza las líneas espectrales en espectros ruidosos"""
    # Normalizar el flujo
    norm_flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
    
    # Aplicar transformación no lineal para realzar características
    enhanced_flux = np.power(norm_flux, enhancement_factor)
    
    # Reescalar al rango original
    return enhanced_flux * (np.max(flux) - np.min(flux)) + np.min(flux)