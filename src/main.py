from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIG --------
# Ruta del archivo FITS (usa r"" para que no dé error con las barras de Windows)
fits_file = r"C:\Users\sergi\Documents\GitHub\Spectrum_Analyzer\datos\med-58859-NT054013S012745S01_sp12-013.fits"

# Rebin: combina N píxeles adyacentes (p.ej., 2–8). 1 = sin rebin
REBIN_FACTOR = 2

# Suavizado: ventana en puntos (impar) y orden del polinomio
SG_WINDOW = 101      # prueba 31, 51, 71 según SNR
SG_POLY   = 3

# Si no tienes scipy, se usará una media móvil de esta ventana
MOVING_AVG_WINDOW = 25

# Normalizar el continuo (True/False)
DO_CONTINUUM_NORM = False
# ------------------------

# >>> Diccionario de líneas de interés (ejemplos entre 4900–5400 Å) <<<
lines = {
    "Hβ": 4861.3,
    "Fe I 4957": 4957.6,
    "Fe I 5006": 5006.1,
    "Fe I 5051": 5051.6,
    "Mg I 5167": 5167.3,
    "Mg I 5172": 5172.7,
    "Mg I 5183": 5183.6,
    "Fe I 5198": 5198.7,
    "Fe I 5227": 5227.2,
    "Fe I 5247": 5247.1,
    "Fe I 5328": 5328.0,
    "Fe I 5371": 5371.5
}

# ---------- utilidades ----------
def valid_mask(flux, ivar):
    m = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    return m

def rebin_spectrum(wl, flux, ivar, factor=2):
    if factor <= 1:
        return wl, flux, ivar
    n = len(wl) // factor
    wl_r = wl[:n*factor].reshape(n, factor).mean(axis=1)
    var = 1.0 / ivar
    var_r = var[:n*factor].reshape(n, factor).mean(axis=1)
    flux_r = flux[:n*factor].reshape(n, factor).mean(axis=1)
    ivar_r = 1.0 / var_r
    return wl_r, flux_r, ivar_r

def try_savgol(y, window, poly):
    try:
        from scipy.signal import savgol_filter
        window = max(3, int(window) | 1)  # impar mínimo 3
        window = min(window, len(y) - (1 - (len(y)%2)))  # no exceder longitud
        if window < poly + 2:
            window = poly + 3
        return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
    except Exception:
        w = max(3, int(MOVING_AVG_WINDOW))
        if w % 2 == 0:
            w += 1
        k = np.ones(w) / w
        y_pad = np.pad(y, (w//2, w//2), mode="edge")
        return np.convolve(y_pad, k, mode="valid")

def running_percentile(y, win=301, q=90):
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
# ---------------------------------

# 1) leer FITS
with fits.open(fits_file) as hdul:
    data = hdul["COADD_B"].data  # si no existe, prueba "COADD_R"
    wl   = np.array(data["WAVELENGTH"][0], dtype=float)
    flux = np.array(data["FLUX"][0], dtype=float)
    ivar = np.array(data["IVAR"][0], dtype=float)

# 2) máscara básica usando IVAR
m = valid_mask(flux, ivar)
wl, flux, ivar = wl[m], flux[m], ivar[m]

# 3) (opcional) rebin para subir SNR
wl_r, flux_r, ivar_r = rebin_spectrum(wl, flux, ivar, factor=REBIN_FACTOR)

# 4) suavizado
flux_smooth = try_savgol(flux_r, window=SG_WINDOW, poly=SG_POLY)

# 5) (opcional) normalización de continuo
if DO_CONTINUUM_NORM:
    cont = running_percentile(flux_smooth, win=501, q=95)
    cont = np.where(cont <= 0, np.nanmedian(cont[cont>0]), cont)
    flux_plot = flux_smooth / cont
    ylab = "Flujo (normalizado)"
else:
    flux_plot = flux_smooth
    ylab = "Flujo"

# 6) gráfico (original rebin + suavizado / normalizado)
plt.figure(figsize=(12,5))
plt.plot(wl_r, flux_r, linewidth=0.5, alpha=0.6, label="Rebin")
plt.plot(wl_r, flux_plot, linewidth=1.0, label="Suavizado" + (" + normalizado" if DO_CONTINUUM_NORM else ""))

# >>> Añadir líneas de referencia <<<
for name, wavelength in lines.items():
    if 4900 <= wavelength <= 5400:   # solo dentro del rango visible
        plt.axvline(wavelength, color="red", linestyle="--", alpha=0.7)
        plt.text(wavelength+2, np.nanmax(flux_plot)*0.9, name,
                 rotation=90, color="red", fontsize=8)

plt.xlabel("Longitud de onda (Å)")
plt.ylabel(ylab)
plt.title("Espectro LAMOST — reducción de ruido")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
