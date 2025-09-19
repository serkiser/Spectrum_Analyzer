"""
Módulo para análisis espectral
"""

import numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks, peak_widths
from scipy.stats import median_abs_deviation

def calculate_snr(flux, window=100):
    """Calcula la relación señal/ruido (SNR) del espectro"""
    n_segments = len(flux) // window
    snr_values = []
    
    for i in range(n_segments):
        segment = flux[i*window:(i+1)*window]
        signal = np.median(segment)
        noise = np.std(segment)
        if noise > 0:
            snr_values.append(signal / noise)
    
    return np.median(snr_values) if snr_values else 0

def measure_line_parameters(wavelengths, flux, line_center, window=10):
    """Mide parámetros importantes de una línea espectral"""
    mask = (wavelengths >= line_center - window) & (wavelengths <= line_center + window)
    wl_window = wavelengths[mask]
    flux_window = flux[mask]
    
    if len(flux_window) == 0:
        return None
    
    # Encontrar mínimo de flujo (máxima absorción)
    min_flux_idx = np.argmin(flux_window)
    observed_center = wl_window[min_flux_idx]
    min_flux = flux_window[min_flux_idx]
    
    # Calcular continuo local
    continuum_left = np.median(flux_window[:5])
    continuum_right = np.median(flux_window[-5:])
    continuum = (continuum_left + continuum_right) / 2
    
    # Calcular ancho equivalente
    equivalent_width = simpson(1 - flux_window/continuum, wl_window)
    
    # Calcular FWHM
    half_max = (continuum + min_flux) / 2
    left_idx = np.where(flux_window[:min_flux_idx] <= half_max)[0]
    right_idx = np.where(flux_window[min_flux_idx:] <= half_max)[0]
    
    if len(left_idx) > 0 and len(right_idx) > 0:
        left_wl = wl_window[left_idx[-1]]
        right_wl = wl_window[min_flux_idx + right_idx[0]]
        fwhm = right_wl - left_wl
    else:
        fwhm = np.nan
    
    # Calcular profundidad de la línea
    depth = 1 - (min_flux / continuum)
    
    return {
        'observed_center': observed_center,
        'equivalent_width': equivalent_width,
        'fwhm': fwhm,
        'depth': depth,
        'continuum_level': continuum
    }

def calculate_redshift(observed_wavelength, rest_wavelength):
    """Calcula el redshift a partir de una línea espectral"""
    return (observed_wavelength - rest_wavelength) / rest_wavelength

def robust_redshift_calculation(redshifts, sigma_clip=3.0):
    """Calcula un redshift robusto eliminando outliers"""
    if len(redshifts) == 0:
        return None, None, 0
    
    # Primera estimación usando mediana y MAD
    median_z = np.median(redshifts)
    mad_z = median_abs_deviation(redshifts)
    
    # Filtrar outliers
    filtered_redshifts = [z for z in redshifts if abs(z - median_z) < sigma_clip * mad_z]
    
    if len(filtered_redshifts) == 0:
        # Si todos son outliers, usar la mediana original
        return median_z, mad_z, len(redshifts)
    
    # Calcular media y desviación estándar de los valores filtrados
    mean_z = np.mean(filtered_redshifts)
    std_z = np.std(filtered_redshifts)
    
    return mean_z, std_z, len(filtered_redshifts)

def calculate_mg_fe_index(wavelengths, flux, mg_line=5175, fe_line=5270, window=20):
    """Calcula el índice Mg/Fe para estimar metalicidad"""
    mg_mask = (wavelengths >= mg_line - window) & (wavelengths <= mg_line + window)
    fe_mask = (wavelengths >= fe_line - window) & (wavelengths <= fe_line + window)
    
    mg_flux = np.mean(flux[mg_mask])
    fe_flux = np.mean(flux[fe_mask])
    
    return mg_flux / fe_flux

def estimate_temperature(hbeta_ew):
    """Estimación simple de temperatura a partir del ancho equivalente de Hβ"""
    if hbeta_ew < 2:
        return "Muy caliente (>10000 K)", 11000
    elif hbeta_ew < 4:
        return "Caliente (8000-10000 K)", 9000
    elif hbeta_ew < 6:
        return "Intermedia (6000-8000 K)", 7000
    else:
        return "Fría (<6000 K)", 5500

def find_emission_lines(wavelengths, flux, height_threshold=0.1, distance=10):
    """Encuentra líneas de emisión en el espectro"""
    # Normalizar el flujo para el detector de picos
    norm_flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
    
    # Encontrar picos (líneas de emisión)
    peaks, properties = find_peaks(norm_flux, height=height_threshold, distance=distance)
    
    # Calcular anchuras de los picos
    widths, width_heights, left_ips, right_ips = peak_widths(norm_flux, peaks, rel_height=0.5)
    
    # Convertir índices a longitudes de onda
    peak_wavelengths = wavelengths[peaks]
    fwhms = widths * (wavelengths[1] - wavelengths[0])  # Convertir a Å
    
    # Preparar resultados
    emission_lines = []
    for i, wl in enumerate(peak_wavelengths):
        emission_lines.append({
            'wavelength': wl,
            'strength': properties['peak_heights'][i],
            'fwhm': fwhms[i]
        })
    
    return emission_lines

def generate_spectral_report(wavelengths, flux, ivar, lines_dict, redshift_sigma_clip=2.0):
    """Genera un reporte con los parámetros espectrales más importantes"""
    report = {}
    
    # Información básica del espectro
    report['wavelength_range'] = {
        'min': float(np.min(wavelengths)),
        'max': float(np.max(wavelengths)),
        'delta': float(np.max(wavelengths) - np.min(wavelengths))
    }
    
    # Calcular SNR
    report['snr'] = float(calculate_snr(flux))
    
    # Medir parámetros para cada línea de absorción
    report['absorption_lines'] = {}
    redshifts = []  # Lista para almacenar todos los redshifts calculados
    
    for name, rest_wl in lines_dict.items():
        measurement = measure_line_parameters(wavelengths, flux, rest_wl)
        if measurement:
            report['absorption_lines'][name] = measurement
            
            # Calcular redshift para esta línea
            z = calculate_redshift(measurement['observed_center'], rest_wl)
            measurement['redshift'] = z
            redshifts.append(z)
    
    # Calcular redshift robusto usando múltiples líneas
    if redshifts:
        mean_z, std_z, n_lines = robust_redshift_calculation(redshifts, sigma_clip=redshift_sigma_clip)
        
        report['redshift'] = {
            'value': float(mean_z),
            'error': float(std_z),
            'n_lines_used': n_lines,
            'n_lines_total': len(redshifts)
        }
        
        # Calcular velocidad radial
        report['radial_velocity'] = {
            'value': float(mean_z * 299792.458),  # km/s
            'error': float(std_z * 299792.458)
        }
    
    # Buscar líneas de emisión
    report['emission_lines'] = find_emission_lines(wavelengths, flux)
    
    # Calcular metalicidad aproximada
    report['mg_fe_ratio'] = float(calculate_mg_fe_index(wavelengths, flux))
    
    # Estimación de metalicidad basada en ratio Mg/Fe
    if report['mg_fe_ratio'] < 0.9:
        report['metallicity_estimate'] = "Baja metalicidad"
    elif report['mg_fe_ratio'] < 1.1:
        report['metallicity_estimate'] = "Metalicidad solar"
    else:
        report['metallicity_estimate'] = "Alta metalicidad"
    
    # Estimación de temperatura si se midió Hβ
    if 'Hβ' in report['absorption_lines']:
        hbeta_ew = report['absorption_lines']['Hβ']['equivalent_width']
        temp_est, temp_val = estimate_temperature(hbeta_ew)
        report['temperature_estimate'] = temp_est
        report['temperature_value'] = temp_val
    
    return report