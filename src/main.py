#!/usr/bin/env python3
"""
Punto de entrada principal - LAMOST Spectrum Analyzer
"""

import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Añade la carpeta src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Ahora los imports deberían funcionar
from lamost_analyzer.core.fits_processor import read_fits_file, valid_mask, rebin_spectrum
from lamost_analyzer.core.utils import try_savgol, running_percentile, enhance_line_detection
from lamost_analyzer.core.spectral_analysis import generate_spectral_report
from lamost_analyzer.config import DEFAULT_PARAMS, SPECTRAL_LINES

def plot_spectrum_with_analysis(wavelengths, flux_original, flux_processed, lines_dict, report):
    """Crea una visualización completa del espectro con análisis"""
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(wavelengths, flux_original, linewidth=0.5, alpha=0.6, label="Original", color='lightgray')
    plt.plot(wavelengths, flux_processed, linewidth=1.0, label="Procesado", color='blue')
    
    y_max = np.nanmax(flux_processed) * 1.1
    for name, wavelength in lines_dict.items():
        if wavelengths.min() <= wavelength <= wavelengths.max():
            plt.axvline(wavelength, color="red", linestyle="--", alpha=0.7)
            measurement = report['absorption_lines'].get(name)
            if measurement:
                plt.text(wavelength+2, y_max*0.9, 
                        f"{name}\nEW={measurement['equivalent_width']:.2f}Å", 
                        rotation=90, color="red", fontsize=7)
    
    plt.xlabel("Longitud de onda (Å)")
    plt.ylabel("Flujo")
    
    title = f"Espectro LAMOST - SNR: {report['snr']:.1f}"
    if 'redshift' in report:
        title += f" - z: {report['redshift']['value']:.6f} ± {report['redshift']['error']:.6f}"
    plt.title(title)
    
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    zoom_region = (5100, 5200)
    zoom_mask = (wavelengths >= zoom_region[0]) & (wavelengths <= zoom_region[1])
    plt.plot(wavelengths[zoom_mask], flux_processed[zoom_mask], linewidth=1.5, color='blue')
    
    for name, wavelength in lines_dict.items():
        if zoom_region[0] <= wavelength <= zoom_region[1]:
            plt.axvline(wavelength, color="red", linestyle="--", alpha=0.7)
            plt.text(wavelength+1, np.max(flux_processed[zoom_mask])*0.9, 
                    name, rotation=90, color="red", fontsize=8)
    
    plt.xlabel("Longitud de onda (Å)")
    plt.ylabel("Flujo")
    plt.title(f"Zoom región {zoom_region[0]}-{zoom_region[1]} Å")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Usar parámetros por defecto
    params = DEFAULT_PARAMS
    lines_dict = SPECTRAL_LINES
    
    # Pedir al usuario el archivo FITS
    fits_file = input("Por favor, introduce la ruta del archivo FITS: ").strip().strip('"')
    
    try:
        # 1) Leer FITS
        wl, flux, ivar = read_fits_file(fits_file)
        
        # Añadir esto después de leer el archivo FITS
        print("\n=== DIAGNÓSTICO DEL ARCHIVO FITS ===")
        print(f"Tamaño del array de longitud de onda: {len(wl)}")
        print(f"Tamaño del array de flujo: {len(flux)}")
        print(f"Tamaño del array de ivar: {len(ivar)}")
    
        # Verificar valores válidos
        print(f"Valores finitos en flujo: {np.sum(np.isfinite(flux))}/{len(flux)}")
        print(f"Valores finitos en ivar: {np.sum(np.isfinite(ivar))}/{len(ivar)}")
        print(f"Valores ivar > 0: {np.sum(ivar > 0)}/{len(ivar)}")
    
        # Estadísticas básicas
        print(f"Flujo - Min: {np.nanmin(flux):.3f}, Max: {np.nanmax(flux):.3f}, Mediana: {np.nanmedian(flux):.3f}")
        print(f"IVAR - Min: {np.nanmin(ivar):.3f}, Max: {np.nanmax(ivar):.3f}, Mediana: {np.nanmedian(ivar):.3f}")
        
        # 2) Máscara básica usando IVAR
        m = valid_mask(flux, ivar)
        wl, flux, ivar = wl[m], flux[m], ivar[m]

        # 3) Rebin para mejorar SNR
        wl_r, flux_r, ivar_r = rebin_spectrum(wl, flux, ivar, factor=params["REBIN_FACTOR"])

        # Validar que el array no esté vacío después del rebinado
        if len(flux_r) == 0:
            print("Error: El array está vacío después del rebinado.")
            return

        # Ajustar SG_WINDOW si es necesario
        current_sg_window = params["SG_WINDOW"]
        if params["SG_WINDOW"] > len(flux_r):
            current_sg_window = len(flux_r) - 1 if len(flux_r) % 2 == 0 else len(flux_r)
            current_sg_window = max(3, current_sg_window)
            print(f"Advertencia: SG_WINDOW ajustado a {current_sg_window} por tamaño de datos")

        # 4) Suavizado
        flux_smooth = try_savgol(flux_r, window=current_sg_window, poly=params["SG_POLY"], moving_avg_window=params["MOVING_AVG_WINDOW"])

        # 4.5) Realce de líneas para espectros con SNR bajo
        flux_enhanced = enhance_line_detection(flux_smooth, enhancement_factor=1.3)

        # 5) (opcional) normalización de continuo
        if params["DO_CONTINUUM_NORM"]:
            cont = running_percentile(flux_enhanced, win=params["CONTINUUM_WINDOW"], q=params["CONTINUUM_PERCENTILE"])
            cont = np.where(cont <= 0, np.nanmedian(cont[cont>0]), cont)
            flux_plot = flux_enhanced / cont
            ylab = "Flujo (normalizado)"
        else:
            flux_plot = flux_enhanced
            ylab = "Flujo"

        # 6) Generar reporte de análisis
        report = generate_spectral_report(wl_r, flux_plot, ivar_r, lines_dict, redshift_sigma_clip=params["REDSHIFT_SIGMA_CLIP"])
        
        # 7) Mostrar resultados importantes
        print("=== REPORTE DE ANÁLISIS ESPECTRAL ===")
        print(f"Rango de longitud de onda: {report['wavelength_range']['min']:.1f} - {report['wavelength_range']['max']:.1f} Å")
        print(f"SNR estimado: {report['snr']:.1f}")
        
        if 'redshift' in report:
            z_info = report['redshift']
            rv_info = report['radial_velocity']
            print(f"Redshift: {z_info['value']:.6f} ± {z_info['error']:.6f}")
            print(f"Velocidad radial: {rv_info['value']:.1f} ± {rv_info['error']:.1f} km/s")
            print(f"Líneas utilizadas: {z_info['n_lines_used']}/{z_info['n_lines_total']}")
        
        if 'temperature_estimate' in report:
            print(f"Temperatura estimada: {report['temperature_estimate']}")
        
        print(f"Ratio Mg/Fe: {report['mg_fe_ratio']:.3f}")
        print(f"Metalicidad estimada: {report['metallicity_estimate']}")
        
        print("\n=== LÍNEAS DE ABSORCIÓN DETECTADAS ===")
        for name, params in report['absorption_lines'].items():
            print(f"{name}: EW={params['equivalent_width']:.3f}Å, FWHM={params['fwhm']:.3f}Å, z={params.get('redshift', 'N/A'):.6f}")
        
        print("\n=== LÍNEAS DE EMISIÓN DETECTADAS ===")
        for i, line in enumerate(report['emission_lines']):
            print(f"Línea {i+1}: {line['wavelength']:.2f}Å, Fuerza: {line['strength']:.3f}")

        # 8) Guardar resultados
        with open('spectral_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        spectrum_df = pd.DataFrame({
            'wavelength': wl_r,
            'flux_original': flux_r,
            'flux_processed': flux_plot,
            'ivar': ivar_r
        })
        spectrum_df.to_csv('processed_spectrum.csv', index=False)
        
        # 9) Crear y guardar gráficos
        fig = plot_spectrum_with_analysis(wl_r, flux_r, flux_plot, lines_dict, report)
        plt.savefig('detailed_spectral_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nAnálisis completado. Resultados guardados en:")
        print("- spectral_analysis_report.json")
        print("- processed_spectrum.csv")
        print("- detailed_spectral_analysis.png")

    except Exception as e:
        print(f"Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()