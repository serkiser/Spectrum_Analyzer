#!/usr/bin/env python3
"""
Punto de entrada principal - LAMOST Spectrum Analyzer
"""

import sys
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Añade la carpeta src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
    
    finite_flux = flux_processed[np.isfinite(flux_processed)]
    y_max = np.max(finite_flux) * 1.1 if len(finite_flux) > 0 else 1.0
    
    for name, wavelength in lines_dict.items():
        if wavelengths.min() <= wavelength <= wavelengths.max():
            plt.axvline(wavelength, color="red", linestyle="--", alpha=0.7)
            measurement = report.get('absorption_lines', {}).get(name)
            if measurement and 'equivalent_width' in measurement:
                plt.text(wavelength+2, y_max*0.9, 
                        f"{name}\nEW={measurement['equivalent_width']:.2f}Å", 
                        rotation=90, color="red", fontsize=7)
    
    plt.xlabel("Longitud de onda (Å)")
    plt.ylabel("Flujo")
    
    title = "Espectro LAMOST"
    if 'snr' in report:
        title += f" - SNR: {report['snr']:.1f}"
    if 'redshift' in report and 'value' in report['redshift']:
        title += f" - z: {report['redshift']['value']:.6f} ± {report['redshift']['error']:.6f}"
    
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    zoom_region = (5100, 5200)
    zoom_mask = (wavelengths >= zoom_region[0]) & (wavelengths <= zoom_region[1])
    
    if np.any(zoom_mask):
        zoom_flux = flux_processed[zoom_mask]
        finite_zoom_flux = zoom_flux[np.isfinite(zoom_flux)]
        
        if len(finite_zoom_flux) > 0:
            plt.plot(wavelengths[zoom_mask], zoom_flux, linewidth=1.5, color='blue')
            y_zoom_max = np.max(finite_zoom_flux) * 0.9
            
            for name, wavelength in lines_dict.items():
                if zoom_region[0] <= wavelength <= zoom_region[1]:
                    plt.axvline(wavelength, color="red", linestyle="--", alpha=0.7)
                    plt.text(wavelength+1, y_zoom_max, name, rotation=90, color="red", fontsize=8)
        else:
            plt.text(0.5, 0.5, "No hay datos válidos en la región de zoom", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, "La región de zoom está fuera del rango de datos", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
    
    plt.xlabel("Longitud de onda (Å)")
    plt.ylabel("Flujo")
    plt.title(f"Zoom región {zoom_region[0]}-{zoom_region[1]} Å")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_file(fits_file):
    """Función que realiza el análisis completo de un archivo FITS"""
    params = DEFAULT_PARAMS
    lines_dict = SPECTRAL_LINES
    
    try:
        wl, flux, ivar = read_fits_file(fits_file)
        if wl is None:
            print("❌ No se pudo leer el archivo FITS. Verifica el archivo.")
            return False

        m = valid_mask(flux, ivar)
        wl, flux, ivar = wl[m], flux[m], ivar[m]

        wl_r, flux_r, ivar_r = rebin_spectrum(wl, flux, ivar, factor=params["REBIN_FACTOR"])
        if len(flux_r) == 0 or not np.any(np.isfinite(flux_r)):
            print("❌ No hay datos válidos después del rebinado.")
            return False

        current_sg_window = params["SG_WINDOW"]
        if params["SG_WINDOW"] > len(flux_r):
            current_sg_window = len(flux_r) - 1 if len(flux_r) % 2 == 0 else len(flux_r)
            current_sg_window = max(3, current_sg_window)
            print(f"SG_WINDOW ajustado a {current_sg_window}")

        flux_smooth = try_savgol(flux_r, window=current_sg_window, poly=params["SG_POLY"], moving_avg_window=params["MOVING_AVG_WINDOW"])
        flux_enhanced = enhance_line_detection(flux_smooth, enhancement_factor=1.3)

        if params["DO_CONTINUUM_NORM"]:
            cont = running_percentile(flux_enhanced, win=params["CONTINUUM_WINDOW"], q=params["CONTINUUM_PERCENTILE"])
            cont = np.where(cont <= 0, np.nanmedian(cont[cont>0]), cont)
            flux_plot = flux_enhanced / cont
        else:
            flux_plot = flux_enhanced

        report = generate_spectral_report(wl_r, flux_plot, ivar_r, lines_dict, redshift_sigma_clip=params["REDSHIFT_SIGMA_CLIP"])

        with open('spectral_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        spectrum_df = pd.DataFrame({
            'wavelength': wl_r,
            'flux_original': flux_r,
            'flux_processed': flux_plot,
            'ivar': ivar_r
        })
        spectrum_df.to_csv('processed_spectrum.csv', index=False)
        
        fig = plot_spectrum_with_analysis(wl_r, flux_r, flux_plot, lines_dict, report)
        plt.savefig('detailed_spectral_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Análisis completado. Archivos guardados:")
        print("- spectral_analysis_report.json")
        print("- processed_spectrum.csv")
        print("- detailed_spectral_analysis.png")
        return True

    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='LAMOST Spectrum Analyzer')
    parser.add_argument('--gui', action='store_true', help='Ejecutar interfaz gráfica')
    parser.add_argument('--file', type=str, help='Archivo FITS para analizar (modo CLI)')
    
    args = parser.parse_args()
    
    if args.gui:
        try:
            from lamost_analyzer.gui.gui import run_gui
            run_gui()
        except ImportError:
            print("⚠️ GUI no disponible. Instala PyQt5 con: pip install pyqt5")
    else:
        fits_file = args.file or input("Introduce la ruta del archivo FITS: ").strip().strip('"')
        analyze_file(fits_file)


if __name__ == "__main__":
    main()
