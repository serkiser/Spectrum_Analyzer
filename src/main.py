#!/usr/bin/env python3
"""
Punto de entrada principal - LAMOST Spectrum Analyzer
Actualizado para soporte Universal (LAMOST / FITS Genéricos / TXT)

Comportamiento por defecto:
- Si se ejecuta sin argumentos -> Abre la Interfaz Gráfica (GUI).
- Si se pasa un archivo (--file) -> Análisis por línea de comandos (CLI).
"""

import sys
import os
import argparse
import json
import traceback

import pandas as pd
import numpy as np

# --- Asegurar que 'src' esté en sys.path ---
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, "src")
if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Backend de Matplotlib
import matplotlib
try:
    import PyQt5
    if not matplotlib.get_backend().lower().startswith('qt'):
        matplotlib.use('Qt5Agg')
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# Importaciones del paquete original
try:
    from lamost_analyzer.core.fits_processor import read_fits_file, valid_mask, rebin_spectrum
    from lamost_analyzer.core.utils import try_savgol, running_percentile, enhance_line_detection
    from lamost_analyzer.core.spectral_analysis import generate_spectral_report
    from lamost_analyzer.config import DEFAULT_PARAMS, SPECTRAL_LINES
except Exception as e:
    print("ERROR: No se pudieron importar módulos de lamost_analyzer:", e)
    traceback.print_exc()
    read_fits_file = valid_mask = rebin_spectrum = None
    try_savgol = running_percentile = enhance_line_detection = None
    generate_spectral_report = None
    DEFAULT_PARAMS = {}
    SPECTRAL_LINES = {}

# ==============================================================================
# CARGADOR UNIVERSAL (PARA ARCHIVOS NO LAMOST)
# ==============================================================================
from astropy.io import fits
from astropy.table import Table

def load_spectrum_universal(file_path):
    """
    Carga FITS o TXT de forma universal.
    Devuelve: (wavelength, flux)
    """
    try:
        # --- LECTURA FITS ---
        with fits.open(file_path) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data
            header = hdul[0].header
            
            wavelength, flux = None, None

            # Caso A: Tabla FITS
            if isinstance(data, fits.fitsrec.FITS_rec):
                colnames = data.columns.names
                flux_col = next((c for c in ['flux', 'FLUX', 'Flux', 'intensity', 'Intensity'] if c in colnames), None)
                wave_col = next((c for c in ['wavelength', 'WAVELENGTH', 'lambda', 'loglam'] if c in colnames), None)
                
                if flux_col and wave_col:
                    flux = data[flux_col]
                    wavelength = data[wave_col]
                else:
                    if len(colnames) >= 2:
                        wavelength = np.array(data[colnames[0]])
                        flux = np.array(data[colnames[1]])

            # Caso B: Imagen FITS 1D
            elif isinstance(data, np.ndarray):
                flux = data.flatten()
                crval = header.get('CRVAL1')
                cdelt = header.get('CDELT1')
                crpix = header.get('CRPIX1', 1)
                
                if crval is not None and cdelt is not None:
                    n_pixels = len(flux)
                    wavelength = crval + (np.arange(n_pixels) - crpix + 1) * cdelt
                else:
                    wavelength = np.arange(len(flux))
            
            if wavelength is not None:
                # Convertir a arrays de numpy estándar por seguridad
                return np.array(wavelength), np.array(flux)

    except Exception as e:
        pass # Silencioso para logs limpios

    # --- LECTURA TXT / CSV ---
    try:
        data = Table.read(file_path, format='ascii')
        return np.array(data.columns[0]), np.array(data.columns[1])
    except Exception:
        pass

    return None, None

# ==============================================================================
# FIN DEL CARGADOR UNIVERSAL
# ==============================================================================


def plot_spectrum_with_analysis(wavelengths, flux_original, flux_processed, lines_dict, report):
    """
    Visualización interactiva (Modo CLI)
    """
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(14, 9), sharey=False)
    fig.canvas.manager.set_window_title("LAMOST Spectrum Analyzer - Visualizador")

    ax_full.plot(wavelengths, flux_original, linewidth=0.5, alpha=0.6, label="Original", color='lightgray')
    ax_full.plot(wavelengths, flux_processed, linewidth=1.0, label="Procesado", color='blue')

    finite_flux = flux_processed[np.isfinite(flux_processed)]
    y_max = np.max(finite_flux) * 1.1 if finite_flux.size > 0 else 1.0

    for name, wavelength in lines_dict.items():
        if wavelengths.min() <= wavelength <= wavelengths.max():
            ax_full.axvline(wavelength, color="red", linestyle="--", alpha=0.6)
            measurement = report.get('absorption_lines', {}).get(name, None)
            if measurement and 'equivalent_width' in measurement:
                ax_full.text(wavelength + 2, y_max * 0.9, f"{name}\nEW={measurement['equivalent_width']:.2f}Å",
                             rotation=90, color="red", fontsize=7)

    ax_full.set_xlabel("Longitud de onda (Å)")
    ax_full.set_ylabel("Flujo")
    title = "Espectro Analizado"
    if 'snr' in report:
        title += f" - SNR: {report['snr']:.1f}"
    if 'redshift' in report and isinstance(report['redshift'], dict) and 'value' in report['redshift']:
        title += f" - z: {report['redshift']['value']:.6f}"
    ax_full.set_title(title)
    ax_full.legend()
    ax_full.grid(alpha=0.3)

    ax_zoom.set_title("Selecciona una región arriba para hacer zoom aquí")
    ax_zoom.set_xlabel("Longitud de onda (Å)")
    ax_zoom.set_ylabel("Flujo")
    ax_zoom.grid(alpha=0.3)

    selected_span = {'xmin': None, 'xmax': None}

    def onselect(xmin, xmax):
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        selected_span['xmin'], selected_span['xmax'] = float(xmin), float(xmax)

        mask = (wavelengths >= xmin) & (wavelengths <= xmax)
        if not np.any(mask):
            ax_zoom.clear()
            ax_zoom.text(0.5, 0.5, "Sin datos en la selección", ha='center', va='center', transform=ax_zoom.transAxes)
            fig.canvas.draw_idle()
            return

        ax_zoom.clear()
        ax_zoom.plot(wavelengths[mask], flux_processed[mask], linewidth=1.2, color='blue')
        y_min = np.nanmin(flux_processed[mask])
        y_max_loc = np.nanmax(flux_processed[mask])
        if not np.isfinite(y_min) or not np.isfinite(y_max_loc):
            ax_zoom.set_ylim(-1, 1)
        else:
            ax_zoom.set_ylim(y_min * 0.95, y_max_loc * 1.05)

        for name, wl_line in lines_dict.items():
            if xmin <= wl_line <= xmax:
                ax_zoom.axvline(wl_line, color="red", linestyle="--", alpha=0.7)
                ax_zoom.text(wl_line + (xmax - xmin) * 0.01, y_max_loc * 0.9, name, rotation=90, color="red", fontsize=8)

        ax_zoom.set_xlim(xmin, xmax)
        ax_zoom.set_title(f"Zoom región {xmin:.1f} - {xmax:.1f} Å")
        ax_zoom.set_xlabel("Longitud de onda (Å)")
        ax_zoom.set_ylabel("Flujo")
        ax_zoom.grid(alpha=0.3)
        fig.canvas.draw_idle()

    span = SpanSelector(ax_full, onselect, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.25, facecolor='tab:blue'))

    plt.tight_layout()
    return fig, selected_span


def analyze_file(fits_file, source_type="LAMOST"):
    """
    Función modificada para aceptar 'source_type'.
    - Si es LAMOST: usa tu lógica original.
    - Si es UNIVERSAL: usa el nuevo cargador.
    """
    params = DEFAULT_PARAMS
    lines_dict = SPECTRAL_LINES

    try:
        # --- SELECCIÓN DE FUENTE DE DATOS ---
        if source_type == "UNIVERSAL":
            print(f"\nModo UNIVERSAL activado para: {fits_file}")
            wl, flux = load_spectrum_universal(fits_file)
            if wl is None:
                print("❌ Error: No se pudo leer el archivo en modo Universal.")
                return False
            
            # Simulamos ivar
            ivar = np.ones_like(flux) * 100.0 
            # NOTA: Para archivos universales, nos saltamos la máscara y el rebinado inicial de LAMOST
            wl_r, flux_r, ivar_r = wl, flux, ivar
            
        else:
            # --- CÓDIGO ORIGINAL LAMOST ---
            print(f"\nModo LAMOST activado para: {fits_file}")
            wl, flux, ivar = read_fits_file(fits_file)
            if wl is None:
                print("❌ No se pudo leer el archivo FITS. Verifica el archivo.")
                return False

            m = valid_mask(flux, ivar)
            wl, flux, ivar = wl[m], flux[m], ivar[m]

            wl_r, flux_r, ivar_r = rebin_spectrum(wl, flux, ivar, factor=params.get("REBIN_FACTOR", 1))
            if len(flux_r) == 0 or not np.any(np.isfinite(flux_r)):
                print("❌ No hay datos válidos después del rebinado.")
                return False

        # --- PROCESAMIENTO (Común para ambos) ---
        current_sg_window = params.get("SG_WINDOW", 101)
        if current_sg_window > len(flux_r):
            current_sg_window = len(flux_r) - 1 if len(flux_r) % 2 == 0 else len(flux_r)
            current_sg_window = max(3, current_sg_window)
            print(f"SG_WINDOW ajustado a {current_sg_window}")

        flux_smooth = try_savgol(flux_r, window=current_sg_window, poly=params.get("SG_POLY", 3),
                                 moving_avg_window=params.get("MOVING_AVG_WINDOW", 25))
        flux_enhanced = enhance_line_detection(flux_smooth, enhancement_factor=1.3)

        if params.get("DO_CONTINUUM_NORM", False):
            cont = running_percentile(flux_enhanced, win=params.get("CONTINUUM_WINDOW", 501),
                                      q=params.get("CONTINUUM_PERCENTILE", 95))
            cont = np.where(cont <= 0, np.nanmedian(cont[cont>0]), cont)
            flux_plot = flux_enhanced / cont
        else:
            flux_plot = flux_enhanced

        report = generate_spectral_report(wl_r, flux_plot, ivar_r, lines_dict,
                                          redshift_sigma_clip=params.get("REDSHIFT_SIGMA_CLIP", 3.0))

        fig, selected_span = plot_spectrum_with_analysis(wl_r, flux_r, flux_plot, lines_dict, report)
        plt.show()

        xmin = selected_span.get('xmin')
        xmax = selected_span.get('xmax')
        if xmin is not None and xmax is not None:
            mask = (wl_r >= xmin) & (wl_r <= xmax)
            if np.any(mask):
                fig_zoom, axz = plt.subplots(1, 1, figsize=(10, 4))
                axz.plot(wl_r[mask], flux_plot[mask], linewidth=1.2, color='blue')
                for name, wl_line in lines_dict.items():
                    if xmin <= wl_line <= xmax:
                        axz.axvline(wl_line, color='red', linestyle='--', alpha=0.7)
                        axz.text(wl_line + (xmax-xmin)*0.01, np.nanmax(flux_plot[mask])*0.9, name,
                                 rotation=90, color='red', fontsize=8)
                axz.set_xlim(xmin, xmax)
                axz.set_xlabel("Longitud de onda (Å)")
                axz.set_ylabel("Flujo")
                axz.set_title(f"Zoom región {xmin:.1f} - {xmax:.1f} Å")
                axz.grid(alpha=0.3)
                fig_zoom.tight_layout()
                fig_zoom.savefig('detailed_spectral_analysis_zoom.png', dpi=300, bbox_inches='tight')
                plt.close(fig_zoom)
                print("Guardado: detailed_spectral_analysis_zoom.png")
        else:
            fig.savefig('detailed_spectral_analysis.png', dpi=300, bbox_inches='tight')
            print("Guardado: detailed_spectral_analysis.png")

        with open('spectral_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        pd.DataFrame({
            'wavelength': wl_r,
            'flux_original': flux_r,
            'flux_processed': flux_plot,
            'ivar': ivar_r
        }).to_csv('processed_spectrum.csv', index=False)

        print("\n✅ Análisis completado.")
        return True

    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='LAMOST Spectrum Analyzer - Universal Edition')
    parser.add_argument('--file', type=str, help='Archivo FITS/TXT para analizar (modo CLI)')
    # Nota: --gui ya no es necesario, es el modo por defecto si no se da archivo
    parser.add_argument('--source', type=str, default='LAMOST', 
                        choices=['LAMOST', 'UNIVERSAL'], 
                        help='Tipo de fuente de datos (default: LAMOST)')
    
    args = parser.parse_args()

    # LÓGICA ACTUALIZADA:
    # Si se pasa un archivo explícitamente, modo consola.
    # Si NO, lanzamos la GUI directamente.
    if args.file:
        analyze_file(args.file, source_type=args.source)
    else:
        # Intentar lanzar la GUI
        try:
            from lamost_analyzer.gui.gui import run_gui
            print("Iniciando Interfaz Gráfica...")
            run_gui()
        except Exception as e:
            print("⚠️ No se pudo iniciar la GUI:", e)
            traceback.print_exc()


if __name__ == "__main__":
    main()