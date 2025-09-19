"""
Módulo para la interfaz gráfica del Spectrum Analyzer
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QGroupBox, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np

# Importaciones de tu proyecto
from lamost_analyzer.core.fits_processor import read_fits_file, valid_mask, rebin_spectrum
from lamost_analyzer.core.utils import try_savgol, running_percentile, enhance_line_detection
from lamost_analyzer.core.spectral_analysis import generate_spectral_report
from lamost_analyzer.config import DEFAULT_PARAMS, SPECTRAL_LINES

class MplCanvas(FigureCanvas):
    """Widget de matplotlib para integrar en PyQt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(2, 1, figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAMOST Spectrum Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Variables para almacenar datos
        self.file_path = None
        self.wl = None
        self.flux = None
        self.ivar = None
        self.report = None
        
        self.init_ui()
        
    def init_ui(self):
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Panel izquierdo (controles)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        
        # Grupo de controles de archivo
        file_group = QGroupBox("Archivo")
        file_layout = QVBoxLayout(file_group)
        
        self.btn_open = QPushButton("Abrir archivo FITS")
        self.btn_open.clicked.connect(self.open_file)
        file_layout.addWidget(self.btn_open)
        
        self.file_label = QLabel("No hay archivo seleccionado")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        # Grupo de análisis
        analysis_group = QGroupBox("Análisis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.btn_analyze = QPushButton("Ejecutar análisis")
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_analyze.setEnabled(False)
        analysis_layout.addWidget(self.btn_analyze)
        
        self.btn_save = QPushButton("Guardar resultados")
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)
        analysis_layout.addWidget(self.btn_save)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        analysis_layout.addWidget(self.progress_bar)
        
        # Añadir grupos al panel izquierdo
        left_layout.addWidget(file_group)
        left_layout.addWidget(analysis_group)
        left_layout.addStretch()
        
        # Panel derecho (gráficos y resultados)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Canvas de matplotlib
        self.canvas = MplCanvas(self, width=8, height=8, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        # Área de texto para resultados
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)
        
        # Configurar layout principal
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo FITS", "", "FITS Files (*.fits)"
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.btn_analyze.setEnabled(True)
            self.results_text.append(f"Archivo cargado: {file_path}")
            
    def analyze(self):
        if not self.file_path:
            QMessageBox.warning(self, "Advertencia", "Por favor, seleccione un archivo FITS primero.")
            return
            
        try:
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            # Leer archivo FITS
            self.wl, self.flux, self.ivar = read_fits_file(self.file_path)
            
            self.progress_bar.setValue(30)
            self.results_text.append("Archivo FITS leído correctamente.")
            QApplication.processEvents()
            
            # Parámetros
            params = DEFAULT_PARAMS
            lines_dict = SPECTRAL_LINES
            
            # 2) Máscara básica usando IVAR
            m = valid_mask(self.flux, self.ivar)
            self.wl, self.flux, self.ivar = self.wl[m], self.flux[m], self.ivar[m]

            self.progress_bar.setValue(40)
            QApplication.processEvents()
            
            # 3) Rebin para mejorar SNR
            wl_r, flux_r, ivar_r = rebin_spectrum(self.wl, self.flux, self.ivar, factor=params["REBIN_FACTOR"])

            # Validar que el array no esté vacío después del rebinado
            if len(flux_r) == 0:
                QMessageBox.critical(self, "Error", "El array está vacío después del rebinado.")
                return

            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            # Ajustar SG_WINDOW si es necesario
            current_sg_window = params["SG_WINDOW"]
            if params["SG_WINDOW"] > len(flux_r):
                current_sg_window = len(flux_r) - 1 if len(flux_r) % 2 == 0 else len(flux_r)
                current_sg_window = max(3, current_sg_window)
                self.results_text.append(f"Advertencia: SG_WINDOW ajustado a {current_sg_window} por tamaño de datos")

            # 4) Suavizado
            flux_smooth = try_savgol(flux_r, window=current_sg_window, poly=params["SG_POLY"], moving_avg_window=params["MOVING_AVG_WINDOW"])

            self.progress_bar.setValue(60)
            QApplication.processEvents()
            
            # 4.5) Realce de líneas para espectros con SNR bajo
            flux_enhanced = enhance_line_detection(flux_smooth, enhancement_factor=1.3)

            # 5) (opcional) normalización de continuo
            if params["DO_CONTINUUM_NORM"]:
                cont = running_percentile(flux_enhanced, win=params["CONTINUUM_WINDOW"], q=params["CONTINUUM_PERCENTILE"])
                cont = np.where(cont <= 0, np.nanmedian(cont[cont>0]), cont)
                flux_plot = flux_enhanced / cont
            else:
                flux_plot = flux_enhanced

            self.progress_bar.setValue(70)
            QApplication.processEvents()
            
            # 6) Generar reporte de análisis
            self.report = generate_spectral_report(wl_r, flux_plot, ivar_r, lines_dict, redshift_sigma_clip=params["REDSHIFT_SIGMA_CLIP"])
            
            self.progress_bar.setValue(90)
            QApplication.processEvents()
            
            # Mostrar resultados
            self.display_results()
            
            # Graficar
            self.plot_spectrum(wl_r, flux_r, flux_plot, lines_dict)
            
            self.progress_bar.setValue(100)
            self.btn_save.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error durante el análisis: {str(e)}")
            import traceback
            self.results_text.append(f"Error: {traceback.format_exc()}")
            
    def display_results(self):
        """Muestra los resultados del análisis en el área de texto"""
        if not self.report:
            return
            
        self.results_text.clear()
        self.results_text.append("=== REPORTE DE ANÁLISIS ESPECTRAL ===")
        self.results_text.append(f"Rango de longitud de onda: {self.report['wavelength_range']['min']:.1f} - {self.report['wavelength_range']['max']:.1f} Å")
        self.results_text.append(f"SNR estimado: {self.report['snr']:.1f}")
        
        if 'redshift' in self.report:
            z_info = self.report['redshift']
            rv_info = self.report['radial_velocity']
            self.results_text.append(f"Redshift: {z_info['value']:.6f} ± {z_info['error']:.6f}")
            self.results_text.append(f"Velocidad radial: {rv_info['value']:.1f} ± {rv_info['error']:.1f} km/s")
            self.results_text.append(f"Líneas utilizadas: {z_info['n_lines_used']}/{z_info['n_lines_total']}")
        
        if 'temperature_estimate' in self.report:
            self.results_text.append(f"Temperatura estimada: {self.report['temperature_estimate']}")
        
        self.results_text.append(f"Ratio Mg/Fe: {self.report['mg_fe_ratio']:.3f}")
        self.results_text.append(f"Metalicidad estimada: {self.report['metallicity_estimate']}")
        
        self.results_text.append("\n=== LÍNEAS DE ABSORCIÓN DETECTADAS ===")
        for name, params in self.report['absorption_lines'].items():
            self.results_text.append(f"{name}: EW={params['equivalent_width']:.3f}Å, FWHM={params['fwhm']:.3f}Å, z={params.get('redshift', 'N/A'):.6f}")
        
        self.results_text.append("\n=== LÍNEAS DE EMISIÓN DETECTADAS ===")
        for i, line in enumerate(self.report['emission_lines']):
            self.results_text.append(f"Línea {i+1}: {line['wavelength']:.2f}Å, Fuerza: {line['strength']:.3f}")
            
    def plot_spectrum(self, wavelengths, flux_original, flux_processed, lines_dict):
        """Grafica el espectro con las líneas detectadas"""
        # Limpiar los ejes
        for ax in self.canvas.axes:
            ax.clear()
        
        # Espectro completo
        ax1 = self.canvas.axes[0]
        ax1.plot(wavelengths, flux_original, linewidth=0.5, alpha=0.6, label="Original", color='lightgray')
        ax1.plot(wavelengths, flux_processed, linewidth=1.0, label="Procesado", color='blue')
        
        y_max = np.nanmax(flux_processed) * 1.1
        for name, wavelength in lines_dict.items():
            if wavelengths.min() <= wavelength <= wavelengths.max():
                ax1.axvline(wavelength, color="red", linestyle="--", alpha=0.7)
                measurement = self.report['absorption_lines'].get(name)
                if measurement:
                    ax1.text(wavelength+2, y_max*0.9, 
                            f"{name}\nEW={measurement['equivalent_width']:.2f}Å", 
                            rotation=90, color="red", fontsize=7)
        
        ax1.set_xlabel("Longitud de onda (Å)")
        ax1.set_ylabel("Flujo")
        
        title = f"Espectro LAMOST - SNR: {self.report['snr']:.1f}"
        if 'redshift' in self.report:
            title += f" - z: {self.report['redshift']['value']:.6f} ± {self.report['redshift']['error']:.6f}"
        ax1.set_title(title)
        
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Zoom en una región específica
        ax2 = self.canvas.axes[1]
        zoom_region = (5100, 5200)
        zoom_mask = (wavelengths >= zoom_region[0]) & (wavelengths <= zoom_region[1])
        
        if np.any(zoom_mask):
            zoom_flux = flux_processed[zoom_mask]
            finite_zoom_flux = zoom_flux[np.isfinite(zoom_flux)]
            
            if len(finite_zoom_flux) > 0:
                ax2.plot(wavelengths[zoom_mask], zoom_flux, linewidth=1.5, color='blue')
                y_zoom_max = np.max(finite_zoom_flux) * 0.9
                
                # Dibujar líneas en la región de zoom
                for name, wavelength in lines_dict.items():
                    if zoom_region[0] <= wavelength <= zoom_region[1]:
                        ax2.axvline(wavelength, color="red", linestyle="--", alpha=0.7)
                        ax2.text(wavelength+1, y_zoom_max, name, rotation=90, color="red", fontsize=8)
            else:
                ax2.text(0.5, 0.5, "No hay datos válidos en la región de zoom", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, "La región de zoom está fuera del rango de datos", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
        
        ax2.set_xlabel("Longitud de onda (Å)")
        ax2.set_ylabel("Flujo")
        ax2.set_title(f"Zoom región {zoom_region[0]}-{zoom_region[1]} Å")
        ax2.grid(alpha=0.3)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def save_results(self):
        """Guarda los resultados del análisis"""
        if not self.report:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para guardar.")
            return
            
        # Aquí puedes implementar la lógica para guardar los resultados
        # en formato JSON, CSV, o generar un reporte PDF
        QMessageBox.information(self, "Guardar", "Funcionalidad de guardado en desarrollo.")
        

def run_gui():
    """Función para ejecutar la interfaz gráfica"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()