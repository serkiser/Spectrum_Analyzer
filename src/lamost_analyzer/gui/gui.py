# lamost_analyzer/gui/gui.py
"""
Módulo para la interfaz gráfica del Spectrum Analyzer
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QGroupBox, QProgressBar, QMessageBox, QAction, QMenuBar)
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
        
        # Variables de datos
        self.file_path = None
        self.wl = None
        self.flux = None
        self.ivar = None
        self.report = None
        
        self.init_ui()
        self.create_menu()
        
    def create_menu(self):
        """Crea la barra de menús superior"""
        menubar = self.menuBar()

        # Menú Archivo
        file_menu = menubar.addMenu("Archivo")
        open_action = QAction("Abrir FITS", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("Guardar resultados", self)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)

        exit_action = QAction("Salir", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Menú Ver
        view_menu = menubar.addMenu("Ver")
        reset_plot_action = QAction("Reiniciar gráficos", self)
        reset_plot_action.triggered.connect(self.reset_plot)
        view_menu.addAction(reset_plot_action)

        # Menú Ayuda
        help_menu = menubar.addMenu("Ayuda")
        about_action = QAction("Acerca de", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Panel izquierdo
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        
        file_group = QGroupBox("Archivo")
        file_layout = QVBoxLayout(file_group)
        
        self.btn_open = QPushButton("Abrir archivo FITS")
        self.btn_open.clicked.connect(self.open_file)
        file_layout.addWidget(self.btn_open)
        
        self.file_label = QLabel("No hay archivo seleccionado")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
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
        
        self.progress_bar = QProgressBar()
        analysis_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(file_group)
        left_layout.addWidget(analysis_group)
        left_layout.addStretch()
        
        # Panel derecho
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.canvas = MplCanvas(self, width=8, height=8, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)
        
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
            QMessageBox.warning(self, "Advertencia", "Seleccione un archivo FITS primero.")
            return
        try:
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            self.wl, self.flux, self.ivar = read_fits_file(self.file_path)
            self.progress_bar.setValue(30)
            
            params = DEFAULT_PARAMS
            lines_dict = SPECTRAL_LINES
            
            m = valid_mask(self.flux, self.ivar)
            self.wl, self.flux, self.ivar = self.wl[m], self.flux[m], self.ivar[m]

            wl_r, flux_r, ivar_r = rebin_spectrum(self.wl, self.flux, self.ivar, factor=params["REBIN_FACTOR"])
            if len(flux_r) == 0:
                QMessageBox.critical(self, "Error", "Array vacío tras rebinado.")
                return

            current_sg_window = params["SG_WINDOW"]
            if params["SG_WINDOW"] > len(flux_r):
                current_sg_window = max(3, len(flux_r)-1)
                self.results_text.append(f"SG_WINDOW ajustado a {current_sg_window}")

            flux_smooth = try_savgol(flux_r, window=current_sg_window, poly=params["SG_POLY"], moving_avg_window=params["MOVING_AVG_WINDOW"])
            flux_enhanced = enhance_line_detection(flux_smooth, enhancement_factor=1.3)

            if params["DO_CONTINUUM_NORM"]:
                cont = running_percentile(flux_enhanced, win=params["CONTINUUM_WINDOW"], q=params["CONTINUUM_PERCENTILE"])
                cont = np.where(cont <= 0, np.nanmedian(cont[cont>0]), cont)
                flux_plot = flux_enhanced / cont
            else:
                flux_plot = flux_enhanced

            self.report = generate_spectral_report(wl_r, flux_plot, ivar_r, lines_dict, redshift_sigma_clip=params["REDSHIFT_SIGMA_CLIP"])
            self.display_results()
            self.plot_spectrum(wl_r, flux_r, flux_plot, lines_dict)

            self.progress_bar.setValue(100)
            self.btn_save.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def display_results(self):
        if not self.report:
            return
        self.results_text.clear()
        self.results_text.append("=== REPORTE DE ANÁLISIS ESPECTRAL ===")
        self.results_text.append(f"Rango λ: {self.report['wavelength_range']['min']:.1f} - {self.report['wavelength_range']['max']:.1f} Å")
        self.results_text.append(f"SNR: {self.report['snr']:.1f}")
        if 'redshift' in self.report:
            z_info = self.report['redshift']
            rv_info = self.report['radial_velocity']
            self.results_text.append(f"Redshift: {z_info['value']:.6f} ± {z_info['error']:.6f}")
            self.results_text.append(f"Vel. radial: {rv_info['value']:.1f} ± {rv_info['error']:.1f} km/s")
        
    def plot_spectrum(self, wavelengths, flux_original, flux_processed, lines_dict):
        for ax in self.canvas.axes:
            ax.clear()
        ax1 = self.canvas.axes[0]
        ax1.plot(wavelengths, flux_original, color='lightgray', alpha=0.6, linewidth=0.5, label="Original")
        ax1.plot(wavelengths, flux_processed, color='blue', linewidth=1, label="Procesado")
        ax1.legend()
        ax1.set_title("Espectro completo")
        ax1.grid(alpha=0.3)

        ax2 = self.canvas.axes[1]
        zoom_region = (5100, 5200)
        mask = (wavelengths >= zoom_region[0]) & (wavelengths <= zoom_region[1])
        if np.any(mask):
            ax2.plot(wavelengths[mask], flux_processed[mask], color='blue')
        ax2.set_title("Zoom 5100–5200 Å")
        ax2.grid(alpha=0.3)

        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def save_results(self):
        if not self.report:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para guardar.")
            return
        QMessageBox.information(self, "Guardar", "Funcionalidad en desarrollo.")

    def reset_plot(self):
        for ax in self.canvas.axes:
            ax.clear()
        self.canvas.draw()

    def show_about(self):
        QMessageBox.information(self, "Acerca de", "LAMOST Spectrum Analyzer\nVersión inicial con GUI en PyQt5.")


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()
