# lamost_analyzer/gui/gui.py
"""
Módulo para la interfaz gráfica del Spectrum Analyzer
Versión Final: Terminal Integrada, Menús Modulares, StyleEngine.
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QGroupBox, QProgressBar, QMessageBox, 
                             QAction, QMenuBar, QTreeView, QFileSystemModel, 
                             QSplitter, QHeaderView, QLineEdit,
                             QTableWidget, QTableWidgetItem, QComboBox,
                             QScrollArea, QGridLayout, QDialog, QDialogButtonBox,
                             QColorDialog, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QDir, QSettings
from PyQt5.QtGui import QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np

# Importaciones nuevas para modo universal
from astropy.io import fits
from astropy.table import Table

from lamost_analyzer.core.fits_processor import read_fits_file, valid_mask, rebin_spectrum
from lamost_analyzer.core.utils import try_savgol, running_percentile, enhance_line_detection
from lamost_analyzer.core.spectral_analysis import generate_spectral_report
from lamost_analyzer.config import DEFAULT_PARAMS, SPECTRAL_LINES


# ==============================================================================
# 1. STYLE ENGINE (Gestor de Estilos Centralizado)
# ==============================================================================
class StyleEngine:
    """Motor centralizado de estilos para evitar repetir código CSS"""
    
    @staticmethod
    def _px(size, scale):
        return int(size * scale)

    @staticmethod
    def _pt(size, scale):
        return int(size * scale)

    @staticmethod
    def get_groupbox_style(theme, scale):
        s = scale
        return f"""
            QGroupBox {{
                font-weight: bold;
                border: {StyleEngine._px(2, s)}px solid {theme['border']};
                border-radius: {StyleEngine._px(5, s)}px;
                margin-top: 1ex;
                padding-top: {StyleEngine._px(10, s)}px;
                background-color: {theme['secondary']};
                color: {theme['text_primary']};
                font-size: {StyleEngine._pt(11, s)}pt;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {StyleEngine._px(10, s)}px;
                padding: 0 5px 0 5px;
                color: {theme['accent']};
            }}
        """

    @staticmethod
    def get_combobox_style(theme, scale):
        s = scale
        return f"""
            QComboBox {{
                background-color: {theme['secondary']};
                border: {StyleEngine._px(1, s)}px solid {theme['border']};
                border-radius: {StyleEngine._px(3, s)}px;
                padding: {StyleEngine._px(4, s)}px;
                color: {theme['text_secondary']};
                min-width: {StyleEngine._px(80, s)}px;
                font-size: {StyleEngine._pt(9, s)}pt;
            }}
            QComboBox:focus {{
                border: {StyleEngine._px(1, s)}px solid {theme['accent']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: {StyleEngine._px(20, s)}px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: {StyleEngine._px(5, s)}px solid transparent;
                border-right: {StyleEngine._px(5, s)}px solid transparent;
                border-top: {StyleEngine._px(5, s)}px solid {theme['text_secondary']};
                width: 0px;
                height: 0px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme['secondary']};
                border: {StyleEngine._px(1, s)}px solid {theme['border']};
                color: {theme['text_secondary']};
                selection-background-color: {theme['accent']};
                font-size: {StyleEngine._pt(9, s)}pt;
            }}
        """

    @staticmethod
    def get_label_style(theme, scale):
        return f"color: {theme['text_secondary']}; font-weight: bold; font-size: {StyleEngine._pt(10, scale)}pt;"

    @staticmethod
    def get_table_style(theme, scale):
        s = scale
        return f"""
            QTableWidget {{
                background-color: {theme['primary']};
                border: {StyleEngine._px(1, s)}px solid {theme['border']};
                border-radius: {StyleEngine._px(4, s)}px;
                color: {theme['text_secondary']};
                gridline-color: {theme['border']};
                font-size: {StyleEngine._pt(9, s)}pt;
            }}
            QTableWidget::item {{
                padding: {StyleEngine._px(6, s)}px;
                border-bottom: {StyleEngine._px(1, s)}px solid {theme['border']};
            }}
            QTableWidget::item:selected {{
                background-color: {theme['accent']};
                color: #ffffff;
            }}
            QHeaderView::section {{
                background-color: {theme['secondary']};
                color: {theme['accent']};
                font-weight: bold;
                padding: {StyleEngine._px(6, s)}px;
                border: none;
                border-bottom: {StyleEngine._px(2, s)}px solid {theme['accent']};
                font-size: {StyleEngine._pt(9, s)}pt;
            }}
        """

    @staticmethod
    def get_scrollarea_style(theme, scale):
        s = scale
        return f"""
            QScrollArea {{
                background-color: {theme['secondary']};
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background-color: {theme['secondary']};
            }}
            QScrollBar:vertical {{
                background-color: {theme['secondary']};
                width: {StyleEngine._px(15, s)}px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {theme['accent']};
                border-radius: {StyleEngine._px(7, s)}px;
                min-height: {StyleEngine._px(20, s)}px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {theme['accent_hover']};
            }}
            QScrollBar:horizontal {{
                background-color: {theme['secondary']};
                height: {StyleEngine._px(15, s)}px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {theme['accent']};
                border-radius: {StyleEngine._px(7, s)}px;
                min-width: {StyleEngine._px(20, s)}px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {theme['accent_hover']};
            }}
        """


# ==============================================================================
# 2. LOGGER WIDGET (Terminal)
# ==============================================================================
class LoggerWidget(QTextEdit):
    """Widget estilo Terminal para ver logs y diagnósticos"""
    def __init__(self, theme_manager=None, scale=1.0):
        super().__init__()
        self.theme_manager = theme_manager
        self.scale = scale
        self.setReadOnly(True)
        self.setMinimumHeight(int(100 * scale))
        self.apply_style()
        
        font = self.font()
        font.setFamily("Consolas, 'Courier New', monospace")
        font.setPointSize(int(9 * scale))
        self.setFont(font)

    def apply_style(self):
        theme = self.theme_manager.get_current_theme()
        s = self.scale
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {theme['primary']};
                color: #00ff00;
                border: {int(1*s)}px solid {theme['border']};
                border-radius: {int(4*s)}px;
                padding: {int(5*s)}px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: {int(9*s)}pt;
            }}
            QScrollBar:vertical {{
                background-color: {theme['secondary']};
                width: {int(10*s)}px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {theme['accent']};
                border-radius: {int(3*s)}px;
            }}
        """)

    def write_log(self, message):
        cursor = self.textCursor()
        cursor.movePosition(cursor.End)
        
        if "error" in message.lower() or "exception" in message.lower():
            self.setTextColor(QColor("#ff5555"))
        else:
            theme = self.theme_manager.get_current_theme()
            self.setTextColor(QColor(theme['text_secondary']))
            
        cursor.insertText(message)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def clear_log(self):
        self.clear()


# ==============================================================================
# 3. STREAM TO LOGGER (Magia para print)
# ==============================================================================
class StreamToLogger:
    """Redirige el flujo de salida estándar a nuestro LoggerWidget"""
    def __init__(self, logger_widget):
        self.logger = logger_widget

    def write(self, message):
        if message.strip() != "":
            self.logger.write_log(message)

    def flush(self):
        pass


# ==============================================================================
# 4. MENU MANAGER (Menús Modulares)
# ==============================================================================
class MenuManager:
    """Gestor modular de menús. Permite agregar/quitarse menús editando la configuración."""
    
    DEFAULT_MENU_STRUCTURE = [
        {
            "name": "File",
            "items": [
                {"text": "Open FITS", "shortcut": "Ctrl+O", "method": "open_file"},
                {"text": "Save Results", "shortcut": "Ctrl+S", "method": "save_results"},
                {"separator": True},
                {"text": "Exit", "shortcut": "Ctrl+Q", "method": "close"}
            ]
        },
        {
            "name": "Edit",
            "items": [
                {"text": "Copy Results", "shortcut": "Ctrl+C", "method": "copy_results"},
                {"text": "Clear Results", "shortcut": "Ctrl+L", "method": "clear_results"},
                {"text": "Clear Log", "method": "clear_log"}
            ]
        },
        {
            "name": "View",
            "items": [
                {"text": "Reset Plot", "shortcut": "Ctrl+R", "method": "reset_plot"},
                {"text": "Toggle Toolbar", "shortcut": "Ctrl+T", "method": "toggle_toolbar"},
                {"text": "Fullscreen", "shortcut": "F11", "method": "toggle_fullscreen"},
                {"text": "Toggle Terminal", "shortcut": "F12", "method": "toggle_terminal"}
            ]
        },
        {
            "name": "Tools",
            "items": [
                {"text": "Run Analysis", "shortcut": "F5", "method": "analyze"},
                {"separator": True},
                {"text": "Theme Settings", "shortcut": "Ctrl+,", "method": "show_theme_settings"},
                {"text": "Batch Processing", "method": "batch_processing"}
            ]
        },
        {
            "name": "Help",
            "items": [
                {"text": "Documentation", "shortcut": "F1", "method": "show_documentation"},
                {"separator": True},
                {"text": "About", "method": "show_about"}
            ]
        }
    ]

    def __init__(self, main_window, menu_bar):
        self.main_window = main_window
        self.menu_bar = menu_bar

    def build_menus(self, menu_config=None):
        self.menu_bar.clear()
        if menu_config is None:
            menu_config = self.DEFAULT_MENU_STRUCTURE

        for menu_data in menu_config:
            menu = self.menu_bar.addMenu(menu_data["name"])
            for item in menu_data.get("items", []):
                if item.get("separator"):
                    menu.addSeparator()
                else:
                    action = QAction(item["text"], self.main_window)
                    if "shortcut" in item:
                        action.setShortcut(item["shortcut"])
                    method_name = item["method"]
                    if hasattr(self.main_window, method_name):
                        action.triggered.connect(getattr(self.main_window, method_name))
                    menu.addAction(action)


# ==============================================================================
# 5. CARGADOR UNIVERSAL
# ==============================================================================
def load_spectrum_universal(file_path):
    """
    Carga FITS o TXT de forma universal.
    Devuelve: (wavelength, flux)
    """
    try:
        with fits.open(file_path) as hdul:
            data = hdul[1].data if len(hdul) > 1 else hdul[0].data
            header = hdul[0].header
            wavelength, flux = None, None

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
                return np.array(wavelength), np.array(flux)

    except Exception as e:
        pass

    try:
        data = Table.read(file_path, format='ascii')
        return np.array(data.columns[0]), np.array(data.columns[1])
    except Exception:
        pass

    return None, None


# ==============================================================================
# 6. THEME MANAGER
# ==============================================================================
class ThemeManager:
    """Gestor centralizado de temas y escalado de la aplicación"""
    
    THEMES = {
        "dark": {
            "primary": "#1e1e1e",
            "secondary": "#2d2d30",
            "tertiary": "#3e3e42",
            "text_primary": "#ffffff",
            "text_secondary": "#cccccc",
            "text_muted": "#888888",
            "accent": "#007acc",
            "border": "#444444",
            "success": "#107c10",
            "warning": "#ffb900",
            "error": "#e81123"
        },
        "light": {
            "primary": "#ffffff",
            "secondary": "#f3f3f3",
            "tertiary": "#e1e1e1",
            "text_primary": "#000000",
            "text_secondary": "#333333",
            "text_muted": "#666666",
            "accent": "#0078d4",
            "border": "#d1d1d1",
            "success": "#107c10",
            "warning": "#ffb900",
            "error": "#e81123"
        }
    }
    
    def __init__(self):
        self.settings = QSettings("LAMOST", "SpectrumAnalyzer")
        self.current_theme_name = self.settings.value("theme", "dark")
        self.custom_accent = self.settings.value("accent_color", "#007acc")
        
        screen = QApplication.primaryScreen().availableGeometry()
        height = screen.height()
        
        if height < 1300:
            self.scale = 0.9
        else:
            self.scale = 0.75
        
    def get_current_theme(self):
        theme = self.THEMES[self.current_theme_name].copy()
        theme["accent"] = self.custom_accent
        theme["accent_hover"] = self._lighten_color(theme["accent"], 20)
        theme["accent_pressed"] = self._darken_color(theme["accent"], 20)
        return theme
    
    def set_theme(self, theme_name):
        if theme_name in self.THEMES:
            self.current_theme_name = theme_name
            self.settings.setValue("theme", theme_name)
    
    def set_accent_color(self, color):
        self.custom_accent = color
        self.settings.setValue("accent_color", color)
    
    def reset_to_defaults(self):
        self.set_theme("dark")
        self.set_accent_color("#007acc")
    
    def _lighten_color(self, color, percent):
        return QColor(color).lighter(100 + percent).name()
    
    def _darken_color(self, color, percent):
        return QColor(color).darker(100 + percent).name()


# ==============================================================================
# 7. SETTINGS DIALOG
# ==============================================================================
class SettingsDialog(QDialog):
    """Diálogo de configuración de temas y colores con ESCALADO REDUCIDO"""
    
    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.parent = parent
        self.scale = theme_manager.scale
        base_font = self.font()
        base_font.setPointSize(int(10 * self.scale))
        self.setFont(base_font)
        self.init_ui()
        self.apply_dialog_theme()
        
    def apply_dialog_theme(self):
        theme = self.theme_manager.get_current_theme()
        s = self.scale
        border_width = int(2 * s)
        radius = int(5 * s)
        padding = int(10 * s)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme['primary']};
                color: {theme['text_primary']};
            }}
            QGroupBox {{
                font-weight: bold;
                font-size: {int(11 * s)}pt;
                border: {border_width}px solid {theme['border']};
                border-radius: {radius}px;
                margin-top: 1ex;
                padding-top: {padding}px;
                padding-bottom: {padding}px;
                background-color: {theme['secondary']};
                color: {theme['text_primary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {int(10 * s)}px;
                padding: 0 5px 0 5px;
                color: {theme['accent']};
            }}
            QRadioButton {{
                color: {theme['text_primary']};
                background-color: {theme['secondary']};
                padding: {int(8 * s)}px;
                spacing: {int(12 * s)}px;
                font-size: {int(10 * s)}pt;
            }}
            QRadioButton::indicator {{
                width: {int(18 * s)}px;
                height: {int(18 * s)}px;
                border-radius: {int(9 * s)}px;
                border: {border_width}px solid {theme['border']};
                background-color: {theme['primary']};
            }}
            QRadioButton::indicator:checked {{
                background-color: {theme['accent']};
                border: {border_width}px solid {theme['accent']};
            }}
            QLabel {{
                color: {theme['text_primary']};
                background-color: transparent;
                font-size: {int(10 * s)}pt;
            }}
            QPushButton {{
                background-color: {theme['accent']};
                color: white;
                border: none;
                border-radius: {int(4 * s)}px;
                padding: {int(10 * s)}px {int(18 * s)}px;
                font-weight: bold;
                font-size: {int(10 * s)}pt;
            }}
            QPushButton:hover {{
                background-color: {theme['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: {theme['accent_pressed']};
            }}
        """)
        
    def init_ui(self):
        self.setWindowTitle("Configuración de Tema y Colores")
        w = int(600 * self.scale)
        h = int(600 * self.scale)
        self.setMinimumSize(w, h) 
        self.resize(w, h)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(int(25 * self.scale)) 
        layout.setContentsMargins(int(30 * self.scale), int(30 * self.scale), int(30 * self.scale), int(30 * self.scale))
        
        theme_group = QGroupBox("Selección de Tema")
        theme_layout = QVBoxLayout(theme_group)
        theme_layout.setSpacing(int(10 * self.scale)) 
        
        self.theme_buttons = QButtonGroup(self)
        self.dark_radio = QRadioButton("Modo Oscuro")
        self.light_radio = QRadioButton("Modo Claro")
        self.system_radio = QRadioButton("Modo del Sistema")
        
        self.theme_buttons.addButton(self.dark_radio)
        self.theme_buttons.addButton(self.light_radio)
        self.theme_buttons.addButton(self.system_radio)
        
        current_theme = self.theme_manager.current_theme_name
        if current_theme == "dark":
            self.dark_radio.setChecked(True)
        elif current_theme == "light":
            self.light_radio.setChecked(True)
        else:
            self.system_radio.setChecked(True)
        
        theme_layout.addWidget(self.dark_radio)
        theme_layout.addWidget(self.light_radio)
        theme_layout.addWidget(self.system_radio)
        
        color_group = QGroupBox("Color Secundario/Acento")
        color_layout = QVBoxLayout(color_group)
        color_layout.setSpacing(int(15 * self.scale))
        
        color_preview_layout = QHBoxLayout()
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(int(70 * self.scale), int(45 * self.scale))
        self.update_color_preview()
        self.color_name = QLabel(self.theme_manager.custom_accent)
        self.color_name.setStyleSheet(f"font-weight: bold; font-size: {int(12 * self.scale)}pt;")
        self.btn_choose_color = QPushButton("Elegir Color...")
        self.btn_choose_color.clicked.connect(self.choose_accent_color)
        self.btn_choose_color.setMinimumHeight(int(35 * self.scale))
        
        color_preview_layout.addWidget(self.color_preview)
        color_preview_layout.addWidget(self.color_name)
        color_preview_layout.addWidget(self.btn_choose_color)
        color_preview_layout.addStretch()
        
        predefined_layout = QVBoxLayout()
        predefined_label = QLabel("Colores predefinidos:")
        predefined_label.setStyleSheet(f"font-size: {int(12 * self.scale)}pt; margin-bottom: 5px;")
        predefined_layout.addWidget(predefined_label)
        
        colors_grid = QGridLayout()
        colors_grid.setSpacing(int(15 * self.scale))
        colors = [
            ("#007acc", "Azul", 0, 0),
            ("#107c10", "Verde", 0, 1),
            ("#d83b01", "Naranja", 0, 2),
            ("#e81123", "Rojo", 1, 0),
            ("#b4009e", "Morado", 1, 1),
            ("#008272", "Turquesa", 1, 2)
        ]
        btn_size = int(45 * self.scale)
        for color_code, color_name, row, col in colors:
            btn = QPushButton("")
            btn.setFixedSize(btn_size, btn_size)
            btn.setStyleSheet(f"QPushButton {{ background-color: {color_code}; border: 2px solid {color_code}; border-radius: 20px; }}"
                            f"QPushButton:hover {{ border: 2px solid #ffffff; }}")
            btn.setToolTip(color_name)
            btn.clicked.connect(lambda checked, c=color_code: self.set_predefined_color(c))
            colors_grid.addWidget(btn, row, col)
        
        predefined_layout.addLayout(colors_grid)
        color_layout.addLayout(color_preview_layout)
        color_layout.addLayout(predefined_layout)
        
        preview_group = QGroupBox("Vista Previa")
        preview_layout = QVBoxLayout(preview_group)
        
        preview_widget = QWidget()
        preview_widget.setFixedHeight(int(100 * self.scale))
        preview_widget.setObjectName("previewWidget")
        
        preview_layout_inner = QHBoxLayout(preview_widget)
        preview_button = QPushButton("Botón de Ejemplo")
        preview_button.setObjectName("previewButton")
        preview_button.setMinimumHeight(int(30 * self.scale))
        preview_label = QLabel("Texto de ejemplo")
        preview_label.setObjectName("previewLabel")
        
        preview_layout_inner.addWidget(preview_button)
        preview_layout_inner.addWidget(preview_label)
        preview_layout_inner.addStretch()
        preview_layout.addWidget(preview_widget)
        
        reset_group = QGroupBox("Restablecer Configuración")
        reset_layout = QHBoxLayout(reset_group)
        self.btn_reset = QPushButton("Restablecer a Valores por Defecto")
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        self.btn_reset.setMinimumHeight(int(35 * self.scale))
        reset_layout.addWidget(self.btn_reset)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        button_layout = button_box.layout()
        for i in range(button_layout.count()):
             item = button_layout.itemAt(i)
             if item.widget():
                 btn = item.widget()
                 btn.setMinimumHeight(int(35 * self.scale))

        button_box.setCenterButtons(True) 
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(theme_group)
        layout.addWidget(color_group)
        layout.addWidget(preview_group)
        layout.addWidget(reset_group)
        
        layout.addStretch(1) 
        
        layout.addWidget(button_box)
        self.update_preview()
        
    def update_color_preview(self):
        theme = self.theme_manager.get_current_theme()
        self.color_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {self.theme_manager.custom_accent};
                border: 2px solid {theme['border']};
                border-radius: 4px;
            }}
        """)
        
    def choose_accent_color(self):
        color = QColorDialog.getColor(QColor(self.theme_manager.custom_accent), self)
        if color.isValid():
            self.set_accent_color(color.name())
            
    def set_predefined_color(self, color_code):
        self.set_accent_color(color_code)
        
    def set_accent_color(self, color_code):
        self.theme_manager.set_accent_color(color_code)
        self.color_name.setText(color_code)
        self.update_color_preview()
        self.update_preview()
        
    def update_preview(self):
        theme = self.theme_manager.get_current_theme()
        s = self.scale
        preview_widget = self.findChild(QWidget, "previewWidget")
        if preview_widget:
            preview_widget.setStyleSheet(f"background-color: {theme['secondary']}; border: 1px solid {theme['border']}; border-radius: 4px;")
        
        preview_button = self.findChild(QPushButton, "previewButton")
        if preview_button:
            preview_button.setStyleSheet(f"""
                QPushButton#previewButton {{
                    background-color: {theme['accent']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: {int(8*s)}px {int(16*s)}px;
                    font-weight: bold;
                    font-size: {int(10*s)}pt;
                }}
                QPushButton#previewButton:hover {{
                    background-color: {theme['accent_hover']};
                }}
            """)
        
        preview_label = self.findChild(QLabel, "previewLabel")
        if preview_label:
            preview_label.setStyleSheet(f"color: {theme['text_primary']}; font-weight: bold; font-size: {int(10*s)}pt;")
        
    def get_selected_theme(self):
        if self.dark_radio.isChecked():
            return "dark"
        elif self.light_radio.isChecked():
            return "light"
        else:
            return "dark"
        
    def apply_changes(self):
        selected_theme = self.get_selected_theme()
        self.theme_manager.set_theme(selected_theme)
        if self.parent:
            self.parent.apply_theme()
            
    def reset_to_defaults(self):
        reply = QMessageBox.question(
            self, 
            "Restablecer configuración",
            "¿Está seguro de que desea restablecer la configuración a los valores por defecto?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.theme_manager.reset_to_defaults()
            self.dark_radio.setChecked(True)
            self.color_name.setText("#007acc")
            self.update_color_preview()
            self.update_preview()
            self.apply_changes()
            
    def accept(self):
        self.apply_changes()
        super().accept()


# ==============================================================================
# 8. THEME AWARE NAVIGATION TOOLBAR
# ==============================================================================
class ThemeAwareNavigationToolbar(NavigationToolbar):
    """Toolbar de matplotlib adaptada al tema"""
    def __init__(self, canvas, parent, theme_manager):
        super().__init__(canvas, parent)
        self.theme_manager = theme_manager
        self.scale = theme_manager.scale
        self.update_style()
        
    def update_style(self):
        theme = self.theme_manager.get_current_theme()
        s = self.scale
        self.setStyleSheet(f"""
            QToolButton {{
                background-color: {theme['secondary']};
                border: {int(1*s)}px solid {theme['border']};
                border-radius: {int(3*s)}px;
                color: {theme['text_primary']};
                padding: {int(4*s)}px;
                font-size: {int(9*s)}pt;
            }}
            QToolButton:hover {{
                background-color: {theme['tertiary']};
                border: {int(1*s)}px solid {theme['accent']};
            }}
            QToolButton:pressed {{
                background-color: {theme['accent']};
            }}
        """)


# ==============================================================================
# 9. FILE EXPLORER WIDGET
# ==============================================================================
class FileExplorerWidget(QWidget):
    """Widget del explorador de archivos con tema y escala aplicados"""
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.theme_manager = theme_manager
        self.scale = theme_manager.scale
        self.history = []
        self.history_index = -1
        self.init_ui()
        
    def init_ui(self):
        s = self.scale
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(int(5 * s))
        
        toolbar_layout = QHBoxLayout()
        self.btn_back = self.create_tool_button("◀ Atrás", "Volver al directorio anterior")
        self.btn_forward = self.create_tool_button("Adelante ▶", "Avanzar al siguiente directorio")
        self.btn_up = self.create_tool_button("↑ Subir", "Subir al directorio padre")
        self.btn_home = self.create_tool_button("🏠 Inicio", "Ir al directorio home")
        self.btn_refresh = self.create_tool_button("🔄 Actualizar", "Refrescar vista")
        
        toolbar_layout.addWidget(self.btn_back)
        toolbar_layout.addWidget(self.btn_forward)
        toolbar_layout.addWidget(self.btn_up)
        toolbar_layout.addWidget(self.btn_home)
        toolbar_layout.addWidget(self.btn_refresh)
        toolbar_layout.addStretch()
        
        path_layout = QHBoxLayout()
        path_label = QLabel("Ruta:")
        self.update_label_style(path_label)
        
        self.path_edit = QLineEdit()
        self.update_lineedit_style(self.path_edit)
        self.path_edit.setText(QDir.currentPath())
        self.path_edit.returnPressed.connect(self.on_path_edited)
        
        self.btn_go = QPushButton("Ir")
        self.btn_go.setFixedSize(int(30 * s), int(25 * s))
        self.update_go_button_style(self.btn_go)
        self.btn_go.clicked.connect(self.on_path_edited)
        
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.btn_go)
        
        self.tree_view = QTreeView()
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        self.model.setNameFilters(["*.fits", "*.fit"])
        self.model.setNameFilterDisables(False)
        
        self.tree_view.setModel(self.model)
        self.tree_view.setRootIndex(self.model.index(QDir.currentPath()))
        self.tree_view.setAnimated(False)
        self.tree_view.setIndentation(int(20 * s))
        self.tree_view.setSortingEnabled(True)
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)
        
        self.update_treeview_style()
        
        self.tree_view.doubleClicked.connect(self.on_file_double_clicked)
        self.tree_view.clicked.connect(self.on_tree_selection_changed)
        self.btn_back.clicked.connect(self.go_back)
        self.btn_forward.clicked.connect(self.go_forward)
        self.btn_up.clicked.connect(self.go_up)
        self.btn_home.clicked.connect(self.go_home)
        self.btn_refresh.clicked.connect(self.refresh)
        
        self.add_to_history(QDir.currentPath())
        
        layout.addLayout(toolbar_layout)
        layout.addLayout(path_layout)
        layout.addWidget(self.tree_view)
    
    def create_tool_button(self, text, tooltip):
        s = self.scale
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(int(28 * s))
        self.update_button_style(btn)
        return btn
    
    def update_style(self):
        for btn in [self.btn_back, self.btn_forward, self.btn_up, self.btn_home, self.btn_refresh]:
            self.update_button_style(btn)
        self.update_go_button_style(self.btn_go)
        self.update_lineedit_style(self.path_edit)
        self.update_treeview_style()
        for label in self.findChildren(QLabel):
            self.update_label_style(label)

    def update_button_style(self, button):
        s = self.scale
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme['secondary']};
                    border: {int(1*s)}px solid {theme['border']};
                    border-radius: {int(3*s)}px;
                    color: {theme['text_primary']};
                    font-weight: bold;
                    font-size: {int(11*s)}pt;
                    padding: {int(4*s)}px {int(8*s)}px;
                    min-width: {int(60*s)}px;
                }}
                QPushButton:hover {{
                    background-color: {theme['tertiary']};
                    border: {int(1*s)}px solid {theme['accent']};
                }}
                QPushButton:pressed {{
                    background-color: {theme['accent']};
                }}
                QPushButton:disabled {{
                    background-color: {theme['primary']};
                    color: {theme['text_muted']};
                    border: {int(1*s)}px solid {theme['border']};
                }}
            """)
            
    def update_label_style(self, label):
        s = self.scale
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            label.setStyleSheet(f"color: {theme['text_secondary']}; font-size: {int(10*s)}pt;")
            
    def update_lineedit_style(self, line_edit):
        s = self.scale
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            line_edit.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {theme['primary']};
                    border: {int(1*s)}px solid {theme['border']};
                    border-radius: {int(3*s)}px;
                    padding: {int(4*s)}px {int(8*s)}px;
                    color: {theme['text_secondary']};
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: {int(10*s)}pt;
                    selection-background-color: {theme['accent']};
                }}
                QLineEdit:focus {{
                    border: {int(1*s)}px solid {theme['accent']};
                }}
            """)
            
    def update_go_button_style(self, button):
        s = self.scale
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme['accent']};
                    border: {int(1*s)}px solid {theme['accent']};
                    border-radius: {int(3*s)}px;
                    color: #ffffff;
                    font-weight: bold;
                    font-size: {int(9*s)}pt;
                }}
                QPushButton:hover {{
                    background-color: {theme['accent_hover']};
                }}
                QPushButton:pressed {{
                    background-color: {theme['accent_pressed']};
                }}
            """)
            
    def update_treeview_style(self):
        s = self.scale
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            self.tree_view.setStyleSheet(f"""
                QTreeView {{
                    background-color: {theme['primary']};
                    border: {int(1*s)}px solid {theme['border']};
                    border-radius: {int(4*s)}px;
                    color: {theme['text_secondary']};
                    outline: none;
                    font-size: {int(10*s)}pt;
                }}
                QTreeView::item {{
                    padding: {int(4*s)}px;
                    border: none;
                }}
                QTreeView::item:selected {{
                    background-color: {theme['accent']};
                    color: #ffffff;
                }}
                QTreeView::item:hover {{
                    background-color: {theme['tertiary']};
                }}
            """)
        
    def on_path_edited(self):
        path = self.path_edit.text().strip().replace('\\', '/')
        if os.path.exists(path):
            if os.path.isdir(path):
                self.set_path(path)
            else:
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir):
                    self.set_path(parent_dir)
                    index = self.model.index(path)
                    if index.isValid():
                        self.tree_view.setCurrentIndex(index)
                        self.tree_view.scrollTo(index)
        else:
            QMessageBox.warning(self, "Ruta no válida", f"La ruta especificada no existe:\n{path}")
            current_path = self.tree_view.rootIndex().data(QFileSystemModel.FilePathRole)
            self.path_edit.setText(current_path)
            
    def on_tree_selection_changed(self, index):
        if index.isValid():
            path = self.model.filePath(index)
            self.path_edit.setText(path)
        
    def add_to_history(self, path):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(path)
        self.history_index = len(self.history) - 1
        self.update_navigation_buttons()
        
    def update_navigation_buttons(self):
        self.btn_back.setEnabled(self.history_index > 0)
        self.btn_forward.setEnabled(self.history_index < len(self.history) - 1)
        
    def go_back(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.set_path(self.history[self.history_index])
            self.update_navigation_buttons()
            
    def go_forward(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.set_path(self.history[self.history_index])
            self.update_navigation_buttons()
            
    def go_up(self):
        current_path = self.tree_view.rootIndex().data(QFileSystemModel.FilePathRole)
        parent_dir = QDir(current_path)
        if parent_dir.cdUp():
            self.set_path(parent_dir.absolutePath())
            
    def go_home(self):
        self.set_path(QDir.homePath())
        
    def refresh(self):
        current_path = self.tree_view.rootIndex().data(QFileSystemModel.FilePathRole)
        self.model.setRootPath(current_path)
        self.tree_view.setRootIndex(self.model.index(current_path))
        
    def set_path(self, path):
        self.tree_view.setRootIndex(self.model.index(path))
        self.path_edit.setText(path)
        self.add_to_history(path)
        
    def on_file_double_clicked(self, index):
        path = self.model.filePath(index)
        if os.path.isfile(path) and path.lower().endswith(('.fits', '.fit', '.txt', '.csv')):
            if self.parent:
                self.parent.load_fits_file(path)
        elif os.path.isdir(path):
            self.set_path(path)


# ==============================================================================
# 10. PARAMETERS PANEL
# ==============================================================================
class ParametersPanel(QWidget):
    """Panel de parámetros optimizado con StyleEngine"""
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.theme_manager = theme_manager
        self.scale = theme_manager.scale
        self.current_params = DEFAULT_PARAMS.copy()
        self.init_ui()
        
    def init_ui(self):
        s = self.scale
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(int(10 * s))
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.update_scrollarea_style(scroll_area)
        
        scroll_widget = QWidget()
        self.update_widget_style(scroll_widget)
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(int(10 * s))
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        
        params_group = QGroupBox("Parámetros de Procesamiento")
        self.update_groupbox_style(params_group)
        params_layout = QGridLayout(params_group)
        params_layout.setVerticalSpacing(int(8 * s))
        params_layout.setHorizontalSpacing(int(10 * s))
        
        param_options = {
            "REBIN_FACTOR": ["2", "4", "6", "8", "10"],
            "SG_WINDOW": ["31", "61", "91", "121", "151"],
            "SG_POLY": ["2", "3", "4", "5"],
            "MOVING_AVG_WINDOW": ["15", "25", "35", "45", "55"],
            "DO_CONTINUUM_NORM": ["True", "False"],
            "SNR_WINDOW": ["100", "125", "150", "175", "200"],
            "CONTINUUM_WINDOW": ["501", "601", "701", "801", "901"],
            "CONTINUUM_PERCENTILE": ["85", "88", "90", "92", "95"],
            "REDSHIFT_SIGMA_CLIP": ["1.5", "2.0", "2.5", "3.0", "3.5"]
        }
        
        self.comboboxes = {}
        row = 0
        for key, value in self.current_params.items():
            label = QLabel(key)
            self.update_label_style(label)
            label.setToolTip(f"Parámetro: {key}")
            
            combo = QComboBox()
            combo.setToolTip(f"Seleccione un valor para {key}")
            combo.setMinimumHeight(int(25 * s))
            self.update_combobox_style(combo)
            
            if key in param_options:
                combo.addItems(param_options[key])
            else:
                combo.addItem(str(value))
            
            current_value = str(value)
            index = combo.findText(current_value)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.addItem(current_value)
                combo.setCurrentText(current_value)
            
            combo.currentTextChanged.connect(lambda text, param=key: self.on_parameter_changed(param, text))
            
            params_layout.addWidget(label, row, 0)
            params_layout.addWidget(combo, row, 1)
            self.comboboxes[key] = combo
            row += 1
        
        params_layout.setColumnStretch(1, 1)
        
        lines_group = QGroupBox("Líneas Espectrales de Referencia")
        self.update_groupbox_style(lines_group)
        lines_layout = QVBoxLayout(lines_group)
        
        self.lines_table = QTableWidget()
        self.lines_table.setColumnCount(2)
        self.lines_table.setHorizontalHeaderLabels(["Línea", "Longitud de Onda (Å)"])
        self.lines_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lines_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.lines_table.setMaximumHeight(int(150 * s))
        
        self.update_table_style(self.lines_table)
        self.update_spectral_lines_table()
        
        lines_layout.addWidget(self.lines_table)
        
        scroll_layout.addWidget(params_group)
        scroll_layout.addWidget(lines_group)
        scroll_layout.addStretch(1)
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
    
    def update_style(self):
        theme = self.theme_manager.get_current_theme()
        s = self.scale
        self.update_scrollarea_style(self.findChild(QScrollArea))
        for group in self.findChildren(QGroupBox):
            group.setStyleSheet(StyleEngine.get_groupbox_style(theme, s))
        for label in self.findChildren(QLabel):
            label.setStyleSheet(StyleEngine.get_label_style(theme, s))
        for combo in self.findChildren(QComboBox):
            combo.setStyleSheet(StyleEngine.get_combobox_style(theme, s))
        self.update_table_style(self.lines_table)

    def update_widget_style(self, widget):
        theme = self.theme_manager.get_current_theme()
        widget.setStyleSheet(f"background-color: {theme['secondary']};")

    def update_groupbox_style(self, groupbox):
        theme = self.theme_manager.get_current_theme()
        groupbox.setStyleSheet(StyleEngine.get_groupbox_style(theme, self.scale))

    def update_label_style(self, label):
        theme = self.theme_manager.get_current_theme()
        label.setStyleSheet(StyleEngine.get_label_style(theme, self.scale))

    def update_combobox_style(self, combobox):
        theme = self.theme_manager.get_current_theme()
        combobox.setStyleSheet(StyleEngine.get_combobox_style(theme, self.scale))

    def update_table_style(self, table):
        theme = self.theme_manager.get_current_theme()
        table.setStyleSheet(StyleEngine.get_table_style(theme, self.scale))

    def update_scrollarea_style(self, scroll_area):
        theme = self.theme_manager.get_current_theme()
        scroll_area.setStyleSheet(StyleEngine.get_scrollarea_style(theme, self.scale))
        
    def on_parameter_changed(self, param_name, new_value):
        try:
            original_value = self.current_params[param_name]
            if isinstance(original_value, bool):
                self.current_params[param_name] = (new_value == "True")
            elif isinstance(original_value, int):
                self.current_params[param_name] = int(new_value)
            elif isinstance(original_value, float):
                self.current_params[param_name] = float(new_value)
            else:
                self.current_params[param_name] = new_value
            if self.parent:
                self.parent.current_params = self.current_params.copy()
        except ValueError as e:
            print(f"Error al convertir valor para {param_name}: {e}")

    def update_spectral_lines_table(self):
        lines = SPECTRAL_LINES
        self.lines_table.setRowCount(len(lines))
        for i, (key, value) in enumerate(lines.items()):
            self.lines_table.setItem(i, 0, QTableWidgetItem(key))
            self.lines_table.setItem(i, 1, QTableWidgetItem(str(value)))


# ==============================================================================
# 11. MPL CANVAS
# ==============================================================================
class MplCanvas(FigureCanvas):
    """Widget de matplotlib con tema y escala aplicada"""
    def __init__(self, parent=None, width=8, height=6, dpi=100, theme_manager=None):
        self.theme_manager = theme_manager
        self.scale = theme_manager.scale
        if theme_manager and theme_manager.current_theme_name == "light":
            plt.style.use('default')
        else:
            plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)
        self.update_plot_theme()
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
    def update_plot_theme(self):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            s = self.scale
            self.fig.patch.set_facecolor(theme['primary'])
            self.ax.set_facecolor(theme['secondary'])
            fontsize_labels = int(10 * s)
            fontsize_title = int(12 * s)
            fontsize_ticks = int(8 * s)
            self.ax.tick_params(colors=theme['text_secondary'], labelsize=fontsize_ticks)
            for spine in self.ax.spines.values():
                spine.set_color(theme['border'])
            self.ax.title.set_color(theme['text_primary'])
            self.ax.title.set_fontsize(fontsize_title)
            self.ax.xaxis.label.set_color(theme['text_secondary'])
            self.ax.yaxis.label.set_color(theme['text_secondary'])
            self.ax.xaxis.label.set_fontsize(fontsize_labels)
            self.ax.yaxis.label.set_fontsize(fontsize_labels)
            legend = self.ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(fontsize_ticks)
            self.ax.grid(True, alpha=0.2, color=theme['border'])


# ==============================================================================
# 12. MAIN WINDOW
# ==============================================================================
class MainWindow(QMainWindow):
    """Ventana principal de la aplicación"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAMOST Spectrum Analyzer")
        self.theme_manager = ThemeManager()
        self.file_path = None
        self.wl = None
        self.flux = None
        self.ivar = None
        self.report = None
        self.current_params = DEFAULT_PARAMS.copy()
        self.source_type = "LAMOST"
        self.scale = self.theme_manager.scale

        # --- INICIALIZACIÓN DEL LOGGER ---
        self.logger = LoggerWidget(self.theme_manager, self.scale)
        sys.stdout = StreamToLogger(self.logger)
        sys.stderr = StreamToLogger(self.logger)
        print("Sistema iniciado correctamente.")
        # ------------------------------------------

        base_font = self.font()
        base_font.setPointSize(int(9 * self.scale))
        self.setFont(base_font)
        self.init_ui()
        
        # --- MENU MANAGER ---
        self.menu_manager = MenuManager(self, self.menuBar())
        self.menu_manager.build_menus()
        # -------------------
        
        self.apply_theme()
        
    def apply_theme(self):
        theme = self.theme_manager.get_current_theme()
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(theme['primary']))
        palette.setColor(QPalette.WindowText, QColor(theme['text_primary']))
        palette.setColor(QPalette.Base, QColor(theme['primary']))
        palette.setColor(QPalette.AlternateBase, QColor(theme['secondary']))
        palette.setColor(QPalette.ToolTipBase, QColor(theme['text_primary']))
        palette.setColor(QPalette.ToolTipText, QColor(theme['text_primary']))
        palette.setColor(QPalette.Text, QColor(theme['text_primary']))
        palette.setColor(QPalette.Button, QColor(theme['secondary']))
        palette.setColor(QPalette.ButtonText, QColor(theme['text_primary']))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(theme['accent']))
        palette.setColor(QPalette.Highlight, QColor(theme['accent']))
        palette.setColor(QPalette.HighlightedText, QColor(theme['secondary']))
        self.setPalette(palette)
        self.setStyleSheet(self.get_main_stylesheet())
        if hasattr(self, 'file_explorer'):
            self.file_explorer.update_style()
        if hasattr(self, 'parameters_panel'):
            self.parameters_panel.update_style()
        if hasattr(self, 'toolbar'):
            self.toolbar.update_style()
        if hasattr(self, 'canvas'):
            self.canvas.update_plot_theme()
            self.canvas.draw()
        self.update_file_label_style()
        self.update()
        QApplication.processEvents()
        
    def get_main_stylesheet(self):
        theme = self.theme_manager.get_current_theme()
        s = self.scale
        return f"""
            QMainWindow {{
                background-color: {theme['primary']};
                color: {theme['text_primary']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: {int(2 * s)}px solid {theme['border']};
                border-radius: {int(5 * s)}px;
                margin-top: 1ex;
                padding-top: {int(10 * s)}px;
                background-color: {theme['secondary']};
                color: {theme['text_primary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {int(10 * s)}px;
                padding: 0 5px 0 5px;
                color: {theme['accent']};
            }}
            QPushButton {{
                background-color: {theme['accent']};
                color: white;
                border: none;
                border-radius: {int(4 * s)}px;
                padding: {int(8 * s)}px {int(16 * s)}px;
                font-weight: bold;
                min-height: {int(20 * s)}px;
            }}
            QPushButton:hover {{
                background-color: {theme['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: {theme['accent_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {theme['border']};
                color: {theme['text_muted']};
            }}
            QLabel {{
                color: {theme['text_secondary']};
                padding: {int(4 * s)}px;
            }}
            QProgressBar {{
                border: {int(1 * s)}px solid {theme['border']};
                border-radius: {int(4 * s)}px;
                text-align: center;
                color: {theme['text_primary']};
                background-color: {theme['secondary']};
            }}
            QProgressBar::chunk {{
                background-color: {theme['accent']};
                border-radius: {int(3 * s)}px;
            }}
            QTextEdit {{
                background-color: {theme['primary']};
                color: {theme['text_secondary']};
                border: {int(1 * s)}px solid {theme['border']};
                border-radius: {int(4 * s)}px;
                padding: {int(8 * s)}px;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
            QMenuBar {{
                background-color: {theme['secondary']};
                color: {theme['text_primary']};
                border-bottom: {int(1 * s)}px solid {theme['border']};
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: {int(4 * s)}px {int(8 * s)}px;
            }}
            QMenuBar::item:selected {{
                background-color: {theme['tertiary']};
            }}
            QMenu {{
                background-color: {theme['secondary']};
                color: {theme['text_primary']};
                border: {int(1 * s)}px solid {theme['border']};
            }}
            QMenu::item {{
                padding: {int(4 * s)}px {int(16 * s)}px;
            }}
            QMenu::item:selected {{
                background-color: {theme['accent']};
            }}
        """
        
    def init_ui(self):
        s = self.scale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(int(10 * s))
        main_layout.setContentsMargins(int(10 * s), int(10 * s), int(10 * s), int(10 * s))
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(int(350 * s))
        left_panel.setMaximumWidth(int(450 * s))
        
        explorer_group = QGroupBox("Project Explorer")
        explorer_layout = QVBoxLayout(explorer_group)
        self.file_explorer = FileExplorerWidget(self, self.theme_manager)
        explorer_layout.addWidget(self.file_explorer)
        
        file_info_group = QGroupBox("File Information")
        file_info_layout = QVBoxLayout(file_info_group)
        self.file_label = QLabel("No hay archivo seleccionado")
        self.file_label.setWordWrap(True)
        file_info_layout.addWidget(self.file_label)
        
        src_layout = QHBoxLayout()
        src_label = QLabel("Modo Fuente:")
        self.update_label_style(src_label)
        src_layout.addWidget(src_label)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["LAMOST", "UNIVERSAL (FITS/TXT)"])
        self.source_combo.currentTextChanged.connect(self.set_source_type)
        self.update_combobox_style(self.source_combo)
        src_layout.addWidget(self.source_combo)
        file_info_layout.addLayout(src_layout)
        
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
        
        progress_label = QLabel("Progreso:")
        analysis_layout.addWidget(progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        analysis_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(explorer_group)
        left_layout.addWidget(file_info_group)
        left_layout.addWidget(analysis_group)
        left_layout.addStretch()
        
        # Panel central
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(int(5 * s))
        
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100, theme_manager=self.theme_manager)
        self.toolbar = ThemeAwareNavigationToolbar(self.canvas, self, self.theme_manager)
        center_layout.addWidget(self.toolbar)
        center_layout.addWidget(self.canvas)
        
        results_label = QLabel("Resultados del análisis:")
        results_label.setStyleSheet("font-weight: bold;")
        center_layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(int(200 * s))
        center_layout.addWidget(self.results_text)
        
        # --- AÑADIR TERMINAL AQUÍ ---
        self.logger.setMaximumHeight(int(150 * s))
        self.logger.apply_style()
        center_layout.addWidget(self.logger)
        # -------------------------
        
        right_panel = QWidget()
        right_panel.setMinimumWidth(int(300 * s))
        right_panel.setMaximumWidth(int(500 * s))
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.parameters_panel = ParametersPanel(self, self.theme_manager)
        right_layout.addWidget(self.parameters_panel)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([int(400*self.scale), int(800*self.scale), int(400*self.scale)])
        
        main_layout.addWidget(splitter)

    def set_source_type(self, text):
        self.source_type = text.split(" ")[0]
        
    def update_file_label_style(self):
        theme = self.theme_manager.get_current_theme()
        s = self.scale
        if self.file_path:
            self.file_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {theme['primary']};
                    border: {int(1*s)}px solid {theme['accent']};
                    border-radius: {int(4*s)}px;
                    padding: {int(8*s)}px;
                    color: {theme['text_primary']};
                    min-height: {int(60*s)}px;
                }}
            """)
        else:
            self.file_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {theme['primary']};
                    border: {int(1*s)}px solid {theme['border']};
                    border-radius: {int(4*s)}px;
                    padding: {int(8*s)}px;
                    color: {theme['text_muted']};
                    min-height: {int(60*s)}px;
                }}
            """)
            
    def update_label_style(self, label):
        s = self.scale
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            label.setStyleSheet(f"color: {theme['text_secondary']}; font-size: {int(10*s)}pt;")

    def update_combobox_style(self, combobox):
        s = self.scale
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            combobox.setStyleSheet(f"""
                QComboBox {{
                    background-color: {theme['secondary']};
                    border: {int(1*s)}px solid {theme['border']};
                    border-radius: {int(3*s)}px;
                    padding: {int(4*s)}px;
                    color: {theme['text_secondary']};
                    min-width: {int(80*s)}px;
                    font-size: {int(9*s)}pt;
                }}
                QComboBox:focus {{
                    border: {int(1*s)}px solid {theme['accent']};
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: {int(20*s)}px;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    border-left: {int(5*s)}px solid transparent;
                    border-right: {int(5*s)}px solid transparent;
                    border-top: {int(5*s)}px solid {theme['text_secondary']};
                    width: 0px;
                    height: 0px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {theme['secondary']};
                    border: {int(1*s)}px solid {theme['border']};
                    color: {theme['text_secondary']};
                    selection-background-color: {theme['accent']};
                    font-size: {int(9*s)}pt;
                }}
            """)
        
    def load_fits_file(self, file_path):
        self.file_path = file_path
        filename = os.path.basename(file_path)
        self.file_label.setText(f"📁 {filename}\n📍 {file_path}")
        self.update_file_label_style()
        self.btn_analyze.setEnabled(True)
        self.results_text.append(f"✓ Archivo cargado: {filename}")
        
    def open_file(self):
        if self.source_type == "UNIVERSAL":
            file_filter = "Todos los archivos (*.fits *.fit *.txt *.csv)"
        else:
            file_filter = "FITS Files (*.fits *.fit)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo", "", file_filter
        )
        if file_path:
            self.load_fits_file(file_path)
            
    def analyze(self):
        if not self.file_path:
            QMessageBox.warning(self, "Advertencia", "Seleccione un archivo primero.")
            return
            
        try:
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            params = self.current_params
            lines_dict = SPECTRAL_LINES
            
            # --- LÓGICA MODIFICADA PARA SOPORTAR UNIVERSAL ---
            if self.source_type == "LAMOST":
                # CÓDIGO ORIGINAL LAMOST
                self.wl, self.flux, self.ivar = read_fits_file(self.file_path)
                self.progress_bar.setValue(30)
                
                m = valid_mask(self.flux, self.ivar)
                self.wl, self.flux, self.ivar = self.wl[m], self.flux[m], self.ivar[m]

                wl_r, flux_r, ivar_r = rebin_spectrum(self.wl, self.flux, self.ivar, factor=params["REBIN_FACTOR"])
                if len(flux_r) == 0:
                    QMessageBox.critical(self, "Error", "Array vacío tras rebinado.")
                    return
            
            else:
                # NUEVO CÓDIGO UNIVERSAL
                print("Usando cargador universal en GUI...")
                wl, flux = load_spectrum_universal(self.file_path)
                
                if wl is None:
                    QMessageBox.critical(self, "Error", "No se pudo leer el archivo en modo Universal.")
                    return
                
                self.wl, self.flux = wl, flux
                # Simulamos ivar (inverso de varianza) para que los filtros posteriores no rompan
                self.ivar = np.ones_like(flux) * 100.0 
                
                # Para datos universales, asumimos que ya vienen bien calibrados y saltamos el rebinado
                wl_r, flux_r, ivar_r = self.wl, self.flux, self.ivar
            # ----------------------------------------------

            current_sg_window = params["SG_WINDOW"]
            if params["SG_WINDOW"] > len(flux_r):
                current_sg_window = max(3, len(flux_r)-1)
                self.results_text.append(f"⚠ SG_WINDOW ajustado a {current_sg_window}")

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
            QMessageBox.critical(self, "Error", f"Error durante el análisis: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def display_results(self):
        if not self.report:
            return
            
        self.results_text.clear()
        self.results_text.append("=== REPORTE DE ANÁLISIS ESPECTRAL ===")
        self.results_text.append(f"📊 Rango λ: {self.report['wavelength_range']['min']:.1f} - {self.report['wavelength_range']['max']:.1f} Å")
        self.results_text.append(f"📈 SNR: {self.report['snr']:.1f}")
        
        if 'redshift' in self.report:
            z_info = self.report['redshift']
            rv_info = self.report['radial_velocity']
            self.results_text.append(f"🔭 Redshift: {z_info['value']:.6f} ± {z_info['error']:.6f}")
            self.results_text.append(f"🚀 Vel. radial: {rv_info['value']:.1f} ± {rv_info['error']:.1f} km/s")
        
    def plot_spectrum(self, wavelengths, flux_original, flux_processed, lines_dict):
        self.canvas.ax.clear()
        theme = self.theme_manager.get_current_theme()
        
        self.canvas.ax.plot(wavelengths, flux_original, color=theme['text_muted'], alpha=0.6, linewidth=0.5, label="Original")
        self.canvas.ax.plot(wavelengths, flux_processed, color=theme['accent'], linewidth=1, label="Procesado")
        
        for name, wl_line in lines_dict.items():
            if wavelengths.min() <= wl_line <= wavelengths.max():
                self.canvas.ax.axvline(wl_line, color=theme['warning'], linestyle='--', alpha=0.7)
                self.canvas.ax.text(wl_line, max(flux_original)*0.9, name, rotation=90, color=theme['text_secondary'], fontsize=8)
        
        self.canvas.ax.legend(facecolor=theme['secondary'], edgecolor=theme['border'], labelcolor=theme['text_secondary'])
        self.canvas.ax.set_title("Espectro completo", color=theme['text_primary'])
        self.canvas.ax.set_xlabel("Longitud de onda (Å)", color=theme['text_secondary'])
        self.canvas.ax.set_ylabel("Flujo", color=theme['text_secondary'])
        self.canvas.ax.grid(True, alpha=0.2, color=theme['border'])

        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
    def save_results(self):
        if not self.report:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para guardar.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar resultados", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("=== REPORTE DE ANÁLISIS ESPECTRAL LAMOST ===\n\n")
                    f.write(f"Archivo analizado: {os.path.basename(self.file_path)}\n")
                    f.write(f"Modo Fuente: {self.source_type}\n")
                    f.write(f"Rango λ: {self.report['wavelength_range']['min']:.1f} - {self.report['wavelength_range']['max']:.1f} Å\n")
                    f.write(f"SNR: {self.report['snr']:.1f}\n")
                    if 'redshift' in self.report:
                        z_info = self.report['redshift']
                        rv_info = self.report['radial_velocity']
                        f.write(f"Redshift: {z_info['value']:.6f} ± {z_info['error']:.6f}\n")
                        f.write(f"Vel. radial: {rv_info['value']:.1f} ± {rv_info['error']:.1f} km/s\n")
                QMessageBox.information(self, "Éxito", f"Resultados guardados en: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudieron guardar los resultados: {str(e)}")

    def reset_plot(self):
        self.canvas.ax.clear()
        self.canvas.update_plot_theme()
        self.canvas.draw()

    def copy_results(self):
        if self.results_text.toPlainText():
            clipboard = QApplication.clipboard()
            clipboard.setText(self.results_text.toPlainText())
            QMessageBox.information(self, "Copiado", "Resultados copiados al portapapeles")
        else:
            QMessageBox.warning(self, "Advertencia", "No hay resultados para copiar")

    def clear_results(self):
        self.results_text.clear()

    def toggle_toolbar(self):
        self.toolbar.setVisible(not self.toolbar.isVisible())

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def toggle_terminal(self):
        if self.logger.isVisible():
            self.logger.hide()
        else:
            self.logger.show()
            
    def clear_log(self):
        self.logger.clear()

    def show_theme_settings(self):
        dialog = SettingsDialog(self.theme_manager, self)
        dialog.exec_()

    def batch_processing(self):
        QMessageBox.information(self, "Procesamiento por lotes", "Funcionalidad de procesamiento por lotes - En desarrollo")

    def show_documentation(self):
        about_text = """
        <h3>LAMOST Spectrum Analyzer - Documentación</h3>
        <p><b>Funcionalidades:</b></p>
        <ul>
            <li>Carga de archivos FITS</li>
            <li><b>MODO UNIVERSAL:</b> Soporte para archivos FITS genéricos y TXT de aficionados.</li>
            <li>Análisis espectral automático</li>
            <li>Detección de líneas espectrales</li>
            <li>Cálculo de redshift y velocidad radial</li>
            <li>Visualización interactiva</li>
            <li>Temas personalizables (oscuro, claro, sistema)</li>
            <li>Colores secundarios configurables</li>
            <li><b>Escalado automático:</b> Interfaz y gráficos se ajustan a tu pantalla.</li>
        </ul>
        <p><b>Atajos de teclado:</b></p>
        <ul>
            <li>Ctrl+O: Abrir archivo</li>
            <li>Ctrl+S: Guardar resultados</li>
            <li>F5: Ejecutar análisis</li>
            <li>Ctrl+R: Reiniciar gráficos</li>
            <li>Ctrl+,: Configuración de temas</li>
            <li>F12: Mostrar/Ocultar Terminal</li>
            <li>F1: Documentación</li>
        </ul>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Documentación")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.exec_()

    def show_about(self):
        theme = self.theme_manager.get_current_theme()
        about_text = f"""
        <h3>LAMOST Spectrum Analyzer</h3>
        <p>Versión Universal con temas personalizables</p>
        <p>Herramienta para análisis espectral de datos FITS y archivos universales</p>
        <p>Desarrollado con PyQt5 y matplotlib</p>
        <p><b>Tema actual:</b> {self.theme_manager.current_theme_name.title()}</p>
        <p><b>Color acento:</b> <span style="color: {theme['accent']};">{theme['accent']}</span></p>
        <hr>
        <p style="color: {theme['text_muted']};">© 2024 LAMOST Analysis Team & Community</p>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Acerca de")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.exec_()


def run_gui():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()