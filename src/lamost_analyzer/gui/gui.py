# lamost_analyzer/gui/gui.py
"""
Módulo para la interfaz gráfica del Spectrum Analyzer
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

from lamost_analyzer.core.fits_processor import read_fits_file, valid_mask, rebin_spectrum
from lamost_analyzer.core.utils import try_savgol, running_percentile, enhance_line_detection
from lamost_analyzer.core.spectral_analysis import generate_spectral_report
from lamost_analyzer.config import DEFAULT_PARAMS, SPECTRAL_LINES


class ThemeManager:
    """Gestor centralizado de temas para la aplicación"""
    
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


class SettingsDialog(QDialog):
    """Diálogo de configuración de temas y colores"""
    
    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.parent = parent
        self.init_ui()
        self.apply_dialog_theme()
        
    def apply_dialog_theme(self):
        theme = self.theme_manager.get_current_theme()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme['primary']};
                color: {theme['text_primary']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {theme['border']};
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: {theme['secondary']};
                color: {theme['text_primary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {theme['accent']};
            }}
            QRadioButton {{
                color: {theme['text_primary']};
                background-color: {theme['secondary']};
                padding: 5px;
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid {theme['border']};
                background-color: {theme['primary']};
            }}
            QRadioButton::indicator:checked {{
                background-color: {theme['accent']};
                border: 2px solid {theme['accent']};
            }}
            QLabel {{
                color: {theme['text_primary']};
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {theme['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
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
        self.setFixedSize(500, 450)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Selección de tema
        theme_group = QGroupBox("Selección de Tema")
        theme_layout = QVBoxLayout(theme_group)
        
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
        
        # Color acento
        color_group = QGroupBox("Color Secundario/Acento")
        color_layout = QVBoxLayout(color_group)
        
        color_preview_layout = QHBoxLayout()
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(50, 30)
        self.update_color_preview()
        
        self.color_name = QLabel(self.theme_manager.custom_accent)
        self.color_name.setStyleSheet("font-weight: bold;")
        
        self.btn_choose_color = QPushButton("Elegir Color...")
        self.btn_choose_color.clicked.connect(self.choose_accent_color)
        
        color_preview_layout.addWidget(self.color_preview)
        color_preview_layout.addWidget(self.color_name)
        color_preview_layout.addWidget(self.btn_choose_color)
        color_preview_layout.addStretch()
        
        # Colores predefinidos
        predefined_layout = QVBoxLayout()
        predefined_label = QLabel("Colores predefinidos:")
        predefined_layout.addWidget(predefined_label)
        
        colors_grid = QGridLayout()
        colors = [
            ("#007acc", "Azul", 0, 0),
            ("#107c10", "Verde", 0, 1),
            ("#d83b01", "Naranja", 0, 2),
            ("#e81123", "Rojo", 1, 0),
            ("#b4009e", "Morado", 1, 1),
            ("#008272", "Turquesa", 1, 2)
        ]
        
        for color_code, color_name, row, col in colors:
            btn = QPushButton("")
            btn.setFixedSize(35, 35)
            btn.setStyleSheet(f"QPushButton {{ background-color: {color_code}; border: 2px solid {color_code}; border-radius: 17px; }}"
                            f"QPushButton:hover {{ border: 2px solid #ffffff; }}")
            btn.setToolTip(color_name)
            btn.clicked.connect(lambda checked, c=color_code: self.set_predefined_color(c))
            colors_grid.addWidget(btn, row, col)
        
        predefined_layout.addLayout(colors_grid)
        color_layout.addLayout(color_preview_layout)
        color_layout.addLayout(predefined_layout)
        
        # Vista previa
        preview_group = QGroupBox("Vista Previa")
        preview_layout = QVBoxLayout(preview_group)
        
        preview_widget = QWidget()
        preview_widget.setFixedHeight(80)
        preview_widget.setObjectName("previewWidget")
        
        preview_layout_inner = QHBoxLayout(preview_widget)
        preview_button = QPushButton("Botón de Ejemplo")
        preview_button.setObjectName("previewButton")
        preview_label = QLabel("Texto de ejemplo")
        preview_label.setObjectName("previewLabel")
        
        preview_layout_inner.addWidget(preview_button)
        preview_layout_inner.addWidget(preview_label)
        preview_layout_inner.addStretch()
        preview_layout.addWidget(preview_widget)
        
        # Reset
        reset_group = QGroupBox("Restablecer Configuración")
        reset_layout = QHBoxLayout(reset_group)
        self.btn_reset = QPushButton("Restablecer a Valores por Defecto")
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        reset_layout.addWidget(self.btn_reset)
        
        # Botones
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(theme_group)
        layout.addWidget(color_group)
        layout.addWidget(preview_group)
        layout.addWidget(reset_group)
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
                    padding: 8px 16px;
                    font-weight: bold;
                }}
                QPushButton#previewButton:hover {{
                    background-color: {theme['accent_hover']};
                }}
            """)
        
        preview_label = self.findChild(QLabel, "previewLabel")
        if preview_label:
            preview_label.setStyleSheet(f"color: {theme['text_primary']}; font-weight: bold;")
        
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


class ThemeAwareNavigationToolbar(NavigationToolbar):
    """Toolbar de matplotlib adaptada al tema"""
    def __init__(self, canvas, parent, theme_manager):
        super().__init__(canvas, parent)
        self.theme_manager = theme_manager
        self.update_style()
        
    def update_style(self):
        theme = self.theme_manager.get_current_theme()
        self.setStyleSheet(f"""
            QToolButton {{
                background-color: {theme['secondary']};
                border: 1px solid {theme['border']};
                border-radius: 3px;
                color: {theme['text_primary']};
                padding: 4px;
            }}
            QToolButton:hover {{
                background-color: {theme['tertiary']};
                border: 1px solid {theme['accent']};
            }}
            QToolButton:pressed {{
                background-color: {theme['accent']};
            }}
        """)


class FileExplorerWidget(QWidget):
    """Widget del explorador de archivos con tema aplicado"""
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.theme_manager = theme_manager
        self.history = []
        self.history_index = -1
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Barra de herramientas
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
        
        # Ruta actual
        path_layout = QHBoxLayout()
        path_label = QLabel("Ruta:")
        self.update_label_style(path_label)
        
        self.path_edit = QLineEdit()
        self.update_lineedit_style(self.path_edit)
        self.path_edit.setText(QDir.currentPath())
        self.path_edit.returnPressed.connect(self.on_path_edited)
        
        self.btn_go = QPushButton("Ir")
        self.btn_go.setFixedSize(30, 25)
        self.update_go_button_style(self.btn_go)
        self.btn_go.clicked.connect(self.on_path_edited)
        
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.btn_go)
        
        # TreeView
        self.tree_view = QTreeView()
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())
        self.model.setNameFilters(["*.fits", "*.fit"])
        self.model.setNameFilterDisables(False)
        
        self.tree_view.setModel(self.model)
        self.tree_view.setRootIndex(self.model.index(QDir.currentPath()))
        self.tree_view.setAnimated(False)
        self.tree_view.setIndentation(20)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.hideColumn(1)
        self.tree_view.hideColumn(2)
        self.tree_view.hideColumn(3)
        
        self.update_treeview_style()
        
        # Conexiones
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
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(28)
        self.update_button_style(btn)
        return btn
    
    def update_style(self):
        self.update_button_style(self.btn_back)
        self.update_button_style(self.btn_forward)
        self.update_button_style(self.btn_up)
        self.update_button_style(self.btn_home)
        self.update_button_style(self.btn_refresh)
        self.update_go_button_style(self.btn_go)
        self.update_lineedit_style(self.path_edit)
        self.update_treeview_style()
        
    def update_button_style(self, button):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme['secondary']};
                    border: 1px solid {theme['border']};
                    border-radius: 3px;
                    color: {theme['text_primary']};
                    font-weight: bold;
                    font-size: 11px;
                    padding: 4px 8px;
                    min-width: 60px;
                }}
                QPushButton:hover {{
                    background-color: {theme['tertiary']};
                    border: 1px solid {theme['accent']};
                }}
                QPushButton:pressed {{
                    background-color: {theme['accent']};
                }}
                QPushButton:disabled {{
                    background-color: {theme['primary']};
                    color: {theme['text_muted']};
                    border: 1px solid {theme['border']};
                }}
            """)
            
    def update_label_style(self, label):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            label.setStyleSheet(f"color: {theme['text_secondary']};")
            
    def update_lineedit_style(self, line_edit):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            line_edit.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {theme['primary']};
                    border: 1px solid {theme['border']};
                    border-radius: 3px;
                    padding: 4px 8px;
                    color: {theme['text_secondary']};
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 10px;
                    selection-background-color: {theme['accent']};
                }}
                QLineEdit:focus {{
                    border: 1px solid {theme['accent']};
                }}
            """)
            
    def update_go_button_style(self, button):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme['accent']};
                    border: 1px solid {theme['accent']};
                    border-radius: 3px;
                    color: #ffffff;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {theme['accent_hover']};
                }}
                QPushButton:pressed {{
                    background-color: {theme['accent_pressed']};
                }}
            """)
            
    def update_treeview_style(self):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            self.tree_view.setStyleSheet(f"""
                QTreeView {{
                    background-color: {theme['primary']};
                    border: 1px solid {theme['border']};
                    border-radius: 4px;
                    color: {theme['text_secondary']};
                    outline: none;
                }}
                QTreeView::item {{
                    padding: 4px;
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
        if os.path.isfile(path) and path.lower().endswith(('.fits', '.fit')):
            if self.parent:
                self.parent.load_fits_file(path)
        elif os.path.isdir(path):
            self.set_path(path)


class ParametersPanel(QWidget):
    """Panel de parámetros con tema aplicado"""
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.theme_manager = theme_manager
        self.current_params = DEFAULT_PARAMS.copy()
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.update_scrollarea_style(scroll_area)
        
        scroll_widget = QWidget()
        self.update_widget_style(scroll_widget)
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        
        # Parámetros de procesamiento
        params_group = QGroupBox("Parámetros de Procesamiento")
        self.update_groupbox_style(params_group)
        params_layout = QGridLayout(params_group)
        params_layout.setVerticalSpacing(8)
        params_layout.setHorizontalSpacing(10)
        
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
            combo.setMinimumHeight(25)
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
        
        # Líneas espectrales
        lines_group = QGroupBox("Líneas Espectrales de Referencia")
        self.update_groupbox_style(lines_group)
        lines_layout = QVBoxLayout(lines_group)
        
        self.lines_table = QTableWidget()
        self.lines_table.setColumnCount(2)
        self.lines_table.setHorizontalHeaderLabels(["Línea", "Longitud de Onda (Å)"])
        self.lines_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lines_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.lines_table.setMaximumHeight(150)
        
        self.update_table_style(self.lines_table)
        self.update_spectral_lines_table()
        
        lines_layout.addWidget(self.lines_table)
        
        scroll_layout.addWidget(params_group)
        scroll_layout.addWidget(lines_group)
        scroll_layout.addStretch(1)
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
    
    def update_style(self):
        self.update_scrollarea_style(self.findChild(QScrollArea))
        
    def update_scrollarea_style(self, scroll_area):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            scroll_area.setStyleSheet(f"""
                QScrollArea {{
                    background-color: {theme['secondary']};
                    border: none;
                }}
                QScrollArea > QWidget > QWidget {{
                    background-color: {theme['secondary']};
                }}
                QScrollBar:vertical {{
                    background-color: {theme['secondary']};
                    width: 15px;
                    margin: 0px;
                }}
                QScrollBar::handle:vertical {{
                    background-color: {theme['accent']};
                    border-radius: 7px;
                    min-height: 20px;
                }}
                QScrollBar::handle:vertical:hover {{
                    background-color: {theme['accent_hover']};
                }}
                QScrollBar:horizontal {{
                    background-color: {theme['secondary']};
                    height: 15px;
                    margin: 0px;
                }}
                QScrollBar::handle:horizontal {{
                    background-color: {theme['accent']};
                    border-radius: 7px;
                    min-width: 20px;
                }}
                QScrollBar::handle:horizontal:hover {{
                    background-color: {theme['accent_hover']};
                }}
            """)
            
    def update_widget_style(self, widget):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            widget.setStyleSheet(f"background-color: {theme['secondary']};")
            
    def update_groupbox_style(self, groupbox):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            groupbox.setStyleSheet(f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 2px solid {theme['border']};
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                    background-color: {theme['secondary']};
                    color: {theme['text_primary']};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: {theme['accent']};
                }}
            """)
            
    def update_label_style(self, label):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            label.setStyleSheet(f"color: {theme['text_secondary']}; font-weight: bold;")
            
    def update_combobox_style(self, combobox):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            combobox.setStyleSheet(f"""
                QComboBox {{
                    background-color: {theme['secondary']};
                    border: 1px solid {theme['border']};
                    border-radius: 3px;
                    padding: 4px;
                    color: {theme['text_secondary']};
                    min-width: 80px;
                }}
                QComboBox:focus {{
                    border: 1px solid {theme['accent']};
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    border-left: 5px solid transparent;
                    border-right: 5px solid transparent;
                    border-top: 5px solid {theme['text_secondary']};
                    width: 0px;
                    height: 0px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {theme['secondary']};
                    border: 1px solid {theme['border']};
                    color: {theme['text_secondary']};
                    selection-background-color: {theme['accent']};
                }}
            """)
            
    def update_table_style(self, table):
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            table.setStyleSheet(f"""
                QTableWidget {{
                    background-color: {theme['primary']};
                    border: 1px solid {theme['border']};
                    border-radius: 4px;
                    color: {theme['text_secondary']};
                    gridline-color: {theme['border']};
                }}
                QTableWidget::item {{
                    padding: 6px;
                    border-bottom: 1px solid {theme['border']};
                }}
                QTableWidget::item:selected {{
                    background-color: {theme['accent']};
                    color: #ffffff;
                }}
                QHeaderView::section {{
                    background-color: {theme['secondary']};
                    color: {theme['accent']};
                    font-weight: bold;
                    padding: 6px;
                    border: none;
                    border-bottom: 2px solid {theme['accent']};
                }}
            """)
        
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


class MplCanvas(FigureCanvas):
    """Widget de matplotlib con tema aplicado"""
    def __init__(self, parent=None, width=8, height=6, dpi=100, theme_manager=None):
        self.theme_manager = theme_manager
        
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
            
            self.fig.patch.set_facecolor(theme['primary'])
            self.ax.set_facecolor(theme['secondary'])
            self.ax.tick_params(colors=theme['text_secondary'])
            
            for spine in self.ax.spines.values():
                spine.set_color(theme['border'])
                
            self.ax.title.set_color(theme['text_primary'])
            self.ax.xaxis.label.set_color(theme['text_secondary'])
            self.ax.yaxis.label.set_color(theme['text_secondary'])
            self.ax.grid(True, alpha=0.2, color=theme['border'])


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAMOST Spectrum Analyzer")
        self.setGeometry(100, 100, 1600, 900)
        
        # Inicializar componentes
        self.theme_manager = ThemeManager()
        self.file_path = None
        self.wl = None
        self.flux = None
        self.ivar = None
        self.report = None
        self.current_params = DEFAULT_PARAMS.copy()
        
        # Configurar interfaz - PRIMERO inicializar UI, LUEGO aplicar tema
        self.init_ui()
        self.create_menu()
        self.apply_theme()  # Ahora se llama después de init_ui()
        
    def apply_theme(self):
        """Aplica el tema actual a toda la aplicación"""
        theme = self.theme_manager.get_current_theme()
        
        # Configurar paleta
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
        
        # Actualizar componentes - ahora existen porque init_ui() ya se llamó
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
        
        return f"""
            QMainWindow {{
                background-color: {theme['primary']};
                color: {theme['text_primary']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {theme['border']};
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: {theme['secondary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {theme['accent']};
            }}
            QPushButton {{
                background-color: {theme['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 20px;
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
                padding: 4px;
            }}
            QProgressBar {{
                border: 1px solid {theme['border']};
                border-radius: 4px;
                text-align: center;
                color: {theme['text_primary']};
                background-color: {theme['secondary']};
            }}
            QProgressBar::chunk {{
                background-color: {theme['accent']};
                border-radius: 3px;
            }}
            QTextEdit {{
                background-color: {theme['primary']};
                color: {theme['text_secondary']};
                border: 1px solid {theme['border']};
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
            QMenuBar {{
                background-color: {theme['secondary']};
                color: {theme['text_primary']};
                border-bottom: 1px solid {theme['border']};
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 4px 8px;
            }}
            QMenuBar::item:selected {{
                background-color: {theme['tertiary']};
            }}
            QMenu {{
                background-color: {theme['secondary']};
                color: {theme['text_primary']};
                border: 1px solid {theme['border']};
            }}
            QMenu::item {{
                padding: 4px 16px;
            }}
            QMenu::item:selected {{
                background-color: {theme['accent']};
            }}
        """
        
    def create_menu(self):
        menubar = self.menuBar()

        # Menú File
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open FITS", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("Save Results", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Menú Edit
        edit_menu = menubar.addMenu("Edit")
        copy_action = QAction("Copy Results", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.copy_results)
        edit_menu.addAction(copy_action)
        
        clear_action = QAction("Clear Results", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self.clear_results)
        edit_menu.addAction(clear_action)

        # Menú View
        view_menu = menubar.addMenu("View")
        reset_plot_action = QAction("Reset Plot", self)
        reset_plot_action.setShortcut("Ctrl+R")
        reset_plot_action.triggered.connect(self.reset_plot)
        view_menu.addAction(reset_plot_action)
        
        toggle_toolbar_action = QAction("Toggle Toolbar", self)
        toggle_toolbar_action.setShortcut("Ctrl+T")
        toggle_toolbar_action.triggered.connect(self.toggle_toolbar)
        view_menu.addAction(toggle_toolbar_action)
        
        view_menu.addSeparator()
        fullscreen_action = QAction("Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Menú Tools
        tools_menu = menubar.addMenu("Tools")
        analyze_action = QAction("Run Analysis", self)
        analyze_action.setShortcut("F5")
        analyze_action.triggered.connect(self.analyze)
        tools_menu.addAction(analyze_action)
        
        tools_menu.addSeparator()
        settings_action = QAction("Theme Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_theme_settings)
        tools_menu.addAction(settings_action)
        
        batch_action = QAction("Batch Processing", self)
        batch_action.triggered.connect(self.batch_processing)
        tools_menu.addAction(batch_action)

        # Menú Help
        help_menu = menubar.addMenu("Help")
        docs_action = QAction("Documentation", self)
        docs_action.setShortcut("F1")
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        help_menu.addSeparator()
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(350)
        left_panel.setMaximumWidth(450)
        
        # Explorador de archivos
        explorer_group = QGroupBox("Project Explorer")
        explorer_layout = QVBoxLayout(explorer_group)
        self.file_explorer = FileExplorerWidget(self, self.theme_manager)
        explorer_layout.addWidget(self.file_explorer)
        
        # Información de archivo
        file_info_group = QGroupBox("File Information")
        file_info_layout = QVBoxLayout(file_info_group)
        self.file_label = QLabel("No hay archivo seleccionado")
        self.file_label.setWordWrap(True)
        file_info_layout.addWidget(self.file_label)
        
        # Análisis
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
        center_layout.setSpacing(5)
        
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100, theme_manager=self.theme_manager)
        self.toolbar = ThemeAwareNavigationToolbar(self.canvas, self, self.theme_manager)
        center_layout.addWidget(self.toolbar)
        center_layout.addWidget(self.canvas)
        
        results_label = QLabel("Resultados del análisis:")
        results_label.setStyleSheet("font-weight: bold;")
        center_layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        center_layout.addWidget(self.results_text)
        
        # Panel derecho
        right_panel = QWidget()
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(500)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.parameters_panel = ParametersPanel(self, self.theme_manager)
        right_layout.addWidget(self.parameters_panel)
        
        # Ensamblar interfaz
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800, 400])
        
        main_layout.addWidget(splitter)
        
    def update_file_label_style(self):
        """Actualiza el estilo de la etiqueta de información de archivo"""
        theme = self.theme_manager.get_current_theme()
        if self.file_path:
            self.file_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {theme['primary']};
                    border: 1px solid {theme['accent']};
                    border-radius: 4px;
                    padding: 8px;
                    color: {theme['text_primary']};
                    min-height: 60px;
                }}
            """)
        else:
            self.file_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {theme['primary']};
                    border: 1px solid {theme['border']};
                    border-radius: 4px;
                    padding: 8px;
                    color: {theme['text_muted']};
                    min-height: 60px;
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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo FITS", "", "FITS Files (*.fits *.fit)"
        )
        if file_path:
            self.load_fits_file(file_path)
            
    def analyze(self):
        if not self.file_path:
            QMessageBox.warning(self, "Advertencia", "Seleccione un archivo FITS primero.")
            return
            
        try:
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            self.wl, self.flux, self.ivar = read_fits_file(self.file_path)
            self.progress_bar.setValue(30)
            
            params = self.current_params
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
            <li>Análisis espectral automático</li>
            <li>Detección de líneas espectrales</li>
            <li>Cálculo de redshift y velocidad radial</li>
            <li>Visualización interactiva</li>
            <li>Temas personalizables (oscuro, claro, sistema)</li>
            <li>Colores secundarios configurables</li>
        </ul>
        <p><b>Atajos de teclado:</b></p>
        <ul>
            <li>Ctrl+O: Abrir archivo</li>
            <li>Ctrl+S: Guardar resultados</li>
            <li>F5: Ejecutar análisis</li>
            <li>Ctrl+R: Reiniciar gráficos</li>
            <li>Ctrl+,: Configuración de temas</li>
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
        <p>Versión con temas personalizables</p>
        <p>Herramienta para análisis espectral de datos FITS</p>
        <p>Desarrollado con PyQt5 y matplotlib</p>
        <p><b>Tema actual:</b> {self.theme_manager.current_theme_name.title()}</p>
        <p><b>Color acento:</b> <span style="color: {theme['accent']};">{theme['accent']}</span></p>
        <hr>
        <p style="color: {theme['text_muted']};">© 2024 LAMOST Analysis Team</p>
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
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()