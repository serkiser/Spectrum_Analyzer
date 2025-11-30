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
                             QSplitter, QHeaderView, QAbstractItemView, QLineEdit,
                             QTabWidget, QFrame)
from PyQt5.QtCore import Qt, QDir
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon  # CORREGIDO: QPalette en lugar de Palettqte
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np

# Importaciones de tu proyecto
from lamost_analyzer.core.fits_processor import read_fits_file, valid_mask, rebin_spectrum
from lamost_analyzer.core.utils import try_savgol, running_percentile, enhance_line_detection
from lamost_analyzer.core.spectral_analysis import generate_spectral_report
from lamost_analyzer.config import DEFAULT_PARAMS, SPECTRAL_LINES


class DarkNavigationToolbar(NavigationToolbar):
    """Toolbar personalizada para tema oscuro"""
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        # Aplicar estilo oscuro a los botones
        self.setStyleSheet("""
            QToolButton {
                background-color: #2d2d30;
                border: 1px solid #444444;
                border-radius: 3px;
                color: #ffffff;
                padding: 4px;
            }
            QToolButton:hover {
                background-color: #3e3e42;
                border: 1px solid #007acc;
            }
            QToolButton:pressed {
                background-color: #007acc;
            }
        """)


class FileExplorerWidget(QWidget):
    """Widget personalizado para el explorador de archivos"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def create_icon_button(self, text, tooltip):
        """Crea un botón con ícono y texto"""
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedHeight(28)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d30;
                border: 1px solid #444444;
                border-radius: 3px;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                padding: 4px 8px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
                border: 1px solid #007acc;
            }
            QPushButton:pressed {
                background-color: #007acc;
            }
            QPushButton:disabled {
                background-color: #1e1e1e;
                color: #666666;
                border: 1px solid #333333;
            }
        """)
        return btn
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Barra de herramientas del explorador
        toolbar_layout = QHBoxLayout()
        
        # Crear botones con la nueva función
        self.btn_back = self.create_icon_button("◀ Atrás", "Volver al directorio anterior")
        self.btn_forward = self.create_icon_button("Adelante ▶", "Avanzar al siguiente directorio")
        self.btn_up = self.create_icon_button("↑ Subir", "Subir al directorio padre")
        self.btn_home = self.create_icon_button("🏠 Inicio", "Ir al directorio home")
        self.btn_refresh = self.create_icon_button("🔄 Actualizar", "Refrescar vista")
        
        # Añadir botones al layout
        toolbar_layout.addWidget(self.btn_back)
        toolbar_layout.addWidget(self.btn_forward)
        toolbar_layout.addWidget(self.btn_up)
        toolbar_layout.addWidget(self.btn_home)
        toolbar_layout.addWidget(self.btn_refresh)
        toolbar_layout.addStretch()
        
        # Ruta actual - ahora editable
        path_layout = QHBoxLayout()
        path_label = QLabel("Ruta:")
        path_label.setStyleSheet("color: #cccccc;")
        
        self.path_edit = QLineEdit()
        self.path_edit.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 4px 8px;
                color: #cccccc;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
                selection-background-color: #007acc;
            }
            QLineEdit:focus {
                border: 1px solid #007acc;
            }
        """)
        self.path_edit.setText(QDir.currentPath())
        self.path_edit.returnPressed.connect(self.on_path_edited)
        
        self.btn_go = QPushButton("Ir")
        self.btn_go.setFixedSize(30, 25)
        self.btn_go.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                border: 1px solid #007acc;
                border-radius: 3px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        self.btn_go.clicked.connect(self.on_path_edited)
        
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.btn_go)
        
        # TreeView para el explorador de archivos
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
        
        # Ocultar columnas que no necesitamos
        self.tree_view.hideColumn(1)  # Tamaño
        self.tree_view.hideColumn(2)  # Tipo
        self.tree_view.hideColumn(3)  # Fecha modificación
        
        # Configurar estilo del tree view
        self.tree_view.setStyleSheet("""
            QTreeView {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #cccccc;
                outline: none;
            }
            QTreeView::item {
                padding: 4px;
                border: none;
            }
            QTreeView::item:selected {
                background-color: #007acc;
                color: #ffffff;
            }
            QTreeView::item:hover {
                background-color: #3e3e42;
            }
        """)
        
        # Conectar señales
        self.tree_view.doubleClicked.connect(self.on_file_double_clicked)
        self.tree_view.clicked.connect(self.on_tree_selection_changed)
        self.btn_back.clicked.connect(self.go_back)
        self.btn_forward.clicked.connect(self.go_forward)
        self.btn_up.clicked.connect(self.go_up)
        self.btn_home.clicked.connect(self.go_home)
        self.btn_refresh.clicked.connect(self.refresh)
        
        # Historial de navegación
        self.history = []
        self.history_index = -1
        self.add_to_history(QDir.currentPath())
        
        layout.addLayout(toolbar_layout)
        layout.addLayout(path_layout)
        layout.addWidget(self.tree_view)
        
    def on_path_edited(self):
        """Maneja cuando el usuario edita y confirma una ruta"""
        path = self.path_edit.text().strip()
        
        # Limpiar y normalizar la ruta
        path = path.replace('\\', '/')
        
        # Si la ruta existe, navegar a ella
        if os.path.exists(path):
            if os.path.isdir(path):
                self.set_path(path)
            else:
                # Si es un archivo, ir al directorio padre
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir):
                    self.set_path(parent_dir)
                    # Seleccionar el archivo en el tree view
                    index = self.model.index(path)
                    if index.isValid():
                        self.tree_view.setCurrentIndex(index)
                        self.tree_view.scrollTo(index)
        else:
            # Mostrar mensaje de error
            QMessageBox.warning(self, "Ruta no válida", 
                               f"La ruta especificada no existe:\n{path}")
            # Revertir a la ruta actual
            current_path = self.tree_view.rootIndex().data(QFileSystemModel.FilePathRole)
            self.path_edit.setText(current_path)
            
    def on_tree_selection_changed(self, index):
        """Actualiza la barra de ruta cuando se selecciona un elemento en el tree"""
        if index.isValid():
            path = self.model.filePath(index)
            self.path_edit.setText(path)
        
    def add_to_history(self, path):
        """Añade una ruta al historial de navegación"""
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(path)
        self.history_index = len(self.history) - 1
        self.update_navigation_buttons()
        
    def update_navigation_buttons(self):
        """Actualiza el estado de los botones de navegación"""
        self.btn_back.setEnabled(self.history_index > 0)
        self.btn_forward.setEnabled(self.history_index < len(self.history) - 1)
        
    def go_back(self):
        """Navega hacia atrás en el historial"""
        if self.history_index > 0:
            self.history_index -= 1
            self.set_path(self.history[self.history_index])
            self.update_navigation_buttons()
            
    def go_forward(self):
        """Navega hacia adelante en el historial"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.set_path(self.history[self.history_index])
            self.update_navigation_buttons()
            
    def go_up(self):
        """Sube un nivel en el directorio"""
        current_path = self.tree_view.rootIndex().data(QFileSystemModel.FilePathRole)
        parent_dir = QDir(current_path)
        if parent_dir.cdUp():
            self.set_path(parent_dir.absolutePath())
            
    def go_home(self):
        """Va al directorio home"""
        home_path = QDir.homePath()
        self.set_path(home_path)
        
    def refresh(self):
        """Refresca la vista actual"""
        current_path = self.tree_view.rootIndex().data(QFileSystemModel.FilePathRole)
        self.model.setRootPath(current_path)
        self.tree_view.setRootIndex(self.model.index(current_path))
        
    def set_path(self, path):
        """Establece la ruta actual del explorador"""
        self.tree_view.setRootIndex(self.model.index(path))
        self.path_edit.setText(path)
        self.add_to_history(path)
        
    def on_file_double_clicked(self, index):
        """Maneja el doble click en un archivo o directorio"""
        path = self.model.filePath(index)
        if os.path.isfile(path) and path.lower().endswith(('.fits', '.fit')):
            # Es un archivo FITS - cargarlo
            if self.parent:
                self.parent.load_fits_file(path)
                # Cambiar a la pestaña de análisis automáticamente
                self.parent.tab_widget.setCurrentIndex(1)
        elif os.path.isdir(path):
            # Es un directorio - navegar a él
            self.set_path(path)


class AnalysisWidget(QWidget):
    """Widget para el análisis y visualización de espectros"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Información del archivo actual
        file_info_group = QGroupBox("Archivo Actual")
        file_info_layout = QVBoxLayout(file_info_group)
        
        self.file_label = QLabel("No hay archivo seleccionado")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 12px;
                color: #888888;
                min-height: 40px;
                font-size: 12px;
            }
        """)
        file_info_layout.addWidget(self.file_label)
        
        # Controles de análisis
        controls_layout = QHBoxLayout()
        
        self.btn_analyze = QPushButton("🔍 Ejecutar Análisis")
        self.btn_analyze.clicked.connect(self.parent.analyze)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 12px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
        """)
        
        self.btn_save = QPushButton("💾 Guardar Resultados")
        self.btn_save.clicked.connect(self.parent.save_results)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #388e3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 12px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #43a047;
            }
            QPushButton:pressed {
                background-color: #2e7d32;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
        """)
        
        controls_layout.addWidget(self.btn_analyze)
        controls_layout.addWidget(self.btn_save)
        controls_layout.addStretch()
        
        # Barra de progreso
        progress_layout = QVBoxLayout()
        progress_label = QLabel("Progreso del análisis:")
        progress_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                background-color: #2d2d30;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
        """)
        
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        # Canvas de matplotlib con toolbar
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100)
        self.toolbar = DarkNavigationToolbar(self.canvas, self)
        
        # Área de resultados
        results_group = QGroupBox("Resultados del Análisis")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(250)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        results_layout.addWidget(self.results_text)
        
        # Ensamblar layout principal
        layout.addWidget(file_info_group)
        layout.addLayout(controls_layout)
        layout.addLayout(progress_layout)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)  # El 1 le da más espacio al gráfico
        layout.addWidget(results_group)


class MplCanvas(FigureCanvas):
    """Widget de matplotlib para integrar en PyQt con tema oscuro"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Configurar estilo oscuro para matplotlib
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 1, figsize=(width, height), dpi=dpi)
        
        # Configurar colores del gráfico para tema oscuro
        self.fig.patch.set_facecolor('#1e1e1e')
        for ax in self.axes:
            ax.set_facecolor('#2d2d30')
            ax.tick_params(colors='#cccccc')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_color('#444444') 
            ax.spines['right'].set_color('#444444')
            ax.spines['left'].set_color('#444444')
            ax.title.set_color('#ffffff')
            ax.xaxis.label.set_color('#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            ax.grid(True, alpha=0.2, color='#444444')
            
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LAMOST Spectrum Analyzer - Dark Mode")
        self.setGeometry(100, 100, 1400, 900)
        
        # Variables de datos
        self.file_path = None
        self.wl = None
        self.flux = None
        self.ivar = None
        self.report = None
        
        # Aplicar tema oscuro
        self.apply_dark_theme()
        self.init_ui()
        self.create_menu()
        
    def apply_dark_theme(self):
        """Aplica el tema oscuro a toda la aplicación"""
        dark_palette = QPalette()
        
        # Colores base
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 48))
        dark_palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.ToolTipText, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.Text, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.Button, QColor(45, 45, 48))
        dark_palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(0, 122, 204))
        dark_palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
        dark_palette.setColor(QPalette.HighlightedText, QColor(45, 45, 48))
        
        self.setPalette(dark_palette)
        
        # Estilo CSS adicional
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #2d2d30;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #2d2d30;
                color: #cccccc;
                padding: 8px 16px;
                margin: 2px;
                border: 1px solid #444444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #007acc;
                color: #ffffff;
                border-color: #007acc;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3e3e42;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3e3e42;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #2d2d30;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #007acc;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
            QLabel {
                color: #cccccc;
                padding: 4px;
            }
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                background-color: #2d2d30;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QMenuBar {
                background-color: #2d2d30;
                color: #ffffff;
                border-bottom: 1px solid #444444;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
            }
            QMenuBar::item:selected {
                background-color: #3e3e42;
            }
            QMenu {
                background-color: #2d2d30;
                color: #ffffff;
                border: 1px solid #444444;
            }
            QMenu::item {
                padding: 4px 16px;
            }
            QMenu::item:selected {
                background-color: #007acc;
            }
        """)
        
    def create_menu(self):
        """Crea la barra de menús superior con File, Edit, View, Tools, Help"""
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

        # Menú View
        view_menu = menubar.addMenu("View")
        
        show_explorer_action = QAction("Show File Explorer", self)
        show_explorer_action.setShortcut("Ctrl+1")
        show_explorer_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        view_menu.addAction(show_explorer_action)
        
        show_analysis_action = QAction("Show Analysis", self)
        show_analysis_action.setShortcut("Ctrl+2")
        show_analysis_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        view_menu.addAction(show_analysis_action)
        
        view_menu.addSeparator()
        
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
        
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
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
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Crear widget de pestañas
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #2d2d30;
            }
        """)
        
        # Crear los widgets para cada pestaña
        self.file_explorer = FileExplorerWidget(self)
        self.analysis_widget = AnalysisWidget(self)
        
        # Añadir pestañas
        self.tab_widget.addTab(self.file_explorer, "📁 Explorador de Archivos")
        self.tab_widget.addTab(self.analysis_widget, "🔍 Análisis Espectral")
        
        main_layout.addWidget(self.tab_widget)
        
    def load_fits_file(self, file_path):
        """Carga un archivo FITS desde el explorador"""
        self.file_path = file_path
        filename = os.path.basename(file_path)
        
        # Actualizar información en ambas vistas
        file_info_text = f"📁 {filename}\n📍 {file_path}"
        
        # Actualizar widget de análisis
        self.analysis_widget.file_label.setText(file_info_text)
        self.analysis_widget.file_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 1px solid #007acc;
                border-radius: 4px;
                padding: 12px;
                color: #ffffff;
                min-height: 40px;
                font-size: 12px;
            }
        """)
        
        # Habilitar botones de análisis
        self.analysis_widget.btn_analyze.setEnabled(True)
        self.analysis_widget.results_text.append(f"✓ Archivo cargado: {filename}")
        
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo FITS", "", "FITS Files (*.fits *.fit)"
        )
        if file_path:
            self.load_fits_file(file_path)
            # Cambiar a la pestaña de análisis
            self.tab_widget.setCurrentIndex(1)
            
    def analyze(self):
        if not self.file_path:
            QMessageBox.warning(self, "Advertencia", "Seleccione un archivo FITS primero.")
            return
        try:
            self.analysis_widget.progress_bar.setValue(10)
            QApplication.processEvents()
            
            self.wl, self.flux, self.ivar = read_fits_file(self.file_path)
            self.analysis_widget.progress_bar.setValue(30)
            
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
                self.analysis_widget.results_text.append(f"⚠ SG_WINDOW ajustado a {current_sg_window}")

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

            self.analysis_widget.progress_bar.setValue(100)
            self.analysis_widget.btn_save.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error durante el análisis: {str(e)}")
            
    def display_results(self):
        if not self.report:
            return
        self.analysis_widget.results_text.clear()
        self.analysis_widget.results_text.append("=== REPORTE DE ANÁLISIS ESPECTRAL ===")
        self.analysis_widget.results_text.append(f"📊 Rango λ: {self.report['wavelength_range']['min']:.1f} - {self.report['wavelength_range']['max']:.1f} Å")
        self.analysis_widget.results_text.append(f"📈 SNR: {self.report['snr']:.1f}")
        if 'redshift' in self.report:
            z_info = self.report['redshift']
            rv_info = self.report['radial_velocity']
            self.analysis_widget.results_text.append(f"🔭 Redshift: {z_info['value']:.6f} ± {z_info['error']:.6f}")
            self.analysis_widget.results_text.append(f"🚀 Vel. radial: {rv_info['value']:.1f} ± {rv_info['error']:.1f} km/s")
        
    def plot_spectrum(self, wavelengths, flux_original, flux_processed, lines_dict):
        for ax in self.analysis_widget.canvas.axes:
            ax.clear()
            
        # Configurar colores para tema oscuro
        ax1 = self.analysis_widget.canvas.axes[0]
        ax1.plot(wavelengths, flux_original, color='#888888', alpha=0.6, linewidth=0.5, label="Original")
        ax1.plot(wavelengths, flux_processed, color='#0098ff', linewidth=1, label="Procesado")
        ax1.legend(facecolor='#2d2d30', edgecolor='#444444', labelcolor='#cccccc')
        ax1.set_title("Espectro completo", color='#ffffff')
        ax1.grid(True, alpha=0.2, color='#444444')

        ax2 = self.analysis_widget.canvas.axes[1]
        zoom_region = (5100, 5200)
        mask = (wavelengths >= zoom_region[0]) & (wavelengths <= zoom_region[1])
        if np.any(mask):
            ax2.plot(wavelengths[mask], flux_processed[mask], color='#0098ff', linewidth=1.5)
        ax2.set_title("Zoom 5100–5200 Å", color='#ffffff')
        ax2.grid(True, alpha=0.2, color='#444444')

        self.analysis_widget.canvas.fig.tight_layout()
        self.analysis_widget.canvas.draw()
        
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
        for ax in self.analysis_widget.canvas.axes:
            ax.clear()
            # Restaurar configuración de tema oscuro
            ax.set_facecolor('#2d2d30')
            ax.tick_params(colors='#cccccc')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_color('#444444') 
            ax.spines['right'].set_color('#444444')
            ax.spines['left'].set_color('#444444')
            ax.title.set_color('#ffffff')
            ax.xaxis.label.set_color('#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            ax.grid(True, alpha=0.2, color='#444444')
        self.analysis_widget.canvas.draw()

    def toggle_toolbar(self):
        """Muestra/oculta la toolbar de matplotlib"""
        self.analysis_widget.toolbar.setVisible(not self.analysis_widget.toolbar.isVisible())

    def toggle_fullscreen(self):
        """Alterna modo pantalla completa"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_settings(self):
        """Muestra diálogo de configuración"""
        QMessageBox.information(self, "Configuración", "Diálogo de configuración - En desarrollo")

    def batch_processing(self):
        """Procesamiento por lotes"""
        QMessageBox.information(self, "Procesamiento por lotes", "Funcionalidad de procesamiento por lotes - En desarrollo")

    def show_documentation(self):
        """Muestra la documentación"""
        about_text = """
        <h3>LAMOST Spectrum Analyzer - Documentación</h3>
        <p><b>Funcionalidades:</b></p>
        <ul>
            <li>Carga de archivos FITS</li>
            <li>Análisis espectral automático</li>
            <li>Detección de líneas espectrales</li>
            <li>Cálculo de redshift y velocidad radial</li>
            <li>Visualización interactiva</li>
        </ul>
        <p><b>Atajos de teclado:</b></p>
        <ul>
            <li>Ctrl+O: Abrir archivo</li>
            <li>Ctrl+S: Guardar resultados</li>
            <li>F5: Ejecutar análisis</li>
            <li>Ctrl+R: Reiniciar gráficos</li>
            <li>F1: Documentación</li>
            <li>Ctrl+1: Explorador de archivos</li>
            <li>Ctrl+2: Análisis espectral</li>
        </ul>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Documentación")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.exec_()

    def show_about(self):
        about_text = """
        <h3>LAMOST Spectrum Analyzer</h3>
        <p>Versión con interfaz de tema oscuro</p>
        <p>Herramienta para análisis espectral de datos FITS</p>
        <p>Desarrollado con PyQt5 y matplotlib</p>
        <hr>
        <p style="color: #888888;">© 2024 LAMOST Analysis Team</p>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Acerca de")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.exec_()


def run_gui():
    app = QApplication(sys.argv)
    
    # Configurar estilo de aplicación
    app.setStyle('Fusion')  # Fusion es un estilo que se adapta bien a temas personalizados
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()