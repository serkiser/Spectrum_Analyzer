import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("Prueba GUI")
label = QLabel("¡PyQt5 funciona!")
window.setCentralWidget(label)
window.show()
sys.exit(app.exec_())