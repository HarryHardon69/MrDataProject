# ui.py
from PyQt5 import QtWidgets, QtGui, QtCore

class CollaborationUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Mr. Data Collaboration Window')
        self.setGeometry(100, 100, 800, 600)
        self.show()

def main():
    app = QtWidgets.QApplication([])
    ui = CollaborationUI()
    app.exec_()

if __name__ == "__main__":
    main()
