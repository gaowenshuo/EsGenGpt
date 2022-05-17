import ui_test
from PySide6 import QtCore
from PySide6.QtWidgets import QWidget,QApplication
import EsGenGpt_predict_withGUI
from sys import exit

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.begin_text = ''
        self.text_len = 0
        self.ui = ui_test.Ui_producer()
        self.ui.setupUi(self)

    @QtCore.Slot()
    def ui_start(self):
        self.begin_text = self.ui.lineEdit.text()
        self.text_len = self.ui.spinBox.value()
        self.ui.resultText.setText(EsGenGpt_predict_withGUI.predict(self.begin_text,self.text_len))
if __name__ == "__main__":
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    exit(app.exec())