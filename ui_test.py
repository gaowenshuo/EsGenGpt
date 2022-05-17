# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'test.ui'
##
## Created by: Qt User Interface Compiler version 6.3.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpinBox,
    QTextBrowser, QWidget)

class Ui_producer(object):
    def setupUi(self, producer):
        if not producer.objectName():
            producer.setObjectName(u"producer")
        producer.resize(1117, 700)
        self.gridLayout = QGridLayout(producer)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_start = QLabel(producer)
        self.label_start.setObjectName(u"label_start")
        font = QFont()
        font.setFamilies([u"\u534e\u6587\u4eff\u5b8b"])
        font.setPointSize(18)
        self.label_start.setFont(font)
        self.label_start.setMouseTracking(False)

        self.horizontalLayout.addWidget(self.label_start)

        self.lineEdit = QLineEdit(producer)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setMaximumSize(QSize(1000, 50))
        font1 = QFont()
        font1.setFamilies([u"\u4eff\u5b8b"])
        font1.setPointSize(14)
        self.lineEdit.setFont(font1)

        self.horizontalLayout.addWidget(self.lineEdit)

        self.label_num = QLabel(producer)
        self.label_num.setObjectName(u"label_num")
        self.label_num.setFont(font)

        self.horizontalLayout.addWidget(self.label_num)

        self.spinBox = QSpinBox(producer)
        self.spinBox.setObjectName(u"spinBox")
        font2 = QFont()
        font2.setFamilies([u"\u4eff\u5b8b"])
        font2.setPointSize(16)
        self.spinBox.setFont(font2)
        self.spinBox.setCursor(QCursor(Qt.IBeamCursor))
        self.spinBox.setMinimum(50)
        self.spinBox.setMaximum(5000)
        self.spinBox.setSingleStep(100)

        self.horizontalLayout.addWidget(self.spinBox)

        self.pushButton = QPushButton(producer)
        self.pushButton.setObjectName(u"pushButton")
        font3 = QFont()
        font3.setFamilies([u"\u4eff\u5b8b"])
        font3.setPointSize(20)
        self.pushButton.setFont(font3)

        self.horizontalLayout.addWidget(self.pushButton)


        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.resultText = QTextBrowser(producer)
        self.resultText.setObjectName(u"resultText")
        self.resultText.setEnabled(True)
        self.resultText.setMaximumSize(QSize(1677215, 1677215))
        font4 = QFont()
        font4.setFamilies([u"\u5b8b\u4f53"])
        font4.setPointSize(14)
        self.resultText.setFont(font4)

        self.gridLayout.addWidget(self.resultText, 1, 0, 1, 1)


        self.retranslateUi(producer)
        self.pushButton.clicked.connect(producer.ui_start)

        QMetaObject.connectSlotsByName(producer)
    # setupUi

    def retranslateUi(self, producer):
        producer.setWindowTitle(QCoreApplication.translate("producer", u"\u7533\u8bba\u751f\u6210\u5668", None))
        self.label_start.setText(QCoreApplication.translate("producer", u"\u5f00\u5934:", None))
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("producer", u"\u8bf7\u8f93\u5165\u5f00\u5934\u7684\u4e00\u53e5\u8bdd", None))
        self.label_num.setText(QCoreApplication.translate("producer", u"\u5b57\u6570:", None))
        self.spinBox.setPrefix("")
        self.pushButton.setText(QCoreApplication.translate("producer", u"\u751f\u6210", None))
        self.resultText.setPlaceholderText(QCoreApplication.translate("producer", u"\u6b64\u5904\u662f\u751f\u6210\u5668\u7684\u5185\u5bb9", None))
    # retranslateUi

