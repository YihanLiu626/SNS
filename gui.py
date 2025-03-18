# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gui.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
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
from PySide6.QtWidgets import (QApplication, QComboBox, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QStatusBar, QTableWidget, QTableWidgetItem,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(908, 680)
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(14)
        MainWindow.setFont(font)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.lineEdit_stock = QLineEdit(self.centralwidget)
        self.lineEdit_stock.setObjectName(u"lineEdit_stock")
        self.lineEdit_stock.setGeometry(QRect(190, 100, 121, 31))
        self.lineEdit_stock.setFont(font)
        self.lineEdit_stock.setStyleSheet(u"background-color: white;\n"
"    border: 1px solid #D0D3D4; /* \u6d45\u7070\u8272\u8fb9\u6846 */\n"
"    padding: 5px;\n"
"")
        self.label_stock = QLabel(self.centralwidget)
        self.label_stock.setObjectName(u"label_stock")
        self.label_stock.setGeometry(QRect(30, 100, 121, 31))
        font1 = QFont()
        self.label_stock.setFont(font1)
        self.label_stock.setStyleSheet(u"color: #2C3E50; /* \u6df1\u7070\u84dd\u8272 */\n"
"    font-size: 14px")
        self.comboBox_days = QComboBox(self.centralwidget)
        self.comboBox_days.addItem("")
        self.comboBox_days.addItem("")
        self.comboBox_days.addItem("")
        self.comboBox_days.addItem("")
        self.comboBox_days.setObjectName(u"comboBox_days")
        self.comboBox_days.setGeometry(QRect(210, 170, 104, 31))
        self.comboBox_days.setStyleSheet(u"background-color: white;\n"
"    border: 1px solid #D0D3D4; /* \u6d45\u7070\u8272\u8fb9\u6846 */\n"
"    padding: 5px;\n"
"")
        self.label_days = QLabel(self.centralwidget)
        self.label_days.setObjectName(u"label_days")
        self.label_days.setGeometry(QRect(30, 170, 171, 31))
        self.label_days.setFont(font1)
        self.label_days.setStyleSheet(u"color: #2C3E50; /* \u6df1\u7070\u84dd\u8272 */\n"
"    font-size: 14px")
        self.btn_predict = QPushButton(self.centralwidget)
        self.btn_predict.setObjectName(u"btn_predict")
        self.btn_predict.setGeometry(QRect(360, 100, 81, 71))
        self.btn_predict.setFont(font1)
        self.btn_predict.setStyleSheet(u"QPushButton {\n"
"    background-color: rgb(100, 149, 237); /* \u67d4\u548c\u84dd\u8272 */\n"
"    color: white;\n"
"    font-size: 14px;\n"
"    border-radius: 8px; /* \u5706\u89d2 */\n"
"    padding: 8px; /* \u589e\u52a0\u5185\u8fb9\u8ddd */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(70, 130, 180); /* \u9f20\u6807\u60ac\u505c\u53d8\u6df1\u4e00\u70b9 */\n"
"}")
        self.label_result = QLabel(self.centralwidget)
        self.label_result.setObjectName(u"label_result")
        self.label_result.setGeometry(QRect(500, 110, 351, 41))
        font2 = QFont()
        font2.setBold(True)
        self.label_result.setFont(font2)
        self.label_result.setStyleSheet(u"QLabel#label_result {\n"
"    font-size: 18px;\n"
"    font-weight: bold;\n"
"    color: #2C3E50; /* \u6df1\u8272\u5b57\u4f53 */\n"
"    padding: 5px;\n"
"    border-bottom: 2px solid rgb(176, 224, 230);\n"
"}")
        self.table_predictions = QTableWidget(self.centralwidget)
        if (self.table_predictions.columnCount() < 2):
            self.table_predictions.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.table_predictions.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.table_predictions.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        self.table_predictions.setObjectName(u"table_predictions")
        self.table_predictions.setGeometry(QRect(80, 240, 201, 311))
        self.table_predictions.setFont(font)
        self.table_predictions.setStyleSheet(u"QTableWidget {\n"
"    background-color: white; /* \u8868\u683c\u80cc\u666f */\n"
"    gridline-color: rgb(200, 200, 200); /* \u7f51\u683c\u7ebf\u989c\u8272 */\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"    background-color: rgb(100, 149, 237); /* \u67d4\u548c\u84dd\u8272 */\n"
"    color: black; /* \u6587\u5b57\u989c\u8272 */\n"
"    font-weight: bold;\n"
"    padding: 5px;\n"
"    border: 1px solid rgb(150, 200, 220); /* \u8f7b\u5fae\u8fb9\u6846 */\n"
"}")
        self.table_predictions.setColumnCount(2)
        self.plot_widget = QWidget(self.centralwidget)
        self.plot_widget.setObjectName(u"plot_widget")
        self.plot_widget.setGeometry(QRect(320, 190, 561, 421))
        self.plot_widget.setStyleSheet(u"QWidget#plot_widget {\n"
"    background-color: rgb(245, 245, 245);\n"
"    border: 2px solid rgb(176, 224, 230);\n"
"    border-radius: 10px;\n"
"}")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 908, 37))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.lineEdit_stock.setText("")
        self.lineEdit_stock.setPlaceholderText(QCoreApplication.translate("MainWindow", u"e.g.TSLA", None))
        self.label_stock.setText(QCoreApplication.translate("MainWindow", u"Enter stock ticker: ", None))
        self.comboBox_days.setItemText(0, QCoreApplication.translate("MainWindow", u"1", None))
        self.comboBox_days.setItemText(1, QCoreApplication.translate("MainWindow", u"5", None))
        self.comboBox_days.setItemText(2, QCoreApplication.translate("MainWindow", u"10", None))
        self.comboBox_days.setItemText(3, QCoreApplication.translate("MainWindow", u"30", None))

        self.label_days.setText(QCoreApplication.translate("MainWindow", u"Select days to predict:", None))
        self.btn_predict.setText(QCoreApplication.translate("MainWindow", u"Predict", None))
        self.label_result.setText(QCoreApplication.translate("MainWindow", u"Predicted Price: $0.00", None))
        ___qtablewidgetitem = self.table_predictions.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"Date", None));
        ___qtablewidgetitem1 = self.table_predictions.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Price", None));
    # retranslateUi

