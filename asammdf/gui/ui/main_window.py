# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\DSUsers\uidn3651\02__PythonWorkspace\asammdf\asammdf\gui\ui\main_window.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PyMDFMainWindow(object):
    def setupUi(self, PyMDFMainWindow):
        PyMDFMainWindow.setObjectName("PyMDFMainWindow")
        PyMDFMainWindow.resize(800, 723)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/asammdf.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        PyMDFMainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(PyMDFMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.stackedWidget.setObjectName("stackedWidget")
        self.stackedWidgetPage1 = QtWidgets.QWidget()
        self.stackedWidgetPage1.setObjectName("stackedWidgetPage1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.stackedWidgetPage1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.files = QtWidgets.QTabWidget(self.stackedWidgetPage1)
        self.files.setDocumentMode(False)
        self.files.setTabsClosable(True)
        self.files.setObjectName("files")
        self.verticalLayout_2.addWidget(self.files)
        self.stackedWidget.addWidget(self.stackedWidgetPage1)
        self.stackedWidgetPage2 = QtWidgets.QWidget()
        self.stackedWidgetPage2.setObjectName("stackedWidgetPage2")
        self.files_layout = QtWidgets.QGridLayout(self.stackedWidgetPage2)
        self.files_layout.setObjectName("files_layout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.cs_split_size = QtWidgets.QDoubleSpinBox(self.stackedWidgetPage2)
        self.cs_split_size.setObjectName("cs_split_size")
        self.gridLayout_2.addWidget(self.cs_split_size, 5, 1, 1, 1)
        self.cs_split = QtWidgets.QCheckBox(self.stackedWidgetPage2)
        self.cs_split.setObjectName("cs_split")
        self.gridLayout_2.addWidget(self.cs_split, 4, 0, 1, 1)
        self.cs_format = QtWidgets.QComboBox(self.stackedWidgetPage2)
        self.cs_format.setObjectName("cs_format")
        self.gridLayout_2.addWidget(self.cs_format, 3, 1, 1, 1)
        self.sync = QtWidgets.QCheckBox(self.stackedWidgetPage2)
        self.sync.setChecked(True)
        self.sync.setObjectName("sync")
        self.gridLayout_2.addWidget(self.sync, 1, 0, 1, 1)
        self.cs_compression = QtWidgets.QComboBox(self.stackedWidgetPage2)
        self.cs_compression.setObjectName("cs_compression")
        self.gridLayout_2.addWidget(self.cs_compression, 6, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 7, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.stackedWidgetPage2)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.concatenate = QtWidgets.QRadioButton(self.groupBox)
        self.concatenate.setChecked(True)
        self.concatenate.setObjectName("concatenate")
        self.verticalLayout_3.addWidget(self.concatenate)
        self.stack = QtWidgets.QRadioButton(self.groupBox)
        self.stack.setObjectName("stack")
        self.verticalLayout_3.addWidget(self.stack)
        self.gridLayout_2.addWidget(self.groupBox, 0, 0, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.stackedWidgetPage2)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 6, 0, 1, 1)
        self.cs_btn = QtWidgets.QPushButton(self.stackedWidgetPage2)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/stack.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cs_btn.setIcon(icon1)
        self.cs_btn.setObjectName("cs_btn")
        self.gridLayout_2.addWidget(self.cs_btn, 8, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 5, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.stackedWidgetPage2)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 3, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.stackedWidgetPage2)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 5, 0, 1, 1)
        self.add_samples_origin = QtWidgets.QCheckBox(self.stackedWidgetPage2)
        self.add_samples_origin.setChecked(True)
        self.add_samples_origin.setObjectName("add_samples_origin")
        self.gridLayout_2.addWidget(self.add_samples_origin, 2, 0, 1, 1)
        self.files_layout.addLayout(self.gridLayout_2, 1, 0, 1, 1)
        self.files_layout.setRowStretch(0, 1)
        self.stackedWidget.addWidget(self.stackedWidgetPage2)
        self.verticalLayout.addWidget(self.stackedWidget)
        PyMDFMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(PyMDFMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        PyMDFMainWindow.setMenuBar(self.menubar)
        self.action_memory_minimum = QtWidgets.QAction(PyMDFMainWindow)
        self.action_memory_minimum.setCheckable(True)
        self.action_memory_minimum.setObjectName("action_memory_minimum")
        self.action_memory_full = QtWidgets.QAction(PyMDFMainWindow)
        self.action_memory_full.setCheckable(True)
        self.action_memory_full.setObjectName("action_memory_full")
        self.action_memory_low = QtWidgets.QAction(PyMDFMainWindow)
        self.action_memory_low.setCheckable(True)
        self.action_memory_low.setObjectName("action_memory_low")

        self.retranslateUi(PyMDFMainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.files.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(PyMDFMainWindow)

    def retranslateUi(self, PyMDFMainWindow):
        _translate = QtCore.QCoreApplication.translate
        PyMDFMainWindow.setWindowTitle(_translate("PyMDFMainWindow", "asammdf"))
        self.cs_split_size.setSuffix(_translate("PyMDFMainWindow", "MB"))
        self.cs_split.setText(_translate("PyMDFMainWindow", "Split data blocks"))
        self.sync.setText(_translate("PyMDFMainWindow", "Sync using measurements timestamps"))
        self.groupBox.setTitle(_translate("PyMDFMainWindow", "Operation"))
        self.concatenate.setText(_translate("PyMDFMainWindow", "Concatenate"))
        self.stack.setText(_translate("PyMDFMainWindow", "Stac&k"))
        self.label_3.setText(_translate("PyMDFMainWindow", "Compression"))
        self.cs_btn.setText(_translate("PyMDFMainWindow", "Concatenate"))
        self.label.setText(_translate("PyMDFMainWindow", "Output format"))
        self.label_2.setText(_translate("PyMDFMainWindow", "Split block size"))
        self.add_samples_origin.setText(_translate("PyMDFMainWindow", "Add samples origin file"))
        self.action_memory_minimum.setText(_translate("PyMDFMainWindow", "minimum"))
        self.action_memory_minimum.setToolTip(_translate("PyMDFMainWindow", "Minimal memory usage by loading only the nedded block addresses"))
        self.action_memory_full.setText(_translate("PyMDFMainWindow", "full"))
        self.action_memory_full.setToolTip(_translate("PyMDFMainWindow", "Load all blocks in the RAM"))
        self.action_memory_low.setText(_translate("PyMDFMainWindow", "low"))
        self.action_memory_low.setToolTip(_translate("PyMDFMainWindow", "Load metdata block in RAM but leave the samples on disk"))


from . import resource_rc
