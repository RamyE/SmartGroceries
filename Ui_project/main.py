# print("This application may takes a minute or more to load, please wait")

# You can uncomment the following lines to ensure that this application uses the right DLLs even in the presence of other incompatible DLLs in the environment

# import os
# import PySide2
# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

import sys
from PySide2.QtWidgets import QApplication, QGridLayout, QMainWindow, QWidget, QComboBox, QPushButton, QLabel,\
                                QLineEdit, QTextBrowser, QProgressBar, QAction, QDialog, QFileDialog, \
                                    QMessageBox, QVBoxLayout, QCheckBox, QDialogButtonBox, QScrollArea
from PySide2.QtCore import QFile, QIODevice, Qt
from PySide2.QtGui import QFont, QIcon
from ui_mainwindow import Ui_MainWindow
from utils.logger import Logger
from utils.executer import Executer, ExecutionResult
from utils import utils
import pandas as pd
import serial
import time
import argparse

DISABLE_MODEL_TRASFER = True
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__(flags=Qt.WindowContextHelpButtonHint|Qt.WindowCloseButtonHint|Qt.CustomizeWindowHint)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setUpMainWindow()

    def setUpMainWindow(self):
        self.setWindowTitle("Lab Tools 1.0 (2021 Edition) - Applications of ML in Mechatronics")
        # set icon
        self.setUpIcon()
        
        #set up the info and start processing button
        self.inputLineEdit = QLineEdit()
        self.outputFolderLineEdit = QLineEdit()
        self.modelLineEdit = QLineEdit()
        self.browseOutputButton = QPushButton()
        self.browseInputButton = QPushButton()
        self.browseModelButton = QPushButton()
        self.startStopButton = QPushButton()
        self.useDefaultModelCheckbox = QCheckBox()

        self.inputLineEdit = self.findChild(QLineEdit, "inputLineEdit")
        self.modelLineEdit = self.findChild(QLineEdit, "modelLineEdit")
        self.outputFolderLineEdit = self.findChild(QLineEdit, "outputFolderLineEdit")
        self.browseOutputButton = self.findChild(QPushButton, "browseOutputButton")
        self.browseInputButton = self.findChild(QPushButton, "browseInputButton")
        self.browseModelButton = self.findChild(QPushButton, "browseModelButton")
        self.startStopButton = self.findChild(QPushButton, "startStopButton")
        self.useDefaultModelCheckbox = self.findChild(QCheckBox, "useDefaultModelCheckbox")
        self.chooseModelLabel = self.findChild(QLabel, "chooseModelLabel")

        #disabling model transfer capability if needed
        self._modelRPiPath = False
        if DISABLE_MODEL_TRASFER:
            self.useDefaultModelCheckbox.hide()
            self.modelLineEdit.setEnabled(1)
            self.browseModelButton.hide()
            self.chooseModelLabel.setText(self.chooseModelLabel.text()+" Name")
            self._modelRPiPath = True

        self.startStopButton.clicked.connect(self.handleStartStopButtonClicked)
        self.browseOutputButton.clicked.connect(self.handleBrowseOutputButton)
        self.browseInputButton.clicked.connect(self.handleBrowseInputButton)
        self.browseModelButton.clicked.connect(self.handleBrowseModelButton)
        self.useDefaultModelCheckbox.stateChanged.connect(self.handleUseDefaultModelCheckboxStateChanged)
        self.useDefaultModelCheckbox.setChecked(True)

        #set up the log and progress bar
        self.logTextBrowser = QTextBrowser()
        self.lastLogTextLabel = QLabel()
        self.logTextBrowser = self.findChild(QTextBrowser, "logTextBrowser")
        self.progressBar = self.findChild(QProgressBar, "progressBar")
        self.clearLogButton = self.findChild(QPushButton, "clearLogButton")
        self.saveLogButton = self.findChild(QPushButton, "saveLogButton")
        self.lastLogTextLabel = self.findChild(QLabel, "lastLogTextLabel")
        self.clearLogButton.clicked.connect(self.handleClearLogButton)
        self.saveLogButton.clicked.connect(self.handleSaveLogButton)

        #set up menu bar
        self.actionHelp = self.findChild(QAction, "actionHelp")
        self.actionAbout = self.findChild(QAction, "actionAbout")
        self.actionHelp.triggered.connect(self.handleActionHelpClicked)
        self.actionAbout.triggered.connect(self.handleActionAboutClicked)

        # Add additional menu actions
        self.utilitiesMenu = self.menuBar().addMenu("Utilities")
        self.actionGetRPiIP = QAction("Get RPi IP")
        self.utilitiesMenu.addAction(self.actionGetRPiIP)
        self.actionGetRPiIP.triggered.connect(self.handleActionGetRPiIPClicked)
        self.actionUpdateRPiScript = QAction("Updare RPi Script")
        self.utilitiesMenu.addAction(self.actionUpdateRPiScript)
        self.actionUpdateRPiScript.triggered.connect(self.handleActionUpdateRPiScript)

        #create objects from the other classes
        self.logger = Logger(self.logTextBrowser, self.lastLogTextLabel)

        #initialize member variables
        self._b_processRunning = False

        # set up lab names
        self.setUpLabNames()

        #set up serial comms
        self.setupSerial()
        self.refreshSerialPorts()
        
        self.logger.log("The application is ready!", type="INFO")

    def setUpIcon(self):
        self.appIcon = QIcon("images/favicon.png")
        self.setWindowIcon(self.appIcon)


    def setUpLabNames(self):
        self.labNameComboBox = QComboBox()

        self.labNameComboBox = self.findChild(QComboBox, "labNameComboBox")
        self.labNameComboBox.currentIndexChanged.connect(self.handleLabNameComboboxCurrentIndexChanged)
        for code, name in utils.lab_names.items():
            self.labNameComboBox.addItem(code+": "+name)
        self.labNameComboBox.setCurrentIndex(1)

    def setupSerial(self):
        self.refreshSerialPortsButton = QPushButton()
        self.connectDisconnectSerialButton = QPushButton()
        self.serialPortComboBox = QComboBox()

        self.refreshSerialPortsButton = self.findChild(QPushButton, "refreshSerialPortsButton")
        self.connectDisconnectSerialButton = self.findChild(QPushButton, "connectDisconnectSerialButton")
        self.serialPortComboBox = self.findChild(QComboBox, "serialPortComboBox")
        self.refreshSerialPortsButton.clicked.connect(self.refreshSerialPorts)
        self.connectDisconnectSerialButton.clicked.connect(self.handleSerialConnectDisconnect)
        self._b_serialConnected = False

    def refreshSerialPorts(self):
        availablePorts = utils.find_serial_ports()
        self.serialPortComboBox.clear()
        for portName in availablePorts:
            self.serialPortComboBox.addItem(portName)

    def handleSerialConnectDisconnect(self):
        if not self.b_serialConnected:
            try:
                currentPortName = self.serialPortComboBox.currentText()
                write_timeout = 4 if DISABLE_MODEL_TRASFER else 240
                self.port = serial.Serial(currentPortName, 115200 , timeout=1, write_timeout=write_timeout, bytesize=8, parity='N', stopbits=1)
                self.port.set_buffer_size(rx_size = 10**3, tx_size = 10**8)
                self.serialPortComboBox.setItemText(self.serialPortComboBox.currentIndex(), currentPortName + " (CONNECTED)")
                self.connectDisconnectSerialButton.setText("Disconnect")
                self.b_serialConnected = True
                self.refreshSerialPortsButton.setDisabled(1)
            except (OSError, serial.SerialException):
                print("Problem with Serial Connection!")
                self.logger.log("Problem with Serial Connection, Make sure you chose the right port", type="ERROR")
        else:
            try:
                self.port.close()
                self.refreshSerialPorts()
                self.connectDisconnectSerialButton.setText("Connect")
                self.b_serialConnected = False
                self.refreshSerialPortsButton.setEnabled(1)
            except (OSError, serial.SerialException):
                print("Problem with Serial Connection!")
                self.logger.log("Problem with Serial Connection", type="ERROR")

    def _startButtonClicked(self):
        self.logger.log("Attempting to start the processing", type="INFO")
        if not self.b_serialConnected:
            self.logger.log("Serial is not connected, Please connect serial first", type="ERROR")
            return
        if self.inputLineEdit.text()[-4:].lower() != ".csv" :
            self.logger.log("Please select a valid input csv file", type="ERROR")
            return
        if self.outputFolderLineEdit.text() == "":
            self.logger.log("Please select an output directory", type="ERROR")
            return
        self.executer = Executer(serialObj=self.port, loggerObj=self.logger)

        if self.modelLineEdit.text() != "":
            modelPath = self.modelLineEdit.text()
        else:
            modelPath = None
            if self._modelRPiPath:
                self.logger.log("Please select a valid model that is already available in the folder saved_models on the RPi", type="ERROR")
                return

        #Read the Input File
        try:
            inputDataFrame = pd.read_csv(self.inputLineEdit.text())
        except:
            self.logger.log("CSV File Reading Failed, select a valid csv file", type="ERROR")

        possibleInputs = list(inputDataFrame.columns)
        chosenInputs = self._askForFieldsDialog(possibleInputs, fields_type="inputs")
        if not chosenInputs:
            return
        chosenOutput = []
        if len(chosenInputs) < len(possibleInputs):
            chosenOutput = self._askForFieldsDialog(list(set(possibleInputs)-set(chosenInputs)), fields_type="output")

        self.startStopButton.setText("Stop Processing")
        self.b_processRunning = True
        executionResult = self.executer.execute(self.labNameComboBox.currentText().split(":")[0], inputDataFrame, \
                                self.outputFolderLineEdit.text(), inputFields=chosenInputs, outputField=chosenOutput, progressBar=self.progressBar, \
                                    model=modelPath if not self._modelRPiPath else "RPI:"+modelPath)
        if executionResult == ExecutionResult.COMPLETED:
            self._stopButtonClicked(finishedProcessing = True)
        elif executionResult == ExecutionResult.INTERRUPTED or executionResult == ExecutionResult.FAILED:
            self._stopButtonClicked()
            if self.executer.reset() == ExecutionResult.FAILED:
                self.logger.log("Resetting the serial state of RPi Failed, please try again or power cycle the RPi if the problem persists", type="ERROR")
            else:
                self.logger.log("The serial state of RPi has been reset", type="INFO")

    def _stopButtonClicked(self, finishedProcessing = False):
        self.startStopButton.setText("Start Processing")
        if finishedProcessing:
            self.logger.log("", special="ProcessingCompleted")
        else:
            self.logger.log("", special="ProcessingStopped")
        self.b_processRunning = False

    def handleStartStopButtonClicked(self):
        if (self.b_processRunning):
            self.executer.requestStop()
        else:
            self._startButtonClicked()
    
    def handleActionHelpClicked(self):
        helpBox = QMessageBox()
        helpBox.setIcon(QMessageBox.Information)
        helpBox.setStandardButtons(QMessageBox.Ok)
        helpBox.setWindowTitle("Need Help?")
        helpBox.setText("For Help, please reach out to your Instructor or TA or read the lab manual")
        helpBox.setTextFormat(Qt.RichText)
        helpBox.setInformativeText(f"You can access the project <a href=\"{utils.docs_link}\">Manual</a> and source in the <a href=\"{utils.repo_link}\">Github Repo!</a>")
        helpBox.setWindowIcon(self.appIcon)
        helpBox.exec_()

    def handleActionAboutClicked(self):
        aboutBox = QMessageBox()
        aboutBox.setIcon(QMessageBox.Information)
        aboutBox.setStandardButtons(QMessageBox.Ok)
        aboutBox.setWindowTitle("About the Software")
        aboutBox.setText(utils.license_text)
        aboutBox.setWindowIcon(self.appIcon)
        aboutBox.exec_()

    def handleBrowseInputButton(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("*.csv")
        if dialog.exec_():
            filePath = dialog.selectedFiles()
            if len(filePath) == 1:
                self.inputLineEdit.setText(filePath[0])

    def handleBrowseModelButton(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(utils.lab_model_extensions[self.labNameComboBox.currentText().split(":")[0]])
        if dialog.exec_():
            filePath = dialog.selectedFiles()
            if len(filePath) == 1:
                self.modelLineEdit.setText(filePath[0])

    def handleBrowseOutputButton(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly)
        if dialog.exec_():
            folderPath = dialog.selectedFiles()
            if len(folderPath) == 1:
                self.outputFolderLineEdit.setText(folderPath[0])

    def handleSaveLogButton(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.AcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("*.txt")
        if dialog.exec_():
            filePath = dialog.selectedFiles()
            if len(filePath) == 1:
                if self.logger.saveLog(filePath[0]) != -1:
                    self.logger.log("The log has been saved, feel free to clear the log now", type="SUCCESS")
                    return        
        self.logger.log("Failed to save the log, please select a valid file", type="ERROR")

    def handleClearLogButton(self):
        self.logger.clearLog()

    def handleUseDefaultModelCheckboxStateChanged(self):
        if self._modelRPiPath:
            return
        if not self.useDefaultModelCheckbox.checkState():
            self.modelLineEdit.setDisabled(0)
            self.browseModelButton.setDisabled(0)
        else:
            self.modelLineEdit.clear()
            self.modelLineEdit.setDisabled(1)
            self.browseModelButton.setDisabled(1)

    def handleLabNameComboboxCurrentIndexChanged(self):
        if self._modelRPiPath:
            self.modelLineEdit.setText(utils.lab_default_models[self.labNameComboBox.currentText().split(":")[0]])
        else:
            self.modelLineEdit.setText("")
        # Disable the model options when the lab selected is Communication Test
        if self.labNameComboBox.currentIndex() == 0:
            self.modelLineEdit.setDisabled(1)
            self.browseModelButton.setDisabled(1)
            self.useDefaultModelCheckbox.setDisabled(1)
            self.modelLineEdit.setText("Model is not required")
        else:
            self.handleUseDefaultModelCheckboxStateChanged()

    def handleActionGetRPiIPClicked(self):
        self.logger.log("Attempting to Get Raspberry Pi IP Address", type="INFO")
        if not self.b_serialConnected:
            replyMessage = "Serial is not connected, Please connect serial first"
        else:
            self.executer = Executer(serialObj=self.port, loggerObj=self.logger)
            ipaddr = self.executer.executeOther("GET_IP")
            replyMessage = ("Raspberry Pi IP is "+ ipaddr) if ipaddr != ExecutionResult.FAILED else "Failed to obtain the IP Address"
        self.logger.log(replyMessage)
        ipAddrMessage = QMessageBox()
        ipAddrMessage.setIcon(QMessageBox.Information)
        ipAddrMessage.setStandardButtons(QMessageBox.Ok)
        ipAddrMessage.setWindowTitle("Raspberry Pi IP Address")
        ipAddrMessage.setText(replyMessage)
        ipAddrMessage.setTextFormat(Qt.RichText)
        ipAddrMessage.setWindowIcon(self.appIcon)
        ipAddrMessage.exec_()

    def handleActionUpdateRPiScript(self):
        self.logger.log('Attempting to "git pull" for the Raspberry Pi Python Script', type="INFO")
        if not self.b_serialConnected:
            replyMessage = "Serial is not connected, Please connect serial first"
        else:
            self.executer = Executer(serialObj=self.port, loggerObj=self.logger)
            result = self.executer.executeOther("UPDATE_SCRIPT")
            if result != ExecutionResult.FAILED and "FAIL" not in result:
                replyMessage = "Raspberry Pi Script has been updated. You still need to reboot or power cycle the Raspberry Pi for the updated script to run" \
                                    "\nReceived: " + result
            else:
                replyMessage = "Failed to update the RPi Script"
                if "FAIL" in str(result):
                    replyMessage = replyMessage + "\nReceived: " + str(result)

        self.logger.log(replyMessage)
        ipAddrMessage = QMessageBox()
        ipAddrMessage.setIcon(QMessageBox.Information)
        ipAddrMessage.setStandardButtons(QMessageBox.Ok)
        ipAddrMessage.setWindowTitle("Raspberry Pi Script Updating Status")
        ipAddrMessage.setText(replyMessage)
        ipAddrMessage.setTextFormat(Qt.RichText)
        ipAddrMessage.setWindowIcon(self.appIcon)
        ipAddrMessage.exec_()

    def _askForFieldsDialog(self, options, fields_type="inputs"):
        #Display a dialog to ask the user to choose what inputs/outputs they want
        dialog = QDialog(self)

        dialog.setWindowTitle(f"Select the model {fields_type.upper()}")
        dialogButtons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialogButtons.button(QDialogButtonBox.Ok).setDisabled(0)
        dialogButtons.accepted.connect(dialog.accept)
        dialogButtons.rejected.connect(dialog.reject)
        
        mainLayout = QVBoxLayout(dialog)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        layoutWidget = QWidget()
        layout = QVBoxLayout(layoutWidget)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(layoutWidget)

        chosenFields=[]
        checkboxes=[]

        def handleCheckboxClicked():
            dialogButtons.button(QDialogButtonBox.Ok).setDisabled(1)
            count = 0
            for checkbox in checkboxes:
                if checkbox.isChecked():
                    count += 1
            if fields_type.lower() == "output":
                setDisabled = True if count > 1 else False
            else:
                setDisabled = True if count == 0 else False
            dialogButtons.button(QDialogButtonBox.Ok).setDisabled(setDisabled)

        for input in options:
            checkbox = QCheckBox(text=input)
            checkbox.clicked.connect(handleCheckboxClicked)
            checkbox.setChecked(True)
            checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        mainLayout.addWidget(QLabel(text=f"Please select the {fields_type.lower()} from the following:"))
        mainLayout.addWidget(scroll)
        mainLayout.addWidget(dialogButtons)
        dialog.setLayout(mainLayout)

        handleCheckboxClicked()

        if dialog.exec_() == QDialog.Accepted:
            for checkbox in checkboxes:
                if checkbox.isChecked():
                    chosenFields.append(checkbox.text())
            self.logger.log(f"The chosen {fields_type.lower()} are: "+ ', '.join(chosenFields), type ="INFO")
            return chosenFields
        else:
            return []


    @property
    def b_processRunning(self):
        return self._b_processRunning

    @property
    def b_serialConnected(self):
        return self._b_serialConnected

    @b_serialConnected.setter
    def b_serialConnected(self, newValue):
        self._b_serialConnected = newValue
        if newValue is True:
            self.logger.log("", special="SerialConnected")
        else:
            self.logger.log("", special="SerialDisconnected")

    @b_processRunning.setter
    def b_processRunning(self, newValue):
        self._b_processRunning = newValue
        if newValue is True:
            self.logger.log("", special="ProcessingStarted")


    def __del__(self):
        if self.b_serialConnected:
            self.port.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RPi HIL Communication GUI')
    parser.add_argument('--ignore-platform', default=False, action='store_true')
    parser.add_argument('--allow-resizing', default=False, action='store_true')
    parser.add_argument('--larger-font', default=False, action='store_true')
    args = parser.parse_args()

    if not sys.platform.startswith('win'):
        print("Unsupported Platform. This application is tested on Windows Only")
        if args.ignore_platform:
            print("The application will continue to run but may crash or not function as intended")
        else:
            exit(1)
    
    app = QApplication(sys.argv)

    if args.larger_font:
        font = app.font()
        font.setPointSize(app.font().pointSize()*1.5)
        font.setWeight(app.font().weight()*1.5)
        app.setFont(font)
        args.allow_resizing = True

    mainWindow = MainWindow()

    if not args.allow_resizing:
        mainWindow.setFixedSize(mainWindow.size())

    mainWindow.show()

    sys.exit(app.exec_())
