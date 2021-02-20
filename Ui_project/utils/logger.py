from PySide2.QtWidgets import QTextBrowser, QLabel
from PySide2.QtGui import QColor, QPalette
from PySide2.QtCore import Qt
from datetime import datetime

specialLogs = {"SerialConnected":["Serial has been conncted successfully!","SUCCESS"],
                "SerialDisconnected":["Serial device is not connected now","INFO"],
                "ProcessingStarted":["The Processing has started","INFO"],
                "ProcessingStopped":["Processing has failed or was interrupted","ERROR"],
                "ProcessingCompleted":["Processing has been completed successfully","SUCCESS"]
}
typeColors = {"WARN":"orange", "ERROR":"red", "INFO":"black", "SUCCESS":"green"}

class Logger:
    def __init__(self, textBrowser, textLabel):
        self.logBox = QTextBrowser()
        self.labelBox = QLabel()
        self.logBox = textBrowser
        self.labelBox = textLabel
        self._enabled = True


    def log(self, logLine, type=None, special=None):
        if not self._enabled:
            return
        string = ""
        if logLine is "":
            try:
                logLine = specialLogs[special][0]
                if type is None:
                    type = specialLogs[special][1]
            except:
                print("An invalid special log has been provided: ", special)
        color = "black"
        if type is None:
            type = "INFO"
        string = string + type + ": "
        if type in typeColors.keys():
            color = typeColors[type]
        string = string + logLine
        self.logBox.append(string)
        self.labelBox.setStyleSheet("color: "+color)
        self.labelBox.setAlignment(Qt.AlignHCenter)
        if type in typeColors.keys():
            self.labelBox.setText(logLine)

    def saveLog(self, filePath):
        try:
            if filePath[-4:] != ".txt":
                filePath = filePath + ".txt"
            with open(filePath, mode='a') as outFile:
                outFile.write("\nThe following log has been saved at {} \n".format(datetime.now()) )
                outFile.write(self.logBox.toPlainText())
        except IOError:
            self.log("An error was found. Please check the output file path!", type="ERROR")
            return -1
        return 0
    def clearLog(self):
        self.logBox.clear()
        self.labelBox.clear()

    def enableLogging(self):
        self._enabled = True

    def disableLogging(self):
        self._enabled = False

