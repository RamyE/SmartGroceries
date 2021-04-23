import pandas as pd 
import enum
import serial
from serial.serialutil import SerialTimeoutException
from utils.utils import lab_names, lab_default_models
from utils.plotter import Plotter
from crccheck.crc import Crc16
from PySide2.QtCore import QMargins, QTimer, QCoreApplication
from PySide2.QtGui import QPixmap, Qt, QIcon, QPalette, QColor
from PySide2.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QListWidget, QMessageBox, QPushButton, QSpacerItem, QTextBrowser, QVBoxLayout, QSizePolicy, QScrollArea, QWidget
from datetime import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
import zlib
import copy
import random
import calendar
from functools import partial

SERIAL_COMMAND_TIMEOUT = 4000 #in ms
SAVE_MODEL_COMMAND_TIMEOUT = 240000 #in ms
SERIAL_COMMAND_MAX_TRIALS = 3 #in number of trials

SERIAL_COMMANDS = ["RESET", "SELECT_LAB", "SAVE_MODEL", "LOAD_MODEL", "PROCESS", "PROCESSING_DONE", "GET_IP", "UPDATE_SCRIPT",
                    "PROCESS_PROJECT_GROUP_2"]
STARTING_BYTE = 0x01

FAILURE_CODE = -1
SUCCESS_CODE = 0

class StopProcessingRequested(Exception):
    pass

class ExecutionResult(enum.Enum):
    COMPLETED = 0
    INTERRUPTED = 1
    FAILED = 2

class SerialState(enum.Enum):
    WaitingToStart = 0
    WaitingForString = 1
    WaitingForChecksum1 = 2
    WaitingForChecksum2 = 3
    CommandDone = 4 #Command Done -> Take action, send ACK and Go to 0
 
class ExecState(enum.Enum):
    NotConnected = 0
    Connected = 1 #Waiting to Select The Lab
    LabSelected = 2 #Ready to load model
    ModelLoaded = 3 #Ready to start processing
    Processing = 4
    Done = 5

class Executer:
    def __init__(self, serialObj, loggerObj):
        self.serialPort = serial.Serial()
        self.serialPort = serialObj
        self.logger = loggerObj
        self.log = self.logger.log
        self.execState = ExecState.NotConnected
            

        self.serialTimeoutTimer = QTimer()
        self.serialTimeoutTimer.setSingleShot(True)
        self.serialTimeoutTimer.setInterval(SERIAL_COMMAND_TIMEOUT)

        # self.checkStopRequestTimer = QTimer()
        # self.checkStopRequestTimer.setInterval(500)
        # self.checkStopRequestTimer.setSingleShot(False)
        # self.checkStopRequestTimer.timeout.connect(self.processCheckStopRequest)
        # self.checkStopRequestTimer.start()
        self._stopRequested  = False


    def execute(self, labCode, inputDataFrame, outputFolder, inputFields=None, outputField=None, progressBar=None, model=None):
        # self.logger.disableLogging()

        self.serialPort.flushInput()
        self.serialPort.flushOutput()
        startTime = time.time()

        # progressBar = None
        if progressBar is not None:
            progressBar.setValue(0)
        try:
            if self.execState == ExecState.NotConnected:
                if self.serialPort.isOpen():
                    self.execState = ExecState.Connected
                else:
                    #This should never happen because this function is called after serial is connected
                    self.log("Execution failed because serial port is not open, something is wrong", type="ERROR")
                    return ExecutionResult.FAILED

            if self.execState == ExecState.Connected:
                if self._sendCommand("SELECT_LAB", labCode, max_retry=SERIAL_COMMAND_MAX_TRIALS*4) == FAILURE_CODE:
                        self.log("Error occured with lab selection", type="ERROR")
                        return ExecutionResult.FAILED
                else:
                    self.execState = ExecState.LabSelected
    
            if self.execState == ExecState.LabSelected:
                if model is not None and not model.startswith("RPI:"):
                    self.log("Started sending the model, this could take a while, please wait", type="INFO")
                    if self._sendSaveModelCommand(model) == FAILURE_CODE:
                        self.log("Failed to send the selected model", type="ERROR")
                        return ExecutionResult.FAILED
                else:
                    if not model:
                        modelName = lab_default_models[labCode]
                    elif model.startswith("RPI:"):
                        modelName = model[4:]
                    if self._sendCommand("LOAD_MODEL", modelName, timeout=SERIAL_COMMAND_TIMEOUT*3) == FAILURE_CODE:
                        self.log("Failed to load the required model", type="ERROR")
                        return ExecutionResult.FAILED

                self.execState = ExecState.ModelLoaded

            if self.execState == ExecState.ModelLoaded:
                #load the inputs
                if inputFields is not None:
                    inputs = inputDataFrame[inputFields]
                else:
                    inputs = inputDataFrame
                if outputField:
                    trueOutput = inputDataFrame[outputField]
                else:
                    trueOutput = None
                self.execState = ExecState.Processing

            if self.execState == ExecState.Processing:
                if labCode == "LabTest":
                    executionResult = self._executeLab(inputs, outputFolder,
                        progressBar=progressBar, plotter=None)
                elif labCode == "Lab1":
                    executionResult = self._executeLab(inputs, outputFolder, outputHeader = "Prediction",
                        progressBar=progressBar, plotter=None, trueOutput=trueOutput, labCode=labCode)
                elif labCode == "Lab2":
                    executionResult = self._executeLab(inputs, outputFolder, outputHeader = "Prediction",
                        progressBar=progressBar, plotter=None, trueOutput=trueOutput, labCode=labCode)
                else:
                    raise ValueError("Lab Code should be one of the implemented lab codes for processing to work")
                    return ExecutionResult.FAILED
                if executionResult == FAILURE_CODE:
                    return ExecutionResult.FAILED
                else:
                    self.execState = ExecState.Done

            if self.execState == ExecState.Done:
                if (self._sendCommand("PROCESSING_DONE", "None") != FAILURE_CODE):
                    if progressBar is not None:
                        progressBar.setValue(100)
                    # self.logger.enableLogging()
                    self.log("Processing completed in {} ms".format((time.time()-startTime)*1000))
                    return ExecutionResult.COMPLETED
                else:
                    self.log("Failed to let RPi know that processing is done", "ERROR")
                    return ExecutionResult.FAILED

        except StopProcessingRequested:
            if progressBar is not None:
                progressBar.setValue(0)
            return ExecutionResult.INTERRUPTED
        except Exception as e:
            self.logger.enableLogging()
            self.log("Caught exception: {}".format(e), type="ERROR")
            self.log(traceback.format_exc())
            print(traceback.format_stack())
            return ExecutionResult.FAILED

    def executeOther(self, function, payload = "None"):
        if function not in SERIAL_COMMANDS:
            return ExecutionResult.FAILED
        result = self._sendCommand(function, payload)
        if result == FAILURE_CODE:
            return ExecutionResult.FAILED
        else:
            return result
    def executeProject(self, mainWindow, project_name):
        if project_name == "SmartGroceries":            
            self.referenceItemsLabels = {}
            self.itemInHand = None
            self.itemsInCart = list()
            self.currentOrderReferenceItems = list()
            self.currentOrderUserId = 0
            self.recommendedItemsButtons = list()
            self.previousRecommendations = list()
            self.numberOfGoodRecommendations = 0

            products = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects", "SmartGroceries", "data", "products.csv"))
            aisles = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects", "SmartGroceries", "data", "aisles.csv"))
            test_data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects", "SmartGroceries", "data", "test_set.csv"))
            # print(products)
            # print(aisles)
            # print(test_data)
            aisle_name_to_id = {k:v for k, v in zip(aisles.aisle, aisles.aisle_id)}
            product_name_to_id =  {k:v for k, v in zip(products.product_name, products.product_id)}
            product_id_to_name =  {k:v for k, v in zip(products.product_id, products.product_name)}

            def changeCurrentitem(itemName):
                if itemName == "":
                    self.itemInHand = None
                    currentItemLabel.setText(f"<b>Select an item from the list or from the recommendations</b>s")
                    addToCartButton.setEnabled(False)
                else:
                    self.itemInHand = product_name_to_id[itemName]
                    currentItemLabel.setText(f"Add <b>{itemName}</b> to Cart")
                    addToCartButton.setEnabled(True)
                    addToCartButton.setFocus()

            def handleNewOrderButtonClicked():
                grouped = test_data.groupby('order_id')
                while True:
                    order_number = random.sample(grouped.indices.keys(), 1)[0]
                    currentOrder = grouped.get_group(order_number)
                    self.currentOrderReferenceItems = currentOrder.product_id.tolist()
                    self.currentOrderUserId = currentOrder.user_id.iloc[0]
                    if len(self.currentOrderReferenceItems) > 1:
                        break
                print(self.currentOrderReferenceItems)
                orderInfo = f"<b>Order ID: </b>{currentOrder.order_id.iloc[0]}<br>"
                orderInfo += f"<b>User ID: </b>{self.currentOrderUserId} | <b>DOW: </b>{calendar.day_name[currentOrder.order_dow.iloc[0]]} | <b>Hour of Day: </b>{currentOrder.order_hour_of_day.iloc[0]}  | <b>Number of Items: </b>{len(self.currentOrderReferenceItems)}"
                orderInfo += "<br><b>Items in the Reference Order:</b>"

                for widget in self.referenceItemsLabels.values():
                    item = referenceItemsLayout.itemAt(0)
                    widget.setVisible(False)
                    referenceItemsLayout.removeItem(item)
                    del item
                self.referenceItemsLabels.clear()
                currentCartItems.clear()
                self.itemsInCart.clear()
                self.previousRecommendations.clear()
                self.numberOfGoodRecommendations = 0

                updateCurrentRecommendations(list())
                for product in self.currentOrderReferenceItems:
                    refItemName = product_id_to_name[product]
                    refItemLabel = QPushButton(refItemName)
                    refItemLabel.setContentsMargins(QMargins(0,0,0,0))
                    refItemLabel.setStyleSheet("Text-align:left")
                    refItemLabel.setFlat(False)
                    refItemLabel.clicked.connect(partial(changeCurrentitem, refItemName))
                    self.referenceItemsLabels[product]= refItemLabel
                    orderInfoLabel.setText(f"<b>Order Information</b><br>{orderInfo}")

                for referenceItemLabel in self.referenceItemsLabels.values():
                    referenceItemsLayout.addWidget(referenceItemLabel)

                runAutoButton.setFocus()

            def handleRunAutomatically():
                for referenceItemLabel in self.referenceItemsLabels.values():
                    referenceItemLabel.click()
                    addToCartButton.click()


            def updateCurrentRecommendations(recommendations):
                for widget in self.recommendedItemsButtons:
                    item = recommendationsLayout.itemAt(0)
                    widget.setVisible(False)
                    recommendationsLayout.removeItem(item)
                    del item
                self.recommendedItemsButtons.clear()
                for product in recommendations:
                    recItemName = product_id_to_name[product]
                    recItemButton = QPushButton(recItemName)
                    recItemButton.setContentsMargins(QMargins(0,0,0,0))
                    recItemButton.setStyleSheet("Text-align:left;")
                    if product not in self.currentOrderReferenceItems:
                        recItemButton.setFlat(True)
                    recItemButton.clicked.connect(partial(changeCurrentitem, recItemName))
                    self.recommendedItemsButtons.append(recItemButton)
                for recItemButton in self.recommendedItemsButtons:
                    recommendationsLayout.addWidget(recItemButton)
                if len(recommendations) > 0:
                    currentRecommendationsLabel.setVisible(True)
                else:
                    currentRecommendationsLabel.setVisible(False)
                self.previousRecommendations += recommendations

            def handleAddToCartButtonClicked():
                print(self.currentOrderReferenceItems)
                print(self.itemInHand)
                if self.itemInHand not in self.currentOrderReferenceItems:
                    QMessageBox(QMessageBox.Critical, "Error adding item to cart", "You can only add items that exists in the reference order").exec_()
                    return
                elif self.itemInHand in self.itemsInCart:
                    QMessageBox(QMessageBox.Critical, "Error adding item to cart", "This item is already in the cart").exec_()
                    return
                self.referenceItemsLabels[self.itemInHand].setFlat(True)
                self.itemsInCart.append(self.itemInHand)
                currentCartItems.addItem(product_id_to_name[self.itemInHand])
                if self.itemInHand in self.previousRecommendations:
                    self.numberOfGoodRecommendations+=1
                    self.referenceItemsLabels[self.itemInHand].setStyleSheet("Text-align:left; background-color:green;")
                    self.referenceItemsLabels[self.itemInHand].setFlat(False)

                #update recommendations
                result = self._sendCommand("PROCESS_PROJECT_GROUP_2", ";".join([str(self.currentOrderUserId), ",".join([str(x) for x in self.itemsInCart]), ",".join([str(x) for x in set(self.previousRecommendations)])]))
                if result == FAILURE_CODE:
                    self.log("Processing Failed, error getting recommendations from the RPi")
                    return
                else:
                    try:
                        recommendations = [int(id) for id in result.split(',')]
                    except:
                        recommendations = []

                updateCurrentRecommendations(recommendations)
                if len(self.itemsInCart) == len(self.currentOrderReferenceItems):
                    completionMessage = QMessageBox(QMessageBox.Information, "Order Completed", f"Order Completed with {self.numberOfGoodRecommendations} Good Recommendation(s)\nPress New Order to start a new order")
                    if self.numberOfGoodRecommendations == 0:
                        completionMessage.setIconPixmap(QPixmap('images/this_is_fine.jpg'))
                    completionMessage.setWindowIcon(appIcon)
                    completionMessage.exec_()
                    newOrderButton.setFocus()

            def aisleChanged():
                aisle_number = aisle_name_to_id[selectAisleCombobox.currentText()]
                products_in_aisle = products[products.aisle_id == aisle_number].product_name.tolist()
                selectproductCombobox.clear()
                selectproductCombobox.addItem("")
                selectproductCombobox.addItems(products_in_aisle)
            def itemChanged():
                current_item = selectproductCombobox.currentText()
                changeCurrentitem(current_item)
            
            dialog = QDialog(mainWindow)
            appIcon = QIcon("images/this_is_fine.jpg")
            dialog.setWindowIcon(appIcon)
            dialog.setMinimumWidth(600)
            dialog.setWindowTitle("Smart Groceries Demo")
            layout = QVBoxLayout()
            newOrderButton = QPushButton("New Order")
            orderInfoLabel = QLabel()
            orderInfoLabel.setTextFormat(Qt.RichText)
            chooseItemLayout = QHBoxLayout()
            verticalSpacer = QSpacerItem(20, 20)
            currentCartItems = QListWidget()

            layoutWidget = QWidget()
            referenceItemsLayout = QVBoxLayout(layoutWidget); referenceItemsLayout.setSpacing(0); referenceItemsLayout.setMargin(0)
            scroll = QScrollArea(dialog)
            scroll.setWidgetResizable(True)
            scroll.setMinimumHeight(150)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setWidget(layoutWidget)

            selectAisleLabel = QLabel("Aisle: ")
            selectProductLabel = QLabel("Product: ")
            selectAisleCombobox = QComboBox()
            selectproductCombobox = QComboBox()
            chooseItemLayout.addWidget(selectAisleLabel, 0,  Qt.AlignLeft)
            chooseItemLayout.addWidget(selectAisleCombobox, 0,  Qt.AlignLeft)
            chooseItemLayout.addWidget(selectProductLabel, 0,  Qt.AlignLeft)
            chooseItemLayout.addWidget(selectproductCombobox, 0,  Qt.AlignLeft)

            addToCartButton = QPushButton("Add to Cart")
            currentItemLabel = QLabel()
            currentItemLabel.setTextFormat(Qt.RichText)
            if self.itemInHand is None:
                currentItemLabel.setText(f"<b>Select an item from the list or from the recommendations</b>")
            addToCartButton.setDisabled(True)
            currentItemLayout = QHBoxLayout(); currentItemLayout.addWidget(currentItemLabel); currentItemLayout.addWidget(addToCartButton); 
            recommendationsLayout = QVBoxLayout();  recommendationsLayout.setSpacing(0); recommendationsLayout.setMargin(0)

            newOrderButton.clicked.connect(handleNewOrderButtonClicked)
            addToCartButton.clicked.connect(handleAddToCartButtonClicked)
            selectproductCombobox.currentIndexChanged.connect(itemChanged)
            selectAisleCombobox.currentIndexChanged.connect(aisleChanged)
            selectAisleCombobox.addItems(aisles.aisle.tolist())

            layout.addWidget(newOrderButton)
            layout.addSpacerItem(verticalSpacer)
            layout.addWidget(orderInfoLabel)
            layout.addWidget(scroll)
            layout.addSpacerItem(verticalSpacer)
            layout.addLayout(chooseItemLayout)
            layout.addSpacerItem(verticalSpacer)
            itemsInTheCartLabel = QLabel("<b>Items in the Cart<b>"); layout.addWidget(itemsInTheCartLabel); itemsInTheCartLabel.setTextFormat(Qt.RichText)
            layout.addWidget(currentCartItems)
            layout.addSpacerItem(verticalSpacer)
            currentRecommendationsLabel = QLabel("<b>Current Recommendations<b>"); layout.addWidget(currentRecommendationsLabel); currentRecommendationsLabel.setTextFormat(Qt.RichText); currentRecommendationsLabel.setVisible(False)
            layout.addLayout(recommendationsLayout)
            layout.addSpacerItem(verticalSpacer)
            layout.addLayout(currentItemLayout)
            runAutoButton = QPushButton("Run and Watch. TRUST ME, IT IS FUN!"); layout.addWidget(runAutoButton); runAutoButton.clicked.connect(handleRunAutomatically)
            dialog.setLayout(layout)
            handleNewOrderButtonClicked()
            dialog.exec_()
            return


    def reset(self):
        try:
            startBytes = bytes([STARTING_BYTE]*50)
            self.serialPort.write(startBytes)
            result = self._sendCommand("RESET", "None")
            if result is FAILURE_CODE:
                return ExecutionResult.FAILED
            else:
                return ExecutionResult.COMPLETED
        except SerialTimeoutException as e:
            self.logger.enableLogging()
            self.log("Please try again or reboot the RPi if the problem persists, Caught exception: {}".format(e), type="ERROR")
            return ExecutionResult.FAILED
        except Exception as e:
            self.logger.enableLogging()
            self.log("Caught exception: {}".format(e), type="ERROR")
            self.log(traceback.format_exc())
            print(traceback.format_stack())
            return ExecutionResult.FAILED

    def _executeLab(self, inputs, outputFolder, trueOutput= None, labCode= None, outputHeader= None, progressBar= None, plotter= None):
        if progressBar is not None:
            progressBarIncrements = 100/len(inputs.index)
            currentProgressBarValue = progressBar.value()
        
        outputFilePath = os.path.join(outputFolder, datetime.now().strftime("%d-%m_%H-%M-%S"))

        if trueOutput is None:
            outputDataFrame = copy.deepcopy(inputs)
        else:
            outputDataFrame = copy.deepcopy(pd.concat([inputs, trueOutput], axis=1))

        with open(outputFilePath+"_OutputsOnly.csv", 'a') as outFile:
            headers = []
            if outputHeader is not None:
                outFile.write(outputHeader+"\n")
                headers = outputHeader.split(",")
            for i in range(len(inputs.index)):
                inputStringParts = [str(n) for n in inputs.iloc[i].values.tolist()]
                inputString = ", ".join(inputStringParts)
                self.log("Now processing: {}".format(inputString), type="INFO")
                result = self._sendCommand("PROCESS", inputString)
                if result is FAILURE_CODE:
                    self.log("Error processing line number {}, possible serial communication issues".format(i+1), type="ERROR")
                    return FAILURE_CODE
                else:                
                    self.log("Output is: {}".format(result), type="SUCCESS")
                    outFile.write(result+"\n")
                if plotter is not None:
                    plotter.addNewData(inputs.iloc[i, 0], float(result.rstrip(' \t\r\n\0').split(',')[0]))
                if progressBar is not None:
                    currentProgressBarValue += progressBarIncrements
                    progressBar.setValue(currentProgressBarValue)
                # print(result)
                outputs = [float(i) for i in result.rstrip(' \t\r\n\0').split(',')]
                for index, output in enumerate(outputs):
                    if index < len(headers):
                        header = headers[index]
                    else:
                        header = f"Unknown_{index+1}"

                    outputDataFrame.loc[i, header] = output
            outputDataFrame.to_csv(outputFilePath+"_Full.csv", index=False)
            self.log(f"Outputs Saved in {outputFilePath+'_OutputsOnly.csv'}\nComplete data saved in {outputFilePath+'_Full.csv'}")

            # calculate accuracy
            if trueOutput is not None and labCode:
                try:
                    if labCode == "Lab1":
                        from sklearn.metrics import r2_score, mean_squared_error
                        r2Score = r2_score(outputDataFrame.iloc[:, -2], outputDataFrame.iloc[:, -1])
                        RMSE = mean_squared_error(outputDataFrame.iloc[:, -2], outputDataFrame.iloc[:, -1], squared=False)
                        self.log(f"Regression R2 Score Calculated is {r2Score :.3f} and RMSE is {RMSE :.3f}")
                    elif labCode == "Lab2":
                        from sklearn.metrics import accuracy_score, recall_score, f1_score
                        accuracyScore = accuracy_score(outputDataFrame.iloc[:, -2], outputDataFrame.iloc[:, -1])
                        recallScore = recall_score(outputDataFrame.iloc[:, -2], outputDataFrame.iloc[:, -1])
                        f1Score = f1_score(outputDataFrame.iloc[:, -2], outputDataFrame.iloc[:, -1])
                        self.log(f"Classification Metrics\n Accuracy: {accuracyScore*100 :.2f}%, Recall: {recallScore:.2f}, F1-Score: {f1Score:.2f}")
                except Exception as e:
                    self.log(f"Failed to calculate accuracy metrics because of exception: {e}", type="ERROR")

            return SUCCESS_CODE

    def _sendCommand(self, command, payload, timeout=SERIAL_COMMAND_TIMEOUT, max_retry=SERIAL_COMMAND_MAX_TRIALS):
        if not command in SERIAL_COMMANDS:
            print("The command provided {} is not a valid serial command".format(command))
            return FAILURE_CODE
        sendBuffer = bytearray()
        sendBuffer.append(STARTING_BYTE)
        sendString = command + ":" + payload
        sendBuffer.extend(sendString.encode("utf-8"))
        sendBuffer.append(0x00)
        newChecksum = Crc16()
        # print("Checksum Calc based on {}".format(sendBuffer[1:]))
        newChecksum.process(sendBuffer[1:])
        checksumBytes = newChecksum.finalbytes()
        sendBuffer.extend(checksumBytes)
        # print(len(sendBuffer))
        for attempt in range(max_retry):
            if attempt != 0:
                self.log(f"Attempt #{attempt+1} to send the command {command} with payload {payload}", type="DEBUG")
                QCoreApplication.processEvents()
            # t = time.time()
            try:
                self.serialPort.flushInput()
                self.serialPort.write(sendBuffer)
            except SerialTimeoutException:
                self.serialPort.flushOutput()
                continue
            self.serialTimeoutTimer.setInterval(timeout)
            self.serialTimeoutTimer.start()
            succeeded, string = self.getSerialAck()
            # print("The time spent from sending a command to receiving a reply (or timeouting) is ",time.time()-t)
            if succeeded:
                return string
            elif not succeeded and "EXCEPTION" in string:
                break 
        return FAILURE_CODE

    def _sendSaveModelCommand(self, model):
        with open(model, 'rb') as modelFile:
            fileToBeSent = modelFile.read()
        fileToBeSent = zlib.compress(fileToBeSent, level=9)
        fileToBeSentStr = " ".join(map(str,fileToBeSent))
        self.log(f"Estimated time for model to be sent is {int(len(fileToBeSentStr)/2000)} seconds", type="INFO")
        return self._sendCommand("SAVE_MODEL", fileToBeSentStr, timeout=SAVE_MODEL_COMMAND_TIMEOUT)
        
    def getSerialAck(self):
        string = ""
        succeeded = False

        self.serialState = SerialState.WaitingToStart
        currentSerialString = ""
        currentCheckSum = bytearray(2)

        while(self.serialTimeoutTimer.remainingTime()>0):
            QCoreApplication.processEvents()
            self.processCheckStopRequest()
            if self.serialState == SerialState.WaitingToStart:
                newByte = self.serialPort.read()
                if len(newByte) == 1:
                    if newByte[0] == STARTING_BYTE:
                        self.serialState = SerialState.WaitingForString
            
            if self.serialState == SerialState.WaitingForString:
                newBytes = self.serialPort.read_until(b'\0')
                if len(newBytes) >= 1:
                    for i in range (len(newBytes)):
                        if newBytes[i] == STARTING_BYTE:
                            pass
                        else:
                            currentSerialString = currentSerialString + newBytes[i:].decode("utf-8")
                            if newBytes[-1] == 0x00:
                                self.serialState = SerialState.WaitingForChecksum1
                            break
            
            if self.serialState == SerialState.WaitingForChecksum1:
                newByte = self.serialPort.read()
                if len(newByte) == 1:
                    currentCheckSum[0] = newByte[0]
                    self.serialState = SerialState.WaitingForChecksum2
                
            if self.serialState == SerialState.WaitingForChecksum2:
                newByte = self.serialPort.read()
                if len(newByte) == 1:
                    currentCheckSum[1] = newByte[0]
                    self.serialState = SerialState.CommandDone

            if self.serialState == SerialState.CommandDone:
                # check the message integrity
                receivedCommandCrc = Crc16()
                receivedCommandCrc.process(currentSerialString.encode('utf-8'))
                receivedCommandCrcBytes = receivedCommandCrc.finalbytes()
                
                # print("Checksum Calc based on {}".format(currentSerialString.encode('utf-8')))
                # print("Checksum Received: {}, Calculated: {}".format(currentCheckSum, receivedCommandCrcBytes))
                if receivedCommandCrcBytes == currentCheckSum:
                    succeeded = True
                    string = currentSerialString.split(":")[1].rstrip(' \t\r\n\0')
                    if string == "None":
                        string = ""
                else:
                    self.log("Acknowledgment Failed, received: {}".format(currentSerialString.rstrip("\t\r\n\0")), type="ERROR")
                    string = currentSerialString
                break

        return succeeded, string

    def processCheckStopRequest(self):
        if not self._stopRequested:
            return
        else:
            self._stopRequested = False
            raise StopProcessingRequested

    def requestStop(self):
        self._stopRequested = True

    @property
    def execState(self):
        return self._execState

    @execState.setter
    def execState(self, newVal):
        # print("Switched to Exec State: {}".format(newVal))
        self._execState = newVal

    @property
    def serialState(self):
        return self._serialState

    @serialState.setter
    def serialState(self, newVal):
        # print("Switched to Serial State: {}".format(newVal))
        self._serialState = newVal

