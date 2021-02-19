import serial
import enum
from crccheck.crc import Crc16
import os
import pickle
import sklearn
import numpy as np
import zlib
from subprocess import check_output, CalledProcessError


STARTING_BYTE = 0x01

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

class LabCode(enum.Enum):
    LabTest = 0
    Lab1 = 1
    Lab2 = 2
    Lab3 = 3
    Lab4 = 4

class PiExecuter():
    def __init__(self, serialPort):
        self.port = serialPort
        self.execState = ExecState.Connected
        self.serialState = SerialState.WaitingToStart

        self._currentLab = ""
        self._currentModelPath = ""

        self._currentSerialString = ""
        self._currentCheckSum = bytearray(2)

    def readSerial(self):
        if self.serialState == SerialState.WaitingToStart:
            newByte = self.port.read()
            if len(newByte) == 1:
                if newByte[0] == STARTING_BYTE:
                    self.serialState = SerialState.WaitingForString
        
        if self.serialState == SerialState.WaitingForString:
            newBytes = self.port.read_until(b'\0')
            if len(newBytes) >= 1:
                for i in range (len(newBytes)):
                    if newBytes[i] == STARTING_BYTE:
                        pass
                    else:
                        self._currentSerialString = self._currentSerialString + newBytes[i:].decode("utf-8")
                        if newBytes[-1] == 0x00:
                            self.serialState = SerialState.WaitingForChecksum1
                        break
            print(len(self._currentSerialString))
            
        if self.serialState == SerialState.WaitingForChecksum1:
            newByte = self.port.read()
            if len(newByte) == 1:
                self._currentCheckSum[0] = newByte[0]
                self.serialState = SerialState.WaitingForChecksum2
            
        if self.serialState == SerialState.WaitingForChecksum2:
            newByte = self.port.read()
            if len(newByte) == 1:
                self._currentCheckSum[1] = newByte[0]
                self.serialState = SerialState.CommandDone

        if self.serialState == SerialState.CommandDone:
            # check the command integrity
            receivedCommandCrc = Crc16()
            receivedCommandCrc.process(self._currentSerialString.encode('utf-8'))
            receivedCommandCrcBytes = receivedCommandCrc.finalbytes()
            print("Checksum Calc based on {}".format(self._currentSerialString.encode('utf-8')))
            print("Checksum Received: {}, Calculated: {}".format(self._currentCheckSum, receivedCommandCrcBytes))
            if receivedCommandCrcBytes == self._currentCheckSum:
                self.processSerialCommand(self._currentSerialString)

            self._currentSerialString = ""
            self._currentCheckSum[0] = 0x00
            self._currentCheckSum[1] = 0x00

            self.serialState = SerialState.WaitingToStart

    def processSerialCommand(self, commandStr):
        ackPayload = "None"
        
        (command, payload) = commandStr.rstrip('\t\r\n\0').split(':')
        print("Received command: {}, with payload: {} in State {}".format(command, payload, self.execState))
        if command == "RESET":
            self.execState = ExecState.Connected
            self._currentSerialString = ""
            self._currentCheckSum[0] = 0x00
            self._currentCheckSum[1] = 0x00
            self.serialState = SerialState.WaitingToStart

        elif command == "GET_IP":
            ackPayload = check_output(['hostname','-I']).decode('utf-8').split(" ")[0]
            
        elif command == "UPDATE_SCRIPT":
            try:
                ackPayload = check_output(['git','pull'], cwd='/home/pi/SFU_ML').decode('utf-8')
            except CalledProcessError as e:
                ackPayload = "FAILED: " + e.output()

        elif self.execState == ExecState.NotConnected:
            raise Exception("Wrong Exec State reached somehow: {}".format(self.execState))

        elif self.execState == ExecState.Connected:
            if command != "SELECT_LAB":
                raise Exception("You need to select the lab in the current state of: {}".format(self.execState))
            else:
                self._currentLab = LabCode[payload]
                self.execState = ExecState.LabSelected

        elif self.execState == ExecState.LabSelected:
            if not command in ["SAVE_MODEL", "LOAD_MODEL"]:
                raise Exception("You need to send or load a default model in the current state of: {}".format(self.execState))
            else:
                if self._currentLab != LabCode.LabTest:
                    self._currentModelPath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')
                    if command == "SAVE_MODEL":
                        savedModelStrings = payload.split(" ")
                        savedModel = bytearray([int(s) for s in savedModelStrings])
                        savedModel = zlib.decompress(savedModel)
                        # print(savedModel)
                        self._currentModelPath = os.path.join(self._currentModelPath, "lastModelSaved")
                        with open(self._currentModelPath, "wb") as modelFile:
                            modelFile.write(bytearray(savedModel))
                    elif command == "LOAD_MODEL":
                        self._currentModelPath = os.path.join(self._currentModelPath, payload)
                    
                    try:
                        self._loadedModel = pickle.load(open(self._currentModelPath, 'rb'))
                        print("Loaded the Model File successfully")
                    except Exception as e:
                        raise Exception("Problem opening the current model with path {} because of exception: {}".format(str(self._currentModelPath), str(e)))

            self.execState = ExecState.ModelLoaded
            self.execState = ExecState.Processing

        elif self.execState == ExecState.ModelLoaded:
            #for any preprocessing required
            pass #NotImplemented

        elif self.execState == ExecState.Processing:
            if not command in ["PROCESS", "PROCESSING_DONE"]:
                raise Exception("You need to send a PROCESS or PROCESSING_DONE command in the current state of: {}".format(self.execState))
            else:
                if command == "PROCESS":
                    if self._currentLab == LabCode.LabTest:
                        ackPayload = self.processLabTest(payload)
                    elif self._currentLab == LabCode.Lab1:
                        ackPayload = self.processLab1(payload)
                    else:
                        raise Exception("The lab code provided was not a valid lab")
                elif command == "PROCESSING_DONE":
                    self.execState = ExecState.Connected

        self.sendSerialAck(ackPayload)

    def processLabTest(self, payload):
        inputs = [float(i) for i in payload.split(',')]
        outputs = [0, 0]
        outputs[0] = sum(inputs)
        outputs[1] = 1
        for input in inputs:
            outputs[1] *= input
        strOutputs = [str(o) for o in outputs]
        outputPayload = ', '.join(strOutputs)
        print("The Acknowledgment Payload is:"+outputPayload)
        return outputPayload

    def processLab1(self, payload):
        # print("Went into process lab 1")
        input_list = [float(i) for i in payload.split(',')]
        y_pred = self._loadedModel.predict(np.array(input_list).reshape(1, -1)).astype('float32')
        outputPayload = ', '.join([str(i) for i in list(y_pred.flatten())])
        print("The Acknowledgment Payload is:"+outputPayload)
        if outputPayload == "":
            raise Exception("Problem with getting a prediction for the input data, \
please check the provided model and restart both the application and Raspberry Pi")
        return outputPayload

    def sendSerialAck(self, result=None):
        outBuffer = bytearray()
        outBuffer.append(STARTING_BYTE)
        if result == None:
            result = "None" 
        ackBytes = ("ACK:"+ result).encode("utf-8")
        outBuffer.extend(ackBytes)
        outBuffer.append(0x00)
        newChecksum = Crc16()
        newChecksum.process(outBuffer[1:])
        checksumBytes = newChecksum.finalbytes()
        outBuffer.extend(checksumBytes)
        self.port.write(outBuffer)
        # print("sent Ack: {}".format(outBuffer))


    @property
    def execState(self):
        return self._execState

    @execState.setter
    def execState(self, newVal):
        print("Switched to Exec State: {}".format(newVal))
        self._execState = newVal

    @property
    def serialState(self):
        return self._serialState

    @serialState.setter
    def serialState(self, newVal):
        # print("Switched to Serial State: {}".format(newVal))
        self._serialState = newVal