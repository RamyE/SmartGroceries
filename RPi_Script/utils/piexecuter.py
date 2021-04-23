import serial
import enum
from crccheck.crc import Crc16
import os
import pickle
import sklearn
import numpy as np
import zlib
from subprocess import check_output, CalledProcessError
import pandas as pd
import ast
import random
import surprise

STARTING_BYTE = 0x01

# rules_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects", "SmartGroceries", "data", 'rules_df.csv'))
# rules_df.Antecedent = rules_df.Antecedent.apply(lambda x: ast.literal_eval(x))
# rules_df.Consequents = rules_df.Consequents.apply(lambda x: ast.literal_eval(x))
pickls_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects", "SmartGroceries", "pickles")
rules_df_categorized = pickle.load(open(os.path.join(pickls_path, 'rules_df_categorized.pkl'), 'rb'))
product_to_category = pickle.load(open(os.path.join(pickls_path, 'product_to_category.pkl'), 'rb'))
collab_filtering_predictions = pickle.load(open(os.path.join(pickls_path, 'collab_filtering_predictions.pkl'), 'rb'))
users_categorized_top_items = pickle.load(open(os.path.join(pickls_path, 'users_categorized_top_items.pkl'), 'rb'))
top_100_items = [24852, 13176, 21137, 21903, 47209, 47766, 47626, 16797, 26209, 27845, 27966, 22935, 24964, 45007, 39275, 49683, 28204, 5876, 8277, 40706, 4920, 30391, 45066, 42265, 49235, 44632, 19057, 4605, 37646, 21616, 17794, 27104, 30489, 31717, 27086, 44359, 28985, 46979, 8518, 41950, 26604, 5077, 34126, 22035, 39877, 35951, 43352, 10749, 19660, 9076, 21938, 43961, 24184, 34969, 46667, 48679, 25890, 31506, 12341, 39928, 24838, 5450, 22825, 5785, 35221, 28842, 33731, 27521, 44142, 33198, 8174, 20114, 8424, 27344, 11520, 29487, 18465, 28199, 15290, 46906, 9839, 27156, 3957, 43122, 23909, 34358, 4799, 9387, 16759, 196, 42736, 38689, 4210, 41787, 41220, 47144, 7781, 33000, 20995, 21709]

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
        elif command == "PROCESS_PROJECT_GROUP_2":
            ackPayload = self.processProjectGroup2(payload)
        
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
                        ackPayload = self.processLab1andLab2(payload)
                    elif self._currentLab == LabCode.Lab2:
                        ackPayload = self.processLab1andLab2(payload)
                    else:
                        raise Exception("The lab selected is not a valid lab. This is likely a software implementation issue")
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

    def processLab1andLab2(self, payload):
        # print("Went into process lab 1 or Lab 2")
        input_list = [float(i) for i in payload.split(',')]
        y_pred = self._loadedModel.predict(np.array(input_list).reshape(1, -1)).astype('float32')
        outputPayload = ', '.join([str(i) for i in list(y_pred.flatten())])
        print("The Acknowledgment Payload is:"+outputPayload)
        if outputPayload == "":
            raise Exception("Problem with getting a prediction for the input data, \
please check the provided model and restart both the application and Raspberry Pi")
        return outputPayload

    def processProjectGroup2(self, payload):
        user_id = int(payload.split(";")[0])
        input_items = [ int(id.strip()) for id in payload.split(";")[1].split(",")]
        try:
            try_to_execlude = [ int(id.strip()) for id in payload.split(";")[2].split(",")]
        except:
            try_to_execlude = []

        def association_rules_algo(current_items, k=5, rules_df=None, no_popular_items=False):
            # print(current_items)
            if rules_df is None:
                raise RuntimeError("Rules Data Frame needs to be provided (acts as a trained model)")
            possible_predictions_df = rules_df[rules_df['Antecedent'].map(lambda x: all(id in current_items for id in x))]
            possible_predictions_df = possible_predictions_df.sort_values('Confidence', ascending=False)
            possible_predictions = []
            for row in possible_predictions_df['Consequents']:
                [possible_predictions.append(x) for x in row if not (x in possible_predictions or x in current_items)]
            print(possible_predictions)
            if no_popular_items:
                possible_predictions = [x for x in possible_predictions if (x not in top_100_items)]
            print(possible_predictions)
            filtered_predictions = possible_predictions.copy()
            possible_predictions.reverse()
            for item in possible_predictions:
                if (item in try_to_execlude) and len(filtered_predictions) > k:
                    filtered_predictions.remove(item)
            print(filtered_predictions)
            return filtered_predictions[:min(k, len(filtered_predictions))]

        def association_rules_categories_algo(current_items, k=5, rules_df=None, user_id = None, no_popular_items=False):
            if rules_df is None:
                raise RuntimeError("Rules Data Frame needs to be provided (acts as a trained model)")
            if user_id is None:
                raise RuntimeError("User ID needs to be provided for this algorithm to work")
            current_categories = set([product_to_category[id] for id in current_items if id in product_to_category.keys()])
            possible_predictions_df = rules_df[rules_df['Antecedent'].map(lambda x: all(category in current_categories for category in x))]
            possible_predictions_df = possible_predictions_df.sort_values('Confidence', ascending=False)
            possible_predictions = []
            for row in possible_predictions_df['Consequents']:
                [possible_predictions.append(x) for x in row if not (x in possible_predictions or x in current_categories)]
            final_predictions = []
            for category in possible_predictions:
                if category in users_categorized_top_items[user_id].keys():
                    for item, count in users_categorized_top_items[user_id][category]:
                        if not no_popular_items or item not in top_100_items:
                            # if item not in try_to_execlude :
                            final_predictions.append(item)
                            break
                    # if len(final_predictions) == k:
                    #     break
            filtered_predictions = final_predictions.copy()
            final_predictions.reverse()
            for item in final_predictions:
                if (item in try_to_execlude) and len(filtered_predictions) > k:
                    filtered_predictions.remove(item)
            return filtered_predictions[:min(k, len(filtered_predictions))]

        def get_predictions_for_user(predictions, uid):
            top_items = []
            for uid, iid, _, est, _ in predictions:
                if uid == user_id:
                    top_items.append((iid, est))
            top_items.sort(key=lambda x: x[1], reverse=True)
            return top_items

        def collab_filtering_algo(cart_current_items, cf_predictions_tuple=None, k=5, user_id=None):
            if user_id is None:
                raise RuntimeError("User ID needs to be provided for this algorithm to work")
            if cf_predictions_tuple is None:
                raise RuntimeError("Provide the saved predictions in the Surprise Predictions format")
            possible_recommendations = [iid for (iid, _) in get_predictions_for_user(cf_predictions_tuple, user_id) if iid not in (cart_current_items)] #try_to_execlude
            return possible_recommendations[:min(k, len(possible_recommendations))]

        def hybrid_algo(cart_current_items, k=5, cf_predictions_tuple=None, user_id=None, rules_df=None, no_popular_items=True):
            if user_id is None or cf_predictions_tuple is None or rules_df is None:
                raise RuntimeError("Please provide all needed keyword arguments user_id, rules_df, and cf_predictions_tuple")
            s = random.randrange(1, k)
            assoc_rules_predictions = association_rules_categories_algo(cart_current_items, k=k-s, rules_df=rules_df, user_id=user_id, no_popular_items=True)
            m = k-len(assoc_rules_predictions)
            collab_filtering_predictions = collab_filtering_algo(cart_current_items, cf_predictions_tuple=cf_predictions_tuple, k=m,user_id=user_id)
            print("assoc", len(assoc_rules_predictions), "collab", len(collab_filtering_predictions))
            return assoc_rules_predictions+collab_filtering_predictions
    
        # recommendations = hybrid_algo(input_items, user_id=user_id, k=5, rules_df=rules_df_categorized, cf_predictions_tuple=collab_filtering_predictions)
        recommendations = association_rules_categories_algo(input_items, user_id=user_id, k=5, rules_df=rules_df_categorized, no_popular_items=True)
        print(recommendations)

        return ",".join([str(x) for x in recommendations])

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