import os
import matplotlib.pyplot as plt
from Lib.log import logger
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class regressor:

    def __init__(self, path=""):
        self.createFolders(path)

    def createFolders(self,path=""):
        """
        This function creates the MLModels directory for the models after
        checking if the path exists. 
        If no path was set, then it assumes that the path is the same as 
        the Labelizer file.

        Args:
            path (str, optional): It takes the path where you want
            to create the models base and store the models in. Defaults to "".
        """
        if path != "":
            ## Ceck if that directory exists
            try:
                ## If it does the we create the Models Base directory
                ## And inside we add the Label model directorie
                
                os.mkdir(path + "\\MLModels\\Regression_model")
                
                self.path = path + "\\MLModels\\Regression_model"
            except Exception as e:
                logger.logError(e)
        else:
            ## We create in the current directory
            try:
                os.mkdir(os.getcwd() + "\\MLModels\\Regression_model")
                self.path = os.getcwd() + "\\MLModels\\Regression_model"
            except Exception as e:
                logger.logError(e)

    def buildModel(self, X_left, X_right):
        self.model_left = Sequential([
            Dense(2, input_dim=X_left.shape[1], activation="relu"),
            Dense(1,activation="relu")
        ])

        self.model_right = Sequential([
            Dense(2, input_dim=X_right.shape[1], activation="relu"),
            Dense(1,activation="relu")
        ])

        self.model_left.compile(optimizer=Adam(learning_rate=.1), loss="mean_absolute_error")
        self.model_right.compile(optimizer=Adam(learning_rate=.001), loss="mean_absolute_error")


    def evaluateLeft(self, y_pred, y_test, X_train, y_train, X, y):
        print(f"R squarred : {r2_score(y_test, y_pred)}")
        print(f"Adjusted R squarred : {1 - (1-r2_score(y_test, y_pred)) * (len(y)-1)/(len(y)-X.shape[1]-1)}")
        print(f"Mean squarred error : {mean_squared_error(y_test, y_pred)}")
        print(f"Mean Absolute Error : {mean_absolute_error(y_test, y_pred)}")
        print(f"Training score : {r2_score(y_train, self.model_left.predict(X_train))}")


    def evaluateRight(self, y_pred, y_test, X_train, y_train, X, y):
        print(f"R squarred : {r2_score(y_test, y_pred)}")
        print(f"Adjusted R squarred : {1 - (1-r2_score(y_test, y_pred)) * (len(y)-1)/(len(y)-X.shape[1]-1)}")
        print(f"Mean squarred error : {mean_squared_error(y_test, y_pred)}")
        print(f"Mean Absolute Error : {mean_absolute_error(y_test, y_pred)}")
        print(f"Training score : {r2_score(y_train, self.model_right.predict(X_train))}")

    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error [Pixel_number]')
        plt.legend()
        plt.grid(True)

    def saveModels(self):
        self.model_left.save(self.path + "\\Left_Line_Model.h5")
        self.model_right.save(self.path + "\\Right_Line_Model.h5")