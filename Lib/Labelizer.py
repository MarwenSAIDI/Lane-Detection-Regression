## Transductive learning (label propagation)
from turtle import ycor
import pickle as pkl
from sklearn.semi_supervised import LabelPropagation
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from Lib.log import logger
import warnings
warnings.filterwarnings('ignore')

class labelizer:
    def __init__(self, path="", model=False):
        self.createFolders(path)
        if model == False:
            self.model = LabelPropagation(n_neighbors=3, kernel="knn", n_jobs=3)
        else:
            
            self.model = pkl.load(open(self.path + "\\Model_label_propagation.pkl", "rb"))
            
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
                
                os.mkdir(path + "\\MLModels")
                os.mkdir(path + "\\MLModels\\Label_model")
                
                self.path = path + "\\MLModels\\Label_model"
            except Exception as e:
                logger.logError(e)
        else:
            ## We create in the current directory
            try:
                os.mkdir(os.getcwd() + "\\MLModels")
                os.mkdir(os.getcwd() + "\\MLModels\\Label_model")
                self.path = os.getcwd() + "\\MLModels\\Label_model"
            except Exception as e:
                logger.logError(e)

    def trainModel(self, X, y):

        self.model.fit(X, y)
        return f"Training score: {self.model.score(X,y)}"

    def labeling(self, df, col_name="label"):
        
        cond = (df.columns == 'frame_id') | (df.columns == 'mask_id') | (df.columns == 'row_number') | (df.columns == 'pixel_number')
        X = df.iloc[:, cond]

        y_pred = self.model.predict(X)
        df[col_name] = y_pred
        return df

    def exportMeanDF(n, inPath, outPath="", outName="new_data", ouType="csv", s=0):
        """
            This function iterates n dataframes
            and, for each "row_number" for each "label", computes the mean
            of "pixel_number" and exports the new dataframes in the 
            specified path by the specified commen name and an index for each one.
            
            Arguments:
                n: Number of files to iterate from (csv or xlsx).
                inPath: The path of where the files are located.
                outPath: The path where we will export the new files.
                    If the path is empty (default), it will take the location 
                    of where the original files where and puts them in there.
                outName: Commen name of the new data frames. By default, 
                    outName="new_data".
                outType: Type of the file (csv, xlsx). Currently it supports CSV and XLSX.
                    By default, ouType="csv".
                s: the dataframe id to start iterating from. Default s=0.
                
        """
        ## Check if the variables are set correctly
        try:
            int(n) 
            try:
                int(inPath)
                return "ValueError: expected type str in 'inPath', instead it got number or str(int)."
            except:
                ##Check if inPath and outPath are valid
                try:
                    assert os.path.dirname(inPath) != ""
                    assert len(os.listdir(inPath)) != 0
                    ## We start getting the dataframes
                    for i in tqdm(range(s,n)):
                        df_glob = {}
                        df = pd.read_csv(inPath+"/data"+str(i)+".csv")
                        labels = df['label'].unique()
                        ##Iterating and processing by every label
                        for label in labels:
                            df_label = (df.loc[(df.label == label)].groupby(["frame_id","mask_id","row_number"]).mean()).reset_index(level=["row_number","frame_id","mask_id"])
                            if label == labels[0]:
                                df_glob = df_label
                            else:
                                df_glob = pd.concat([df_glob, df_label])
                        ##Reset the indexes
                        df_glob.reset_index(drop=True, inplace=True)
                        ##Generate the dataframe in the outpath
                        if ouType.lower() == "csv":
                            df_glob.to_csv(os.path.join(outPath, outName +str(i)+ '.csv'), index=False)
                        elif ouType.lower() == "xlsx":
                            df_glob.to_excel(os.path.join(outPath, outName + '.xlsx'), index=False)
                except Exception as e:
                    return (f"{type(e).__name__}: {e.args}")
        except Exception as e:
            return (f"{type(e).__name__}: {e}")

    def linkDataFrames(self, df_prev, df_next):
        """
        This function takes two dataframes the current and previous to it
        and links them.

        Args:
            df_prev (pd.DataFrame): The previous dataframe.
            df_next (pd.DataFrame): The next dataframe.

        Returns:
            pd.DatFrame: Teh new linked dataframe.
        """
        df_next = df_next.join(df_prev.iloc[:,-3:].set_index('row_number'), on='row_number', rsuffix="_prev")
        df_next = df_next[df_next['label'] == df_next['label_prev']]
        return df_next

    def exportModel(self):
        pkl.dump(self.model, open(self.path + "\\Model_label_propagation.pkl", "wb"))
        

    
