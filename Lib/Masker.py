import time
import os
from Lib.log import logger
import numpy as np
import cv2
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class masker:
    def __init__(self, path=""):
        self.createFolders(path)


    def createFolders(self, path=""):
        """
        This function ccreates the learning base direstory
        plus the frames and masks directories inside after
        checking if the path exists and if the learning base folders exists. 
        If no path was set, then it assumes that the path is the same as 
        the Masker file.

        Args:
            path (str, optional): It takes the path where you want
            to create the learning base and stor the data in. Defaults to "".
        """
        if path != "":
            ## Ceck if that directory exists
            try:
                ## If it does the we create the Learning Base directory
                ## And inside we add the frames and masks directories
                os.mkdir(path + "\\Learning_Base")
                os.mkdir(path + "\\Learning_Base\\frames")
                os.mkdir(path + "\\Learning_Base\\masks")
                os.mkdir(path + "\\Learning_Base\\dataframes")
            except Exception as e:
                logger.logError(e)
        else:
            ## We create in the current directory
            try:
                os.mkdir(os.getcwd() + "\\Learning_Base")
                os.mkdir(os.getcwd() + "\\Learning_Base\\frames")
                os.mkdir(os.getcwd() + "\\Learning_Base\\masks")
                os.mkdir(os.getcwd() + "\\Learning_Base\\dataframes")
            except Exception as e:
                logger.logError(e)
        

        def unwarp(self, img, ptA=[253, 233], ptB=[379, 233], ptC=[513, 335], ptD=[83, 335], warping=True):
            """
            This function takes a frame of the road, takes a segment that is
            defined by ptA, ptB, ptC, ptD. And based on that segment it applies
            warpping and returns a "500x500" warped image of the segment.

            Args:
                - img (numpy.Array): The frame of the video.

                - ptA (list, optional): The coordinates of the 
                top left corner of the segment. Defaults to [253, 233].

                - ptB (list, optional): The coordinates of the 
                top right corner of the segment. Defaults to [379, 233].

                - ptC (list, optional): The coordinates of the 
                bottom right corner of the segment. Defaults to [513, 335].

                - ptD (list, optional): The coordinates of the 
                bottom left corner of the segment. Defaults to [83, 335].

                - warping (bool, optional): If warping is set to True then 
                we will perform the warping if not then it's the unwarping. Defaults to True.

            Returns:
                numpy.Array: It returns the warped segment of the road.
            """
            try:
                src = np.float32([
                    ptA,
                    ptB,
                    ptD,
                    ptC
                ])

                dst = np.float32([
                    [0,0],
                    [500, 0],
                    [0, 500],
                    [500, 500]
                ])

                matrix = cv2.getPerspectiveTransform(src, dst)
                matrix_inv = cv2.getPerspectiveTransform(dst, src)

                if warping == True :
                    warp = cv2.warpPerspective(img, matrix, (500, 500))
                else :
                    warp = cv2.warpPerspective(img, matrix_inv, (ptC[0], ptC[1]))
                return warp
            except Exception as e:
                logger.logError(e)
            

        def masker(self, warp, threshold = (1, 225)):
            """
            This function takes as parameters the warped segment of the road,
            apply a color mask for the white pixels via a threshold and return a
            black and white mask.

            Args:
                - warp (np.Array): The warped segment of the road.
                
                - threshold (tuple, optional): Defaults to (1, 225).

            Returns:
                np.Array: _description_
            """
            try:
                ## color mask
                output = np.zeros_like(warp[:,:,0])
                output[(warp[:,:,0] >= threshold[0]) & (warp[:,:,0] <= threshold[1])] = 255
                    
                ## invert mask
                inv_warp = cv2.bitwise_not(output)
                return inv_warp
            except Exception as e:
                logger.logError(e)

        def getWhitPixels(self, mask, frame_id):
            """
            This function returns a dataframe containing the id of the frame
            and the mask plus the row and the position of the white pixels
            (row and index in the row).

            Args:
                - mask (np.Array): The mask of the seguemnt.
                - frame_id (int): The id of the frame.

            Returns:
                pd.DataFrame: The dataframe coeesponding to the mask.
                bool: If True the the mask containes white pixels.
                If not, it returns False.
            """
            
            df = {
                "frame_id":[],
                "mask_id":[],
                "row_number":[],
                "pixel_number":[]
            }
            try:
                nonzero = np.nonzero(mask)
                
                df["row_number"] = nonzero[0]
                df["pixel_number"] = nonzero[1]
                
                if len(nonzero[0]) > 0 :
                    df["frame_id"].append(frame_id)
                    df["mask_id"].append(frame_id)
                    df["mask_id"] = df["mask_id"] * len(nonzero[0])
                    df["frame_id"] = df["frame_id"] * len(nonzero[0])
                    
                    return pd.DataFrame(df), True
                else:
                    return None, False

            except Exception as e:
                logger.logError(e)
            
            

