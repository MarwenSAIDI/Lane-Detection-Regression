import datetime
import os

def logError(e):
    """
    This function gets the error message and outputs it into
    a log file in "logs/file.log"

    Args:
        e: Error message or the Exception.
    """
    with open(os.getcwd()+"/logs/file.log","a") as logFile:
        msg = f"{datetime.datetime.now()} - {type(e).__name__}:  {e.args}\n"
        logFile.write(msg)