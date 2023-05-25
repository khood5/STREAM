import librosa
import os
import csv
import pandas as pd
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from pathlib import Path 

class Settings:
    _INPUT_DIRECTORY    = None  # directory with audio files to convert
    _OUTPUT_DIRECTORY   = None  # directory to place melspectrogram csv files
    _INPUT_INDEX        = None  # index file of the orignal mp3 audio files, should be in a csv format with files as the first column and label as the second column.
    _SILENT             = None  # if true program should not output anything  
    
    def __new__(cls):
            if not hasattr(cls, 'instance'):
                    cls.instance = super(Settings, cls).__new__(cls)
            return cls.instance 
    
    def save_settings(self):
                dt_string = datetime.now().strftime("%d.%m.%Y_%H-%M-%S-%f")
                try:
                    f = open(f"./preprocess_logs/preprocess_{dt_string}_saved_args.csv", "w+")
                except FileNotFoundError as e: 
                    print("!!! Missing 'preprocess_logs' directory\n writing logs to this directory")
                    f = open(f"./preprocess_{dt_string}_saved_args.csv", "w+")
                f.write(f"run at {dt_string}")
                f.close()
                
if __name__ == "__main__":
    INDEX_FILE_NAME = "index.csv"
    
    parser = argparse.ArgumentParser(description='''Process audio files (mp3) and creates their melspectrogram.\n\
        Melspectrogram files will be stored with the same file name as the audio file they are from. They are written in CSV format.\n\
        An index file of the orignal mp3 audio files will be needed. This should be in a csv format with files as the first column and\n\
        label as the second column.\n\
        An index file will be placed in the output directory containing the file to label mapping.\n''', 
        formatter_class=RawTextHelpFormatter )
    parser.add_argument('input_directory', help='Directory with input files\n',
                        type=str)
    parser.add_argument('index_file', help='Index file for the input mp3 files with labels (should be a csv)\n',
                        type=str)
    parser.add_argument('output_directory', help='Directory to place output files and new index file\n',
                    type=str)
    parser.add_argument('-s','--silent', help="switches to silent mode i.e. hides output\n", 
                        action=argparse.BooleanOptionalAction)
    # parser.add_argument('-l','--length', help='''number of seconds to use for training (events after this time will not be included in the spike trains)\n\
    #                     DEFAULT: {}'''.format(DEFAULT_LENGTH), 
    #                     type=float,
    #                     default=DEFAULT_LENGTH)
    
    args = parser.parse_args()
    settings = Settings()
    settings._INPUT_DIRECTORY = args.input_directory
    settings._INPUT_INDEX = args.index_file
    settings._OUTPUT_DIRECTORY = args.output_directory
    settings._SILENT = args.silent

    if settings._SILENT != True:
        print(f"reading files from  :{Path(settings._INPUT_DIRECTORY)}")
        print(f"creating files in   :{Path(settings._OUTPUT_DIRECTORY)}")
        
    if os.path.isdir(settings._INPUT_DIRECTORY):
        inputFiles = os.listdir(settings._INPUT_DIRECTORY)
    elif os.path.isfile(settings._INPUT_DIRECTORY):
        inputFiles = [os.path.basename(settings._INPUT_DIRECTORY)]
        settings._INPUT_DIRECTORY = Path(settings._INPUT_DIRECTORY).parent.absolute()
    else:
        print(f"ERROR: {settings._INPUT_DIRECTORY} is not a file or directory")
    
    indexes = []
    audioIndex = np.array(pd.read_csv(settings._INPUT_INDEX,header=None))
    inputFiles = [
        f"{os.path.join(audioIndex[i][0])}" for i in range(len(audioIndex)) 
    ]
    labels = list(audioIndex[:, 1])
    total = len(labels)
    currentProgress = 0
    for inFile in inputFiles:
        if settings._SILENT != True: print(f"preprocessing: {'{0:.0f}%'.format(currentProgress/total * 100)}... ",end="\r")
        supported_file_types = ["mp3"]
        if inFile.split(".")[-1].lower() not in supported_file_types:
            continue # skip none mp3 files
        y, sr = librosa.load(os.path.join(settings._INPUT_DIRECTORY,inFile))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        outFileName = f"{inFile}.csv"
        outFile = os.path.join(f"{settings._OUTPUT_DIRECTORY}", outFileName)
        np.savetxt(outFile, S, delimiter=",")
        indexes.append([outFileName,labels.pop(0)])
        currentProgress = currentProgress + 1
    outputIndexFile = os.path.join(f"{settings._OUTPUT_DIRECTORY}",INDEX_FILE_NAME)
    with open(outputIndexFile, 'a+', newline='') as outFile:
        write = csv.writer(outFile)
        write.writerows(indexes)
    settings.save_settings()