"""
MACHINE LEARNING II - FINAL PROJECT
TOPIC: FACE GENERATOR USING DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS (DCGAN)
AUTHORS:
    Aira Domingo
    Renzo Castagnino
DATE: April 2020
"""
# code lines: 6

#%% ------------------------------------------- IMPORT PACKAGES --------------------------------------------------------
# Aira
import os
# os.system("sudo pip install --user googledrivedownloader")
from google_drive_downloader import GoogleDriveDownloader as gdd

#%% --------------------------------------------- DOWNLOAD THE DATA ----------------------------------------------------

# ID FOR DATASET'S SHARABLE LINK
id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'

# DESTINATION PATH FOR DATA
dest_path = './train/img_align_celeba.zip'

# DOWNLOAD DATASET TO PATH & UNZIP
gdd.download_file_from_google_drive(file_id=id, dest_path=dest_path, unzip=True)
