# Face Generator Using DCGAN
The current project aims to develop DCGAN model to make create fake faces images.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
The following package might need to be installed
os.system("sudo pip install --user googledrivedownloader")

### How to run the code.

1. load_data.py
    The process begins by downloading the dataset using the file 'load_data'. this file will download the dataset
    from the internet and unzip it.

2. main.py
    This will run the model, and save the images created as a pickle file.
    
3. plot_images.py
    This file will plot the fake images created by the model
