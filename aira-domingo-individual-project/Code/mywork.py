#%% ------------------------------------------- IMPORT PACKAGES --------------------------------------------------------

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


#%% ------------------------------------------- DATA LOADER ------------------------------------------------------------

def get_dataloader(batch_size, image_size, data_dir):
    transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader
train_loader = get_dataloader(batch_size, img_size, data_dir=DATA_DIR)

#%% ------------------------------------------- SCALE IMAGES -----------------------------------------------------------

def scale(x, feature_range=(-1, 1)):
    min, max = feature_range
    x = x * (max - min) + min
    return x

#%% ------------------------------------------- GENERATOR --------------------------------------------------------------
# Aira
def gen_conv(input_c, output, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    decon = nn.ConvTranspose2d(input_c, output, kernel_size, stride, padding, bias=False)
    layers.append(decon)

    if batch_norm:
        layers.append(nn.BatchNorm2d(output))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim * 8 * 2 * 2)
        self.layer_1 = gen_conv(conv_dim * 8, conv_dim * 4, 4)
        self.layer_2 = gen_conv(conv_dim * 4, conv_dim * 2, 4)
        self.layer_3 = gen_conv(conv_dim * 2, conv_dim, 4)
        self.layer_4 = gen_conv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.conv_dim * 8, 2, 2)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = torch.tanh(self.layer_4(x))
        return x

