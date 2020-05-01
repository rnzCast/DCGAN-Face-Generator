"""
MACHINE LEARNING II - FINAL PROJECT
TOPIC: FACE GENERATOR USING DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS (DCGAN)
AUTHORS:
    Aira Domingo
    Renzo Castagnino
DATE: April 2020
"""

# Total code lines: 178
#%% ------------------------------------------- IMPORT PACKAGES --------------------------------------------------------
import torch
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from pathlib import Path
# import os

#%% --------------------------------------------- DATA DIR -------------------------------------------------------------
DATA_DIR = (str(Path(__file__).parents[0]) + '/train/')

#%% -------------------------------------------- MODEL HYPERPARAMETERS -------------------------------------------------
d_conv_dim = 64
g_conv_dim = 64
z_size = 100
n_epochs = 5
batch_size = 128
img_size = 32

#%% ------------------------------------------- DATA LOADER ------------------------------------------------------------
# Aira
def get_dataloader(batch_size, image_size, data_dir):
    transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader
train_loader = get_dataloader(batch_size, img_size, data_dir=DATA_DIR)

#%% ------------------------------------------- SCALE IMAGES -----------------------------------------------------------
# Aira
def scale(x, feature_range=(-1, 1)):
    min, max = feature_range
    x = x * (max - min) + min
    return x

#%% ------------------------------------------- PLOT REAL FACES --------------------------------------------------------
# Renzo
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

data_iter = iter(train_loader)
images, _ = data_iter.next()

fig = plt.figure(figsize=(20, 4))
plot_size = 20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size / 2, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])

#%% ------------------------------------------- DISCRIMINATOR ----------------------------------------------------------
# Renzo
def disc_conv(input_c, output, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    con = nn.Conv2d(input_c, output, kernel_size, stride, padding, bias=False)
    layers.append(con)
    if batch_norm:
        layers.append(nn.BatchNorm2d(output))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.layer_1 = disc_conv(3, conv_dim, 4, batch_norm=False)
        self.layer_2 = disc_conv(conv_dim, conv_dim * 2, 4)
        self.layer_3 = disc_conv(conv_dim * 2, conv_dim * 4, 4)
        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.layer_1(x))
        x = F.leaky_relu(self.layer_2(x))
        x = F.leaky_relu(self.layer_3(x))
        x = x.view(-1, self.conv_dim * 4 * 4 * 4)
        x = self.fc(x)
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

#%% ------------------------------------------- INITIALIZE WEIGHTS -----------------------------------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()

#%% ------------------------------------------- BUILD FULL NETWORK -----------------------------------------------------
def create_model(d_conv_dim, g_conv_dim, z_size):
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
    return D, G
D, G = create_model(d_conv_dim, g_conv_dim, z_size)

#%% ------------------------------------------- CHECK FOR GPU ----------------------------------------------------------
train_on_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% ------------------------------------------- DEFINE LOSS FUNCTIONS --------------------------------------------------
def real_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

#%% ------------------------------------------- OPTIMIZERS -------------------------------------------------------------
d_optimizer = optim.Adam(D.parameters(), lr=.0002, betas=[0.5, 0.999])
g_optimizer = optim.Adam(G.parameters(), lr=.0002, betas=[0.5, 0.999])

#%% ------------------------------------------- DEFINE TRAIN FUNCTION --------------------------------------------------
print('Start Training...')
def train(D, G, n_epochs, print_every=50):
    if train_on_gpu:
        D.cuda()
        G.cuda()

    samples = []
    loss = []
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    for epoch in range(n_epochs):
        for batch_i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = scale(real_images)
            if train_on_gpu:
                real_images = real_images.cuda()

            # Train D
            d_optimizer.zero_grad()
            d_out_real = D(real_images)
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            d_loss = real_loss(d_out_real) + fake_loss(D(G(z)))
            d_loss.backward()
            d_optimizer.step()

            # Train G
            G.train()
            g_optimizer.zero_grad()
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            g_loss = real_loss(D(G(z)))
            g_loss.backward()
            g_optimizer.step()

            if batch_i % print_every == 0:
                loss.append((d_loss.item(), g_loss.item()))
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, n_epochs, d_loss.item(), g_loss.item()))

        G.eval()
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()

    with open('trained_model.pkl', 'wb') as f:
        pkl.dump(samples, f)
    return loss


#%% ------------------------------------------- CALL TRAIN -------------------------------------------------------------
loss = train(D, G, n_epochs=n_epochs)

#%% ------------------------------------------- PLOT LOSS ------------------------------------------------------------
# Renzo
fig, ax = plt.subplots()
loss = np.array(loss)
plt.plot(loss.T[0], label='Discriminator', alpha=0.5)
plt.plot(loss.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()

