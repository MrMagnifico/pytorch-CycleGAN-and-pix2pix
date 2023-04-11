import numpy
from PIL import Image
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import os
import sys
import torch
import torchvision.transforms as transforms
from models.networks import NLayerDiscriminator

modelName = sys.argv[1]
print(modelName)
# stolen :)
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

pathImageFolder = 'results/{parent}/test_latest/images'.format(parent=modelName)
files = os.listdir(pathImageFolder)
real = []
fake = []
orig = []
transform = transforms.Compose([
            transforms.PILToTensor()
        ])

# load images:
for file in files:
    if file.find("real_B")>=0:
        # Convert the PIL image to Torch tensor
        im = transform(Image.open('{path}/{file}'.format(path=pathImageFolder, file = file)).convert('RGB'))
        real.append(im.unsqueeze(0))

    elif file.find("fake_B")>=0:
        im = transform(Image.open('{path}/{file}'.format(path=pathImageFolder, file = file)).convert('RGB'))
        fake.append(im.unsqueeze(0))

    else:
        im = transform(Image.open('{path}/{file}'.format(path=pathImageFolder, file = file)).convert('RGB'))
        orig.append(im.unsqueeze(0))


len_real = len(real)

print(im.shape)
 
model = NLayerDiscriminator(6)
model.load_state_dict(torch.load('checkpoints/{model}/latest_net_D.pth'.format(model=modelName)))
print(model)
actReal = []
actFake = []

for i in range(len_real):
    inp = torch.cat((orig[i], fake[i]), 1)
    inp = inp.type(torch.FloatTensor)
    
    act1 = model.forward(inp).detach().numpy()
    
    actFake.append(act1)
    inp = torch.cat((orig[i], real[i]), 1)
    inp = inp.type(torch.FloatTensor)
    act2 = model.forward(inp).detach().numpy()
    actReal.append(act2)
    
actReal = numpy.array(actReal).reshape((len_real,30*30))
actFake = numpy.array(actFake).reshape((len_real,30*30))
print(actReal.shape)
res = calculate_fid(actReal , actFake)
print("FID SCORE ", res)
    # print(act1)
# print(act2)

