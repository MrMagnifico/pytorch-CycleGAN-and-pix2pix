import numpy
from PIL import Image
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import os
import torch
import torchvision.transforms as transforms
from models.networks import NLayerDiscriminator
parent = "pix2pixpspnet"

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

path = 'results/{parent}/test_latest/images'.format(parent=parent)
files = os.listdir(path)
real = []
fake = []
orig = []
transform = transforms.Compose([
            transforms.PILToTensor()
        ])
transform_list = [transforms.ToTensor()]
transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)
for file in files:
    if file.find("real_B")>=0:
        
        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        im = transform(Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB'))
       
        real.append(im.unsqueeze(0))
    elif file.find("fake_B")>=0:
        im = transform(Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB'))
        fake.append(im.unsqueeze(0))
    else:
        im = transform(Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB'))
        orig.append(im.unsqueeze(0))
len_real = len(real)
# real = numpy.array(real)
# # print(real)


# len_fake = len(fake)
# print(len_real, len_fake)
# fake = numpy.array(fake)
# print(fake)

# res = calculate_fid(fake, real)
# print(res)
print(im.shape)

model = NLayerDiscriminator(6)
model.load_state_dict(torch.load("latest_net_D.pth"))
print(model)
actReal = []
actFake = []
for i in range(len_real):
    inp = torch.cat((orig[i], fake[i]), 1)
    inp = inp.type(torch.FloatTensor)
    
    act1 = model.forward(inp).detach().numpy().reshape((1,30*30))
    # print(act1)
    actFake.append(act1)
    inp = torch.cat((orig[i], real[i]), 1)
    inp = inp.type(torch.FloatTensor)
    act2 = model.forward(inp).detach().numpy().reshape((1,30*30))
    print(act2.shape)
    actReal.append(act2)
    # print(act1)
actReal = numpy.array(actReal).reshape((len_real,30*30))
print(actReal.shape)
res = calculate_fid(actReal ,numpy.array(actFake).reshape((len_real,30*30)))
print(res)
    # print(act1)
# print(act2)

