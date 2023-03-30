
import numpy
from PIL import Image
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import os
import torch
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
for file in files:
    if file.find("real_B")>=0:
        im = numpy.array(Image.open('{path}/{file}'.format(path=path, file = file)))
        real.append(im)
    elif file.find("fake_B")>=0:
        im = numpy.array(Image.open('{path}/{file}'.format(path=path, file = file)))
        fake.append(im)
    else:
        im = numpy.array(Image.open('{path}/{file}'.format(path=path, file = file)))
        orig.append(im)
len_real = len(real)
real = numpy.array(real)
# print(real)
real = real.astype(numpy.float32)
# real = real.reshape((len_real, 256, 256, 3))

len_fake = len(fake)
print(len_real, len_fake)
fake = numpy.array(fake)
fake = fake.astype(numpy.float32)
# print(fake)
# fake = fake.reshape((len_fake, 256, 256, 3))
# res = calculate_fid(fake, real)
# print(res)
print(im.shape)

model = NLayerDiscriminator(6)
model.load_state_dict(torch.load("latest_net_D.pth"))
print(model)
actReal = []
actFake = []
for i in range(len_real):
    inp = numpy.concatenate((orig[i], fake[i]), 1)
    print(inp.shape)
    act1 = model.forward(torch.from_numpy(inp))
    # act2 = model.forward(torch.cat((orig[i], real[i]), 1))
    print(act1)

