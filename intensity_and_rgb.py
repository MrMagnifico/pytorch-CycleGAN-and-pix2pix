import numpy as np
from PIL import Image

import os
import sys
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm



import matplotlib.pyplot as plt


modelName = sys.argv[1]

path = 'results/{parent}/test_latest/images'.format(parent=modelName)
files = os.listdir(path)
real = []
fake = []
orig = []
r_real = np.zeros((256,))
g_real = np.zeros((256,))
b_real = np.zeros((256,))

r_fake = np.zeros((256,))
g_fake = np.zeros((256,))
b_fake = np.zeros((256,))
for file in files:
    if file.find("real_B")>=0:
        
        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB')
        for x in range(im.width):
            for y in range(im.height):
                r_real[im.getpixel((x,y))[0]]+=1
                g_real[im.getpixel((x,y))[1]]+=1
                b_real[im.getpixel((x,y))[2]]+=1
  
                    
        real.append(np.array(im.convert('L')).mean())
    elif file.find("fake_B")>=0:
        im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB')
        for x in range(im.width):
            for y in range(im.height):
                r_fake[im.getpixel((x,y))[0]]+=1
                g_fake[im.getpixel((x,y))[1]]+=1
                b_fake[im.getpixel((x,y))[2]]+=1
        fake.append(np.array(im.convert('L')).mean())
    else:
        im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('L')
        orig.append(np.array(im.convert('L')))
print("REAL AVERAGE INTENSITY: ", sum(real)/len(real))
print("FAKE AVERAGE INTENSITY: ", sum(fake)/len(fake))
x = np.arange(0,256,1)

fig, ax = plt.subplots()
line1, = ax.plot(x, r_real, color="black", label='real red')
line2, = ax.plot(x, r_fake, color="red", label='fake red')
ax.legend(handles=[line1, line2])
fig.savefig('red_{model}.png'.format(model = modelName))

fig, ax = plt.subplots()
line1, = ax.plot(x, b_real, color="black", label='real blue')
line2, = ax.plot(x, b_fake, color="blue", label='fake blue')
ax.legend(handles=[line1, line2])
fig.savefig('blue_{model}.png'.format(model = modelName))

fig, ax = plt.subplots()
line1, = ax.plot(x, g_real, color="black", label='real green')
line2, = ax.plot(x, g_fake, color="green", label='fake green')
ax.legend(handles=[line1, line2])
fig.savefig('green_{model}.png'.format(model = modelName))