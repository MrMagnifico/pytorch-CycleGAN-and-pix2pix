import sys
import os
import numpy as np
import re


import matplotlib.pyplot as plt


modelName = sys.argv[1]

path = 'checkpoints/{parent}/loss_log.txt'.format(parent=modelName)
epoch = 0
g_gan = []
g_l1 = []
d_real = []
d_fake = []
with open(path) as fp:
    tmp_g_gan = []
    tmp_g_l1 = []
    tmp_d_real = []
    tmp_d_fake = []
    for line in fp:

        try:
            epoch_curr = int(re.search('epoch: (.+?),', line).group(1))
            tmp_g_gan.append(float(re.search('G_GAN: (.+?) G_L1:', line).group(1)))
            tmp_g_l1.append(float(re.search('G_L1: (.+?) D_real:', line).group(1)))
            tmp_d_real.append(float(re.search('D_real: (.+?) D_fake:', line).group(1)))
            tmp_d_fake.append(float(re.search('D_fake: (.+?)\n', line).group(1)))
            if epoch_curr != epoch:
                epoch = epoch_curr
                g_gan.append(sum(tmp_g_gan)/len(tmp_g_gan))
                print(tmp_g_gan)
                tmp_g_gan = []
                g_l1.append(sum(tmp_g_l1)/len(tmp_g_l1))
                tmp_g_l1 = []
                d_real.append(sum(tmp_d_real)/len(tmp_d_real))
                tmp_d_real = []
                d_fake.append(sum(tmp_d_fake)/len(tmp_d_fake))
                tmp_d_fake = []

        except AttributeError:
            continue
x = np.arange(0,200,1)

fig, ax = plt.subplots()
line1, = ax.plot(x, g_gan, color="black", label='G_GAN')
line2, = ax.plot(x, g_l1, color="blue", label='G_L1')
ax.legend(handles=[line1, line2])
fig.savefig('GAN_{model}.png'.format(model = modelName))

fig, ax = plt.subplots()
line1, = ax.plot(x, d_real, color="black", label='D_REAL')
line2, = ax.plot(x, d_fake, color="blue", label='D_FAKE')
ax.legend(handles=[line1, line2])
fig.savefig('D_{model}.png'.format(model = modelName))