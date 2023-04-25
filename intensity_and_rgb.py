from eval_batch import EXP_MODEL_PAIRS
from enum import Enum, IntEnum
from PIL import Image
from scipy.stats import kstest
from os import listdir, path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Colors(IntEnum):
    RED     = 0
    GREEN   = 1
    BLUE    = 2

    __str__ = Enum.__str__

def load_rgb_arrays(path: str):
    files = listdir(path)
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
    return (r_real, g_real, b_real), (r_fake, g_fake, b_fake)

def plot_rgb_values(r_real, g_real, b_real,
                    r_fake, g_fake, b_fake,
                    experiment_name, model_name):
    x = np.arange(0, 256, 1)

    fig, ax = plt.subplots()
    line1, = ax.plot(x, r_real, color="black", label='real')
    line2, = ax.plot(x, r_fake, color="red", label='generated')
    ax.legend(handles=[line1, line2])
    plt.title(f"Red Color Distribution - {model_name}")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    fig.savefig(f'red_{experiment_name}.png')

    fig, ax = plt.subplots()
    line1, = ax.plot(x, b_real, color="black", label='real')
    line2, = ax.plot(x, b_fake, color="blue", label='generated')
    ax.legend(handles=[line1, line2])
    plt.title(f"Blue Color Distribution - {model_name}")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    fig.savefig(f'blue_{experiment_name}.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    line1, = ax.plot(x, g_real, color="black", label='real')
    line2, = ax.plot(x, g_fake, color="green", label='generated')
    ax.legend(handles=[line1, line2])
    plt.title(f"Green Color Distribution - {model_name}")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    fig.savefig(f'green_{experiment_name}.png', bbox_inches='tight')

def ks_test(sampled, ground_truth):
    ks_res = kstest(sampled, ground_truth)
    return f"Test statistic is {ks_res.statistic} with p-value {ks_res.pvalue}"

if __name__ == "__main__":
    for experiment, model in tqdm(EXP_MODEL_PAIRS):
        print(f"===== Processing model {model} =====")
        path = f'results/{experiment}/test_latest/images'

        print("Computing RGB values...")
        reals, fakes = load_rgb_arrays(path)

        print("Plotting values...")
        plot_rgb_values(reals[Colors.RED], reals[Colors.GREEN], reals[Colors.BLUE],
                        fakes[Colors.RED], fakes[Colors.GREEN], fakes[Colors.BLUE],
                        experiment, model)

        print("KS Test results:")
        for channel in Colors:
            ks_res = ks_test(fakes[channel], reals[channel])
            print(f"{channel}: {ks_res}")
