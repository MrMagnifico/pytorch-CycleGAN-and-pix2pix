import numpy as np
import os
import sys
import os
from PIL import Image
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input


class EvalGAN:
    def __init__(self, real_images, fake_images):
        self.real_images = preprocess_input(
            np.array([
                i for i in real_images
            ])
        )
        self.fake_images = preprocess_input(
            np.array([
               i for i in fake_images
            ])
        )
        self.model = InceptionV3(
            include_top=False, 
            pooling='avg', 
            input_shape=(256, 256, 3)
        )
    
    def calculate_fid(self):
        """
        Calculates the Frechet Inception Distance (FID) score.
        Implemented from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
        """
        # calculate activations
        act1 = self.model.predict(self.real_images)
        act2 = self.model.predict(self.fake_images)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

if __name__ == "__main__":
    modelName = sys.argv[1]

    path    = 'results/{parent}/test_latest/images'.format(parent=modelName)
    files   = os.listdir(path)
    real    = []
    fake    = []
    orig    = []

    for file in files:
        if file.find("real_B")>=0:
            # Convert the PIL image to Torch tensor
            im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB')
            real.append(np.array(im))
        elif file.find("fake_B")>=0:
            im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB')
            fake.append(np.array(im))
        else:
            im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB')
            orig.append(np.array(im))

    geval   = EvalGAN(real, fake)
    fid     = geval.calculate_fid()
    print("FID SCORE:", fid)
