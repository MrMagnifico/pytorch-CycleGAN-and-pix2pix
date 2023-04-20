import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
from os import listdir, path 
from PIL import Image
from tqdm import tqdm, trange

FIRST_CHECKPOINT    = 5
CHECKPOINT_STEP     = 5
LAST_CHECKPOINT     = 200
EXP_MODEL_PAIRS     = [
    ("pix2pixdeeplabv3plus", "deeplabv3plus"),
    ("pix2pixhrnet48", "hrnet_48"),
    ("pix2pixlinknet", "linknet"),
    ("pix2pixpspnetnew", "pspnet"),
    ("pix2pixresnet9blocks", "resnet_9blocks"),
    ("pix2pixunet256", "unet_256"),
    ("pix2pixunetppnew", "unetpp")
]

def eval_all_models_and_epochs():
    for exp_name, generator in tqdm(EXP_MODEL_PAIRS, desc="Evaluating models"):
        for checkpoint in trange(FIRST_CHECKPOINT, LAST_CHECKPOINT + CHECKPOINT_STEP, CHECKPOINT_STEP, desc="Evaluating checkpoints"):
            test_cmd = f"{sys.executable} test.py --dataroot {path.join('datasets', 'facades')} --name {exp_name} --model pix2pix --netG {generator} --direction BtoA --epoch {checkpoint}"
            subprocess.run(test_cmd.split(), stdout=sys.stdout)

def calculate_fid_scores() -> dict:
    from eval import EvalGAN

    scores = {}
    for exp_name, generator in tqdm(EXP_MODEL_PAIRS, desc="Evaluating models"):
        scores[generator] = dict()
        for checkpoint in trange(FIRST_CHECKPOINT, LAST_CHECKPOINT + CHECKPOINT_STEP, CHECKPOINT_STEP, desc="Evaluating checkpoints"):
            # Collect needed images
            path    = f'results/{exp_name}/test_{checkpoint}/images'
            files   = listdir(path)
            real    = []
            fake    = []
            orig    = []
            for file in files:
                if file.find("real_B")>=0:
                    im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB')
                    real.append(np.array(im))
                elif file.find("fake_B")>=0:
                    im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB')
                    fake.append(np.array(im))
                else:
                    im = Image.open('{path}/{file}'.format(path=path, file = file)).convert('RGB')
                    orig.append(np.array(im))

            # Compute and store FID score
            geval                           = EvalGAN(real, fake)
            fid                             = geval.calculate_fid()
            scores[generator][checkpoint]   = fid
    return scores

def plot_fid_scores(scores: dict):
    for exp_name, generator in EXP_MODEL_PAIRS:
        # Get values to plot
        epochs_str = list(scores[generator].keys())
        epochs_int = list(map(lambda epoch: int(epoch), epochs_str))
        fid_scores = list(scores[generator].values())

        # Create graph, save to file, and clear for next figure to be able to draw
        plt.plot(epochs_int, fid_scores)
        plt.title(generator)
        plt.xlabel("Epoch")
        plt.ylabel("FID Score")
        plt.savefig(path.join("results", exp_name, "fid_scores.png"), bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    # Generation
    eval_all_models_and_epochs()
    fid_scores = calculate_fid_scores()

    # Storage
    with open(path.join("results", 'fid_scores.json'), 'x') as scores_file:
        json.dump(fid_scores, scores_file, indent=1)

    # Plotting
    with open(path.join("results", 'fid_scores.json'), 'r') as scores_file:
        fid_scores = json.load(scores_file)
        plot_fid_scores(fid_scores)
