from fastai.basics import *
from fastai.vision import models
from fastai.vision.all import *
from fastai.metrics import *
from fastai.data.all import *
from fastai.callback import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from semtorch import get_segmentation_learner


def retrain(csvfile, modelpath, numeph,bs, lr):
    # define paths
    model_name = os.path.basename(modelpath)
    model_name = str.replace(model_name, '.pth', '_retrain')
    outdir = os.path.dirname(modelpath)

    # read in csv for training loop
    fns = pd.read_csv(csvfile)

    # define datablock and load data
    codes = ['Background','Tumor']
    segdata = DataBlock(blocks=(ImageBlock,MaskBlock),splitter=ColSplitter(),get_x=ColReader('img'),get_y=ColReader('label'),item_tfms=[Resize((500,500))],batch_tfms=[Normalize.from_stats(*imagenet_stats), Rotate(), RandomErasing(p=0.7, sl = 0.05, sh = 0.05, min_aspect=0.05, max_count = 1) , Contrast(max_lighting = 0.7, p=0.9)])
    segdata.summary(fns)
    dls=segdata.dataloaders(fns,bs=bs,tfm_y=True)

    # show random batch
    dls.show_batch(figsize=(12,12))
    plt.savefig('retrain_showbatch.png')

    # define learner
    learn = get_segmentation_learner(dls=dls, number_classes=2, segmentation_type="Semantic Segmentation",
                                     cbs=[SaveModelCallback(fname=model_name)],
                                     architecture_name="deeplabv3+", backbone_name="resnet50",
                                     metrics=[Dice(), JaccardCoeff()],wd=1e-2).to_fp16()
    # load pre-trained weights
    learn.load(str.replace(modelpath,'.pth',''))
    learn.fit_one_cycle(numeph, lr)
    learn.to_fp32()

    # save pkl and pth files
    learn.save(os.path.join(outdir,model_name))
    learn.export(os.path.join(outdir,model_name+'.pkl'))

    # show results
    learn.remove_cbs([SaveModelCallback])
    learn.show_results(max_n=6, figsize=(7,8))
    plt.savefig('retrain_results.png')


if __name__ == '__main__':
    csv_fullpath = './trainval_dummy.csv'
    model_fullpath = './models/deeplabv3_resnet50_10ep_lr1e4_nonorm.pth'
    bs = 4 # batch size
    numeph = 25 # number epochs for fine tuning
    lr = 1e-4 # learning rate