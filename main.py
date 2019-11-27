import numpy as np
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from dataset_builder import *
import matplotlib.pyplot as plt
from keras.callbacks import *
from models import *


def train_model_denoising(name):
    XTrain = read_faces(r'/media/gabriel/TOSHIBA EXT/img_align_celeba/')

    if name == 'dnet':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model = DeconvNet(
            (XTrain.shape[1],
             XTrain.shape[2],
             XTrain.shape[3]),
            denoising=True)
    elif name == 'unet':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        model = UNet((XTrain.shape[1], XTrain.shape[2], XTrain.shape[3]),
                     denoising=True)
    opt = RMSprop(0.0001)
    model.compile(loss='mse', optimizer=opt)
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=10e-7,
            min_delta=0.001,
            verbose=1),
        # LMSKerasCallback(),
        ModelCheckpoint('./outputs/' + name + '_denoising_model.h5',
                        monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
        # SavePreview((WRITING_DIR / 'preview'), test_data, prefix="model", wnet= False),
        # MetricRecMSE(val, metrics, filename= pathlib.Path(WRITING_DIR / "metrics.txt"), wnet= False),
    ]
    history = model.fit(
        x=XTrain,
        y=XTrain,
        epochs=100,
        validation_split=0.1,
        batch_size=16,
        callbacks=callbacks
    )


if __name__ == "__main__":
    train_model_denoising('unet')
