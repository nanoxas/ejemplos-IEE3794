import numpy as np
from keras.models import *
from keras.optimizers import Adam, RMSprop
from dataset_builder import *
import matplotlib.pyplot as plt
from keras.callbacks import *
from models import *


def test_model_denoising(name):
    if name == 'dnet':
        model = load_model('./outputs/dnet_denoising_model.h5')
    elif name == 'unet':
        model = load_model('./outputs/unet_denoising_model.h5')
    img_orig = np.array(Image.open('lenna.jpg'))
    resized = (cv2.resize(
        img_orig, (128, 128), interpolation=cv2.INTER_AREA) / 255)
    resized = (resized - np.amin(resized)) / \
        (np.amax(resized) - np.amin(resized))
    noisy = resized + np.random.normal(0, 0.5, (128, 128, 3))
    noisy = (noisy - np.amin(noisy)) / \
        (np.amax(noisy) - np.amin(noisy))

    noisy = np.expand_dims(noisy, axis=0)
    predicted = model.predict(noisy)
    print(predicted[0].shape)
    print(img_orig.shape)
    print(noisy[0].shape)

    stitched = np.concatenate((noisy[0], resized, predicted[0]), axis=1)
    stitched = (stitched - np.amin(stitched)) / \
        (np.amax(stitched) - np.amin(stitched))
    plt.imsave('./denoised_lenna' + name + '.png', stitched)


def train_model_denoising(name):
    XTrain, YTrain = read_faces(
        r'/media/gabriel/TOSHIBA EXT/img_align_celeba/')

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
    model.compile(loss='mae', optimizer=opt)
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
        y=YTrain,
        epochs=100,
        validation_split=0.1,
        batch_size=16,
        callbacks=callbacks
    )


if __name__ == "__main__":
    # train_model_denoising('dnet')
    test_model_denoising('dnet')
