from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def read_faces(base_path_celeb):
    count = 0
    celeb_faces = []
    clean_faces = []
    max_imgs = 100000
    print('celeb_faces')
    for filename in Path(base_path_celeb).glob('**/*.jpg'):
        if count < max_imgs:
            try:
                img_orig = np.array(Image.open(filename))
                # print(img_orig)

                resized = cv2.resize(
                    img_orig, (128, 128), interpolation=cv2.INTER_AREA)

                if len(resized.shape) != 3:
                    continue

                resized = (resized / 255)
                resized = (resized - np.amin(resized)) / \
                    (np.amax(resized) - np.amin(resized))
                noisy = resized + np.random.normal(0, 0.5, (128, 128, 3))
                noisy = (noisy - np.amin(noisy)) / \
                    (np.amax(noisy) - np.amin(noisy))

                clean_faces.append(resized)
                celeb_faces.append(noisy)
                count += 1
            except Exception as e:
                print(e)
        else:
            break
    return np.array(celeb_faces), np.array(clean_faces)
