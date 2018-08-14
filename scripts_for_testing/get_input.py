import tensorflow as tf
from src.input_pipe import get_input_fn
"""
import matplotlib.pyplot as plt

def plot_image(imgs):
    f, axarr = plt.subplots(1, len(imgs), dpi=75)
    for i, img in enumerate(imgs):
        if len(img.shape) == 4:
            img = img.squeeze()
        axarr[i].imshow(img)

    plt.show()
"""

input_fn = get_input_fn()

train_input = input_fn['train']()
valid_input = input_fn['valid']()

with tf.Session() as sess:

    for inp in [train_input, valid_input]:
        try:
            for i in range(2):
                features, labels = sess.run(inp)
                masks = labels['mask']

                print(features['img'].shape, labels['mask'].shape)
        except:
            pass
        print("Next\n")
