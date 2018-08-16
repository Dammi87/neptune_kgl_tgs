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
"""
generator = get_input_fn(is_test=True)['test']()
for i, img_idx in enumerate(generator):
        img, idx = img_idx
        print(img.shape, img.dtype, img.min())
        raise
"""
input_fn = get_input_fn()

train_input = input_fn['train']()
valid_input = input_fn['valid']()

with tf.Session() as sess:
    for _ in range(2):
        for inp in [train_input, valid_input]:
            try:
                while True:
                    features, labels = sess.run(inp)
                    masks = labels['mask']

                    print(features['img'].dtype, features['img'].shape, labels['mask'].dtype, labels['mask'].shape)
            except Exception as e:
                print(e)
