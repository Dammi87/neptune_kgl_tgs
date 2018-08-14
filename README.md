# Introduction
This repository is compatible with [build-with-docker](https://github.com/Dammi87/build-with-docker). If you install build-with-docker, then follow these steps

```bash
git clone https://github.com/Dammi87/neptune_kgl_tgs
cd neptune_kgl_tgs
bwd -build_image
```
This will build all Docker images.

# Run a training job
You can use bwd to run the project, simply do
```bash
cd neptune_kgl_tgs
bwd -s src/trainer/task.py
```
If you look into bwd.json, you can see that it will actually wrap this command and run the following
```bash
neptune run --config ./config/neptune.yaml src/trainer/task.py
```

So, to change different settings, edit the .yaml file before running.

# Running WITHOUT docker
FROM tensorflow/tensorflow:1.8.0-gpu-py3

```bash
pip install tensorflow==1.8.0 \
            kaggle \
            pydensecrf \
            scikit-image \
            neptune-cli \
            opencv-python==3.3.0.9 \
            imgaug \
git clone https://github.com/tensorflow/models
cd models/research/slim && pip install .
```
Good luck :D
