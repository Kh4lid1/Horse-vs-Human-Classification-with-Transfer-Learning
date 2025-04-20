# Horse-vs-Human-Classification-with-Transfer-Learning
This project demonstrates a binary image classification task to distinguish between images of horses and humans using Transfer Learning with the InceptionV3 model.


Approach:
We apply transfer learning using the InceptionV3 architecture pre-trained on ImageNet:

The convolutional base (include_top=False) is frozen (non-trainable).

The output is taken from the mixed7 layer.

A custom classification head is added (Flatten → Dense → Dropout → Dense).

Training:

Binary classification with sigmoid activation.

Callback stops training when accuracy exceeds 99.9%.

Image Preprocessing:

Images resized to 150x150.

Normalization (rescale=1./255).

Data augmentation on training images (rotation, zoom, flip, etc.).
