from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Dropout
from tensorflow.keras.layers import concatenate, Activation
from tensorflow.keras.models import Model

import cv2


def DenseNet_BC_Model():
    # training parameters
    batch_size = 3
    epochs = 200
    data_augmentation = False

    # network parameters
    num_classes = 3
    num_dense_blocks = 3
    use_max_pool = False

    # DenseNet-BC with dataset augmentation
    # Growth rate   | Depth |  Accuracy (paper)| Accuracy (this)      |
    # 12            | 100   |  95.49%          | 93.74%               |
    # 24            | 250   |  96.38%          | requires big mem GPU |
    # 40            | 190   |  96.54%          | requires big mem GPU |
    growth_rate = 12
    depth = 100
    num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)

    num_filters_bef_dense_block = 2 * growth_rate
    compression_factor = 0.5

    input_shape = (64, 64, 3)

    def lr_schedule(epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    # start model definition
    # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(num_filters_bef_dense_block,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal')(x)
    x = concatenate([inputs, x])

    # stack of dense blocks bridged by transition layers
    for i in range(num_dense_blocks):
        # a dense block is a stack of bottleneck layers
        for j in range(num_bottleneck_layers):
            y = BatchNormalization()(x)
            y = Activation('relu')(y)
            y = Conv2D(4 * growth_rate,
                       kernel_size=1,
                       padding='same',
                       kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = Dropout(0.2)(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(growth_rate,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = Dropout(0.2)(y)
            x = concatenate([x, y])

        # no transition layer after the last dense block
        if i == num_dense_blocks - 1:
            continue

        # transition layer compresses num of feature maps and reduces the size by 2
        num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
        num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
        y = BatchNormalization()(x)
        y = Conv2D(num_filters_bef_dense_block,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = Dropout(0.2)(y)
        x = AveragePooling2D()(y)

    # add classifier on top
    # after average pooling, size of feature map is 1 x 1
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    kernel_initializer='he_normal',
                    activation='softmax')(y)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    # inference
    test_data = cv2.imread("Dataset/257.JPG")
    test_data = cv2.resize(test_data, (64, 64))
    test_data = test_data.reshape(1, 64, 64, 3)
    test_data = test_data.astype("float32")
    test_data /= 255


    model = DenseNet_BC_Model()
    model.load_weights("saved_models/cifar10_densenet_model.55_95.23%.h5")
    ans = model.predict(test_data)
    print(ans)
