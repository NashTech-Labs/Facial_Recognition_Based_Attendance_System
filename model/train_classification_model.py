import tensorflow as tf
from database import load_and_split_dataset as data

# Image Shape/Resolution (Width,Height, Array size of 3 for RGB Colors)
IMAGE_SHAPE = (data.IMAGE_SIZE, data.IMAGE_SIZE, 3)

# Loading the dataset.
train_data, val_data = data.split_dataset()
# Triggering a training generator for all the batches
for image_batch, label_batch in train_data:
    break
# Number of EPOCHS
EPOCH = 10
# Saved Models Directory
MODEL_DIR = 'trained_model'
# Creating a base model MobileNet V2 developed by Google, pre-trained on the ImageNet Dataset
# Instantiate an MobileNet V2 model preloaded with weights trained on ImageNet.
# By specifying the `include_top=False` argument, we load a network that doesn't include the
# classification layers at the top, which is ideal for feature extraction.

MOBILE_NET_MODEL = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE,
                                                     include_top=True,
                                                     weights='imagenet',
                                                     pooling=None,
                                                     classes=1000,
                                                     classifier_activation='softmax'
                                                     )


def un_tuned_classification():
    # Feature extraction
    # will be tweaking the model with our own classifications
    # Tweaks should not affect the layers in the 'base_model'
    # Hence we disable their training
    MOBILE_NET_MODEL.trainable = False
    # Adding a classification layer to base model

    model = tf.keras.Sequential([
        MOBILE_NET_MODEL,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    # Compiling our model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Model Summary
    model.summary()

    # Number of Training variables
    print('Number of training variables: {}'.format(len(model.trainable_variables)))

    # Training the model
    trained_model = model.fit(train_data,
                              epochs=EPOCH,
                              validation_data=val_data)

    # Saving the model
    model.save(MODEL_DIR + '/MobileNet_un_tuned_model.h5')
    print(f'Un tuned model is saved at {MODEL_DIR}/MobileNet_un_tuned_model.h5')


def fine_tuned_classification():
    # Feature extraction
    # will be tweaking the model with our own classifications
    # Fine-tuning
    # Un-freezing the top layers of model.
    MOBILE_NET_MODEL.trainable = True

    # Number of layers in the base model
    print("Number of layers in the base model: ", len(MOBILE_NET_MODEL.layers))

    # Fine tune from this layer
    fine_tune_from = 100

    # Freezing all the layers before the layer from which fine tune starts
    for layer in MOBILE_NET_MODEL.layers[:fine_tune_from]:
        layer.trainable = False
    # Adding a classification layer to base model
    model = tf.keras.Sequential([
        MOBILE_NET_MODEL,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    # Compiling our model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Model summary
    model.summary()

    # Printing Training Variables
    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    # Training the model
    history_fine = model.fit(train_data,
                             epochs=EPOCH,
                             validation_data=val_data
                             )
    # Saving fine tuned model
    model.save(MODEL_DIR + '/MobileNet_fine_tuned_model.h5')
    print(f'Fine tuned model is saved at {MODEL_DIR}/MobileNet_fine_tune_model.h5')


# Driver Code
# To train the model
if __name__ == "__main__":
    # un_tuned_classification()
    fine_tuned_classification()
