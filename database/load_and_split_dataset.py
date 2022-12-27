import tensorflow as tf
import json
from utils.path_helper import images, labels

BASE_DIR = images

# minimize the resolution of images without losing the Features of picture
IMAGE_SIZE = 224
#  the number of training examples in one forward/backward pass. The higher the batch size,
#  the more memory space you'll need
BATCH_SIZE = 5


def split_dataset():
    # Use `ImageDataGenerator` to rescale the images.
    # Create the train generator and specify where the train dataset directory, image size, batch size.
    # Create the validation generator with similar approach as the train generator
    # with the flow_from_directory() method.
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.1)

    # Training data generator
    train_data = data_generator.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training'
    )
    # Validation data generator
    val_data = data_generator.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    # Triggering a training generator for all the batches
    # for image_batch, label_batch in train_data:
    # break

    # Labels assigned to batches
    print(train_data.class_indices)

    # Writing the labels assigned to batches
    with open(labels / "labels.json", "w") as outfile:
        json.dump(train_data.class_indices, outfile, indent=4)

    return train_data, val_data


# Driver Code
# To test the split_dataset() function
if __name__ == "__main__":
    split_dataset()
