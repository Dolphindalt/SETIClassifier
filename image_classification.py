import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from image_loader import load_seti_dataset
import config
import prepare_data as pd

import resnet

TOTAL_GPU = 2
BS_PER_GPU = 128
NUM_EPOCH = 60
STEPS_EPOCH = 50

NUM_TRAIN_SAMPLES = 100

LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

def preprocess(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y

def augmentation(x, y):
    #x = tf.image.resize_with_crop_or_pad(
    #    x, config.HEIGHT + 8, config.WIDTH + 8)
    #x = tf.image.random_crop(x, [config.WIDTH, config.HEIGHT, NUM_CHANNELS])
    #x = tf.image.random_flip_left_right(x)
    return x, y

def schedule(epoch):
    initial_learning_rate = LEARNING_RATE * BS_PER_GPU / 128
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
        if epoch >= start_epoch:
            learning_rate = initial_learning_rate * mult
        else:
            break
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

#train_dataset = load_seti_dataset("train_data.csv")
#test_dataset = load_seti_dataset("test_data.csv")
#valid_dataset = load_seti_dataset("valid_data.csv")

tf.random.set_seed(2727)
#train_dataset = train_dataset.map(augmentation).map(preprocess).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * TOTAL_GPU, drop_remainder=True)
#test_dataset = test_dataset.map(preprocess).batch(BS_PER_GPU * TOTAL_GPU, drop_remainder=True)

train_generator, valid_generator, test_generator, train_num, valid_num, test_num = pd.get_datasets()

input_shape = (config.HEIGHT, config.WIDTH, config.NUM_CHANNELS)
image_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

if TOTAL_GPU == 1:
    model = resnet.resnet56(img_input=image_input, classes=config.NUM_CLASSES)
    model.compile(
        optimizers=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = resnet.resnet56(img_input=image_input, classes=config.NUM_CLASSES)
        model.compile(
            optimizers=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    update_freq='batch',
    histogram_freq=1
)

lr_schedule_callback = LearningRateScheduler(schedule)

model.fit(
    train_generator,
    epochs=NUM_EPOCH,
    steps_per_epoch=train_num // config.BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=valid_num // config.BATCH_SIZE,
    validation_freq=1,
    callbacks=[tensorboard_callback, lr_schedule_callback]
)
model.evaluate(test_generator)
model.save("model.h5")
new_model = keras.models.load_model("model.h5")
new_model.evaulate(test_generator)