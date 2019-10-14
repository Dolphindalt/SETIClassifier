import tensorflow as tf
import os

def load_seti_dataset(path):
    with open(path) as f:
        dataset_file = f.read().splitlines()
    data = tf.data.Dataset.from_tensor_slices(dataset_file)
    data = data.map(_parse_records, num_parallel_calls=4)
    data = data.repeat()
    data = data.shuffle(buffer_size=1000)
    data = data.batch(batch_size=20)
    data = data.prefetch(buffer_size=1)
    return data

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, [512, 384])
    image = 1. - image / 127.5
    return image

def _parse_records(line):
    image_path, image_label = tf.io.decode_csv(line, ["", 0])
    image = load_image(image_path)
    return image, image_label

def tensorToKeras(data):
    reinit_itr = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    return reinit_itr.make_initializer(data)

#data = load_seti_dataset("test_data.csv")
#for image, label in data:
#    print("Image shape: ", image.numpy().shape)
#    print("Label: ", label.numpy())