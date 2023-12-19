import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

# Function to create TFRecord example
def create_tf_example(row, path):
    # Read image file
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(row['filename'])), 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    # Convert image format to bytes
    filename = row['filename'].encode('utf8')
    image_format = b'jpg'
    region_shape_attributes = [row['region_shape_attributes']]

    classes_text = [row['file_size']]
    classes = [1]  # Adjust class index as needed

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/region_shape_attributes': dataset_util.float_list_feature(region_shape_attributes),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Function to create TFRecord file
def create_tf_record(output_filename, examples):
    with tf.io.TFRecordWriter(output_filename) as writer:
        for index,example in examples.iterrows():
            tf_example = create_tf_example(example, path_to_images)
            writer.write(tf_example.SerializeToString())

# Path to the directory containing your images
path_to_images = '/home/sepi0l/Downloads/Train'

# Load annotations CSV file
annotations = pd.read_csv('/home/sepi0l/Downloads/Train/annotations.csv')

# Split the dataset if needed (train, test, validation)
train_examples = annotations  # Adjust as needed

# Output TFRecord file for training
output_path = 'train.record'
create_tf_record(output_path, train_examples)

