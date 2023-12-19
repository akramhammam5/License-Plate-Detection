import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

# Function to create TFRecord example
def create_tf_example(row, path):
    # Convert the file name to a string
    filename = str(row['filename'])

    # Check if the filename is 'nan' or if the file does not exist
    if pd.isna(filename) or not os.path.exists(os.path.join(path, filename)):
        print(f"Warning: Skipping example with filename '{filename}' as it does not exist.")
        return None

    # Read image file
    with tf.io.gfile.GFile(os.path.join(path, filename), 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    # Convert image format to bytes
    filename = filename.encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # Assuming 'region_shape_attributes' contains bounding box information in JSON format
    regions = eval(row['region_shape_attributes'])

    for region in regions:

        classes_text.append(row['region_attributes'].encode('utf8'))
        classes.append(1)  # Adjust class index as needed

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Function to create TFRecord file
def create_tf_record(output_filename, examples):
    with tf.io.TFRecordWriter(output_filename) as writer:
        for index, example in examples.iterrows():
            tf_example = create_tf_example(example, path_to_images)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

# Path to the directory containing your images
path_to_images = '/home/sepi0l/Downloads/Test'

# Load annotations CSV file
annotations = pd.read_csv('/home/sepi0l/Downloads/Test/annotations.csv')

# Split the dataset if needed (train, test, validation)
train_examples = annotations  # Adjust as needed

# Output TFRecord file for training
output_path = 'test.record'
create_tf_record(output_path, train_examples)

