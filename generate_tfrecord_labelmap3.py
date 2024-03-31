from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
from tensorflow.python.framework.versions import VERSION
if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
flags.DEFINE_string('label_map_output_path', '', 'Path to output label map')
FLAGS = flags.FLAGS

def class_text_to_int(row_label):
    class_mapping = {
        "ba": 1, "ca": 2, "da": 3, "dha": 4, "ga": 5,
        "ha": 6, "ja": 7, "ka": 8, "la": 9, "ma": 10,
        "na": 11, "nga": 12, "nya": 13, "pa": 14, "ra": 15,
        "sa": 16, "ta": 17, "tha": 18, "wa": 19, "ya": 20
    }
    return class_mapping.get(row_label, None)

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

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

def create_label_map(pbtxt_path):
    class_mapping = {
        "ba": 1, "ca": 2, "da": 3, "dha": 4, "ga": 5,
        "ha": 6, "ja": 7, "ka": 8, "la": 9, "ma": 10,
        "na": 11, "nga": 12, "nya": 13, "pa": 14, "ra": 15,
        "sa": 16, "ta": 17, "tha": 18, "wa": 19, "ya": 20
    }
    label_map_content = ''
    for class_name, class_id in class_mapping.items():
        label_map_content += f"item {{\n  id: {class_id}\n  name: '{class_name}'\n}}\n"

    with open(pbtxt_path, 'w') as f:
        f.write(label_map_content)

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

    if FLAGS.label_map_output_path:
        create_label_map(FLAGS.label_map_output_path)
        print('Successfully created the label map file: {}'.format(FLAGS.label_map_output_path))

if __name__ == '__main__':
    tf.app.run()