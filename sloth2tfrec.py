import json
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
import os

LABEL_DICT =  {
    "card_A" : 1,
    "card_2" : 2,
    "card_3" : 3,
    "card_4" : 4,
    "card_5" : 5,
    "card_6" : 6,
    "card_7" : 7,
    "card_8" : 8,
    "card_9" : 9,
    "card_J" : 10,
    "card_Q" : 11,
    "card_K" : 12,
    }

flags = tf.app.flags
flags.DEFINE_string('output_path', 'cards_train.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS



def create_tf_example(example):
  file = example['filename']
  filename = file.encode()

  im = Image.open('data/frames/' + file)
  width, height = im.size


  # imgByteArr = io.BytesIO()
  # im.save(imgByteArr, format='JPEG')
  # imgByteArr = imgByteArr.getvalue()
  # encoded_image_data = imgByteArr # Encoded image bytes

  with tf.gfile.GFile('data/frames/' + example['filename'], 'rb') as fid:
      encoded_image_data = fid.read()

  image_format = b'jpeg' # or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  for annot in example['annotations']:
      xmins.append(float(annot['x'] / width))
      xmaxs.append(float((annot['x'] + annot['width']) / width))
      ymins.append(float(annot['y'] / height))
      ymaxs.append(float((annot['y'] + annot['height']) / height))
      classes_text.append(annot['class'].encode())
      classes.append(int(LABEL_DICT[annot['class']]))


  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example



writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

valid_images = 0
with open('sloth-annot.json') as json_data:
    data = json.load(json_data)
    for item in data:
        if len(item['annotations']) > 0:
            valid_images += 1
            item['filename'] = os.path.basename(item['filename'])
            tf_example = create_tf_example(item)
            writer.write(tf_example.SerializeToString())

print('Finished with %s marked images' % valid_images)
writer.close()
