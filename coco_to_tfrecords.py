import os
import sys
import random
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from datasets.coco_common import COCO_LABELS

# Original dataset organisation.
COCO_DIR ='data/mscoco/'
ANNO_DIR = 'Annotations/train2014'
IMG_DIR  = 'train2014/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1000


def _process_image(directory, name):
    '''
    get coco image, shape, bbox and label
    '''
    # read image
    img_file=os.path.join(COCO_DIR,IMG_DIR, name+'.jpg')
    image_data = tf.gfile.FastGFile(img_file, 'rb').read()
    # get shape
    img_ = Image.open(img_file)
    img = np.asarray(img_)
    if len(img.shape)==2:
        c=1
        h,w= img.shape
    else:
        h,w,c=img.shape
    shape=[h,w,c]

    # read annotation
    ann_file=os.path.join(COCO_DIR, ANNO_DIR, name+'.json')

    bboxes=[]
    labels=[]
    with open(ann_file, "r+") as f:
        allData = json.load(f)
        data=allData['annotation']
        print("read ready: ", name)
        for ann in data:
            label=COCO_LABELS[int(ann['category_id'])][0]
            labels.append(label)
            bbox=ann['bbox']
            xmin=float(bbox[0])
            ymin=float(bbox[1])
            xmax=xmin+float(bbox[2])
            ymax=ymin+float(bbox[3])
            bboxes.append((ymin / shape[0],
                           xmin / shape[1],
                           ymax / shape[0],
                           xmax / shape[1]
                            ))

    return image_data, shape, bboxes, labels


def _convert_to_example(image_data, labels, bboxes, shape):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels = _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, bboxes, shape)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%05d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='coco_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    path=os.path.join(COCO_DIR,IMG_DIR)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename.split('.')[0]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MSCOCO dataset!')

