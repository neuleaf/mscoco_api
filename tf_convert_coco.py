import tensorflow as tf
import coco_to_tfrecords

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
            'dataset_name', 'mscoco',
                'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
            'dataset_dir', 'data/mscoco/',
                'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
            'output_name', 'coco_train',
                'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
            'output_dir', 'data/tfrecords',
                'Output directory where to store TFRecords files.')


def main(_):
        if not FLAGS.dataset_dir:
                    raise ValueError('You must supply the dataset directory with --dataset_dir')
                    print('Dataset directory:', FLAGS.dataset_dir)
                        print('Output directory:', FLAGS.output_dir)

                            if FLAGS.dataset_name == 'mscoco':
                                        coco_to_tfrecords.run(FLAGS.dataset_dir, FLAGS.output_dir, FLAGS.output_name)
                                            else:
                                                        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)

                                                    if __name__ == '__main__':
                                                            tf.app.run()

