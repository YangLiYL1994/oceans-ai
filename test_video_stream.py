"""This utility simulates a video stream input for the poller.

This utility copies images from a given directory to the input directory of
the poller, at a frame rate of about 10 frames per second.
"""

import os
import shutil
import time

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', None, 'Directory to read images from.')
flags.DEFINE_string('output_dir', None, 'Directory to copy images to.')
flags.DEFINE_integer('max_images', None, 'Maximum number of images to copy.')

flags.mark_flags_as_required(['input_dir', 'output_dir'])


def main(argv):
    input_dir = FLAGS.input_dir
    output_dir = FLAGS.output_dir

    input_files = os.listdir(FLAGS.input_dir)

    input_files = sorted(input_files)    

    if FLAGS.max_images:
        max_images = min(len(input_files), FLAGS.max_images)
    else:
        max_images = len(input_files)

    if len(input_files) > max_images:
        input_files = input_files[:max_images]

    for input_file in input_files:
        logging.info(f'Copying {input_file}')
        shutil.copyfile(os.path.join(input_dir, input_file),
                        os.path.join(output_dir, input_file))
        time.sleep(0.1)


if __name__ == '__main__':
    app.run(main)
