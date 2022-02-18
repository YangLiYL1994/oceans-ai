import os
import time

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

FLAGS = flags.FLAGS
_IMAGE_TYPE = tf.uint8
_NUM_ITER = 100

flags.DEFINE_string('model_path', None, 'Model path')
flags.DEFINE_string('image_path', None, 'Image path')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_bool('is_tf_model', True,
                  'If True, assumes the model is a TF model with signatures.')

def parse_image(filename):
  image = tf.io.read_file(filename)
  image = tf.io.decode_jpeg(image)
  image = tf.image.resize(image, [1080, 1920], preserve_aspect_ratio=True)
  if _IMAGE_TYPE == tf.float32:
    image = tf.image.convert_image_dtype(image, tf.float32)
  else:
    image = tf.cast(tf.round(image), _IMAGE_TYPE)
  return filename, image


def main(unused_argv):
  list_ds = tf.data.Dataset.list_files(os.path.join(FLAGS.image_path, '*.jpg'))
  images_ds = list_ds.map(
    parse_image,
    num_parallel_calls=tf.data.AUTOTUNE
  ).prefetch(tf.data.AUTOTUNE)

  start = time.time()
  model = tf.saved_model.load(FLAGS.model_path)

  if FLAGS.is_tf_model:
    serving = model.signatures['serving_default']

  @tf.function(input_signature=[
    tf.TensorSpec((None, None, None, 3), _IMAGE_TYPE)])
  def model_fn(data):
    if FLAGS.is_tf_model:
      return serving(data)
    else:
      return model(data)

  print(f'model loaded in {time.time() - start:.3f}s')

  times = []
  num_images = 0
  
  # JIT warmup.
  num_tries = 0
  for filenames, images in images_ds.batch(FLAGS.batch_size).take(_NUM_ITER):
    num_tries += 1
    if num_tries >= 5:
      break
    _ = model_fn(images)

  for filenames, images in images_ds.batch(FLAGS.batch_size).take(_NUM_ITER):
    start = time.time()
    _ = model_fn(images)
    end = time.time()
    print(f'Inference in {end - start:.3f}s')
    times.append([end - start])
    num_images += len(filenames)
  print(f'Average time taken: {np.mean(times):.3f}s, std: {np.std(times):.3f}')
  print(f'Average FPS: {num_images / np.sum(times):.2f}')


if __name__ == '__main__':
    app.run(main)
