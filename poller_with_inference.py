r"""Watches specific directory and runs inference on the images.

Usage:
  python poller_with_inference.py --watch_path=path/to/watch \
    --output_file=path/to/output.csv --model_path=path/to/model --batch_size=4
"""

import collections
import multiprocessing
import os
import queue
import time
import threading

from absl import app
from absl import flags
from absl import logging
import cv2
from cots_tracker_v2 import OpticalFlowTracker
from cots_tracker_v2_types import Detection
import numpy as np
import tensorflow as tf
from watchdog import events
from watchdog import observers

FLAGS = flags.FLAGS

flags.DEFINE_string('watch_path', None, 'Path to watch for new images.')
flags.DEFINE_string('output_file', None, 'A csv file to append new detections.')
flags.DEFINE_integer('batch_size', 4,
                     'number of images to send for inference at once.')

flags.DEFINE_string('model_path', None, 'Path to inference SavedModel.')
flags.DEFINE_string('model_signature', 'serving_default',
                    'Signature of the model to run.')
flags.DEFINE_float('detection_threshold', 0.4,
                   'Detection confidence threshold to return.')

flags.mark_flags_as_required(['watch_path', 'output_file', 'model_path'])

_IMAGE_TYPE = tf.uint8
_POLLER_TIMEOUT_SEC = 0.4
_CLASS_ID_TO_LABEL = ('COTS',)
_MAX_DETECTION_FPS = 10

file_queue = multiprocessing.Queue()
tracking_queue = multiprocessing.Queue()
terminate_tracking_thread = False

image_shape = None

class Detector():
  """Loads a COTS detection model and runs inference."""

  def __init__(self, model_path):
    super().__init__()
    start = time.time()
    logging.info('Loading model..')
    self._model = tf.saved_model.load(model_path)

    try:
      serving_fn = self._model.signatures[FLAGS.model_signature]
    except KeyError:
      raise KeyError(f'Model does not have signature {FLAGS.model_signature}. '
                     f'Available signatures: {list(self._model.signatures)}')

    @tf.function(
        input_signature=[tf.TensorSpec((None, None, None, 3), _IMAGE_TYPE)])
    def model_fn(data):
      return serving_fn(data)

    self._model_fn = model_fn

    # Warm up.
    # TODO: Read model input size from model.
    for i in range(10):
      self._model_fn(tf.zeros((FLAGS.batch_size, 1080, 1920, 3), dtype=tf.uint8))

    logging.info(f'Model loading done in {time.time() - start:.02f}s.')

  def process_images(self, images):
    logging.info(f'Inference request with tensor shape: {images.shape}')

    if images.shape[0] < FLAGS.batch_size:
      images = tf.pad(images, [[0, FLAGS.batch_size - images.shape[0]], [0, 0], [0, 0], [0, 0]])

    detections = self._model_fn(images)

    batch_size, img_h, img_w = images.shape[0:3]

    num_detections = detections['num_detections'].numpy().astype(np.int32)
    detection_boxes = detections['detection_boxes'].numpy()
    detection_classes = detections['detection_classes'].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'].numpy()

    # print('Detected:', num_detections)

    detections = [[] for _ in range(batch_size)]

    for batch_index in range(batch_size):
      valid_indices = detection_scores[batch_index, :] >= FLAGS.detection_threshold
      classes = detection_classes[batch_index, valid_indices]
      scores = detection_scores[batch_index, valid_indices]
      boxes = detection_boxes[batch_index, valid_indices, :]

      for class_id, score, box in zip(classes, scores, boxes):
        detections[batch_index].append(Detection(
            class_id=class_id,
            score=score,
            x0=box[1] * img_w,
            y0=box[0] * img_h,
            x1=box[3] * img_w,
            y1=box[2] * img_h,
        ))

    return detections


def data_gen():
  try:
    yield file_queue.get(timeout=_POLLER_TIMEOUT_SEC)
  except queue.Empty:
    pass  # Allow empty dataset to be passed to prevent process from blocking


def format_tracker_response(file_path, tracks):
  """Formats tracker response in csv format."""
  result = os.path.basename(file_path)
  for track in tracks:
    detection_columns = [
        _CLASS_ID_TO_LABEL[track.det.class_id],
        str(track.det.score),
        str(track.id),
        str(len(track.linked_dets)),
        str(track.det.x0),
        str(track.det.y0),
        str(track.det.x1 - track.det.x0),
        str(track.det.y1 - track.det.y0)
    ]
    result += ', { ' + ','.join(detection_columns) + '}'
  result += ','
  return result


class Handler(events.FileSystemEventHandler):
  """Event handler for newly created images."""

  def __init__(self):
    # Keep track of the last N timestamps of frames that were forwarded to
    # the detector, so we can try to reach a target FPS by dropping frames.
    self._frame_timestamps = collections.deque()
    self._max_frame_timestamps = 20
    self._min_timestamp_diff = self._max_frame_timestamps / _MAX_DETECTION_FPS

  def on_created(self, event):
    event_path = event.src_path
    if event.is_directory:
      return
    try:
      # The on_created is often called when the file is opened, but not written
      # to yet, so try to wait a short while to see if will be written to.
      num_tries = 5
      file_size = 0
      while num_tries > 0 and file_size == 0:
        file_size = os.path.getsize(event_path)
        if file_size == 0:
          num_tries -= 1
          time.sleep(0.01)
      if file_size == 0:
        logging.info(f'Ignoring {event_path} - Empty file.')
        return
    except OSError:
      logging.info(f'Ignoring {event_path} - File was deleted.')
      return
    if event_path[-4:] != '.jpg':
      logging.info(f'Ignoring {event_path} - Not a jpeg file.')
      return

    current_timestamp = time.time()
    if len(self._frame_timestamps) == self._max_frame_timestamps:
      if current_timestamp - self._frame_timestamps[0] < self._min_timestamp_diff:
        logging.info(f'Ignoring {event_path} - Too many frames per second.')
        return

    logging.info(f'Reading {event_path}.')

    self._frame_timestamps.append(current_timestamp)
    while len(self._frame_timestamps) > self._max_frame_timestamps:
      self._frame_timestamps.popleft()

    global image_shape
    if image_shape is None:
      image = tf.io.read_file(event_path)
      image = tf.io.decode_jpeg(image, try_recover_truncated=True)
      image_shape = image.numpy().shape
      logging.info(f'Using image shape {image_shape}')

    image_bgr = cv2.imread(event_path, cv2.IMREAD_COLOR)
    if image_bgr.size == 0:
      return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    file_queue.put((time.time(), event_path, image_rgb))


def run_inference(data, detector):
  """Runs inference on a batch of images."""
  inference_start = time.time()
  detector_output = detector.process_images(data[2])
  logging.info('Inference: %.2fms', (time.time() - inference_start) * 1000)

  for timestamp, file_path, image, detections in zip(data[0], data[1], data[2], detector_output):
    tracking_queue.put((timestamp, file_path, image, detections))


def tracking_thread_fn():
  tracker = OpticalFlowTracker(tid=1)

  while not terminate_tracking_thread:
    output_lines = []

    try:
      (timestamp, file_path, image, detections) = tracking_queue.get(timeout=1.0)

      # Always call tracker to propagate previous detections.
      tracks = tracker.update(image.numpy(), detections, timestamp)
        
      file_path = file_path.numpy().decode('utf-8')
      output_lines.append(format_tracker_response(file_path, tracks))
      output_lines.append('')

      try:
        with open(FLAGS.output_file, 'a') as output_file:
          output_file.write('\n'.join(output_lines))
      except (OSError, IOError) as e:
        logging.error('Error writing to file %s', e.strerror)
    except queue.Empty:
      pass


def poller(detector):
  """Runs main poller loop that fetches files and runs inference."""
  image_ds = tf.data.Dataset.from_generator(
      data_gen,
      output_types=(tf.float32, tf.string, tf.uint8),
      output_shapes=(tf.TensorShape([]), tf.TensorShape([]),
                     tf.TensorShape([1080, 1920, 3])),
  )

  image_count = 0
  elapsed_sec = 0
  while True:
    start = time.time()
    for data in image_ds.repeat().batch(FLAGS.batch_size):
      run_inference(data, detector)
      elapsed_sec += time.time() - start
      image_count += data[0].numpy().size
      logging.info('Total inference: %d, FPS: %.2f', image_count,
                    image_count / elapsed_sec)
      start = time.time()


def main(unused_argv):
  tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
  tf.config.optimizer.set_jit(True)

  detector = Detector(FLAGS.model_path)
  event_handler = Handler()
  observer = observers.Observer()
  observer.schedule(event_handler, FLAGS.watch_path, recursive=True)
  observer.start()

  tracking_thread = threading.Thread(target=tracking_thread_fn)
  tracking_thread.start()

  try:
    poller(detector)
  except KeyboardInterrupt:
    observer.stop()
    terminate_tracking_thread = True
  observer.join()
  tracking_thread.join()


if __name__ == '__main__':
  app.run(main)
