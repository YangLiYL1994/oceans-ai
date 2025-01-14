r"""Watches specific directory and sends new file to detector service.

Usage:
  python poller.py --watch_path=path/to/watch --output_file=path/to/output.csv \
    --batch_size=8

  # disabling resize
  python poller.py --watch_path=path/to/watch --output_file=path/to/output.csv \
    --batch_size=8 --noresize
"""
import collections
import multiprocessing
import os
import queue
import sys
import time

from absl import app
from absl import flags
from absl import logging

import cots_tracker_v2
from cots_tracker_v2_types import Detection
import grpc
import service_pb2
import service_pb2_grpc
import tensorflow as tf
import time
from watchdog import events
from watchdog import observers

FLAGS = flags.FLAGS

flags.DEFINE_string('watch_path', None, 'Path to watch for new images.')
flags.DEFINE_string('output_file', None, 'A csv file to append new detections.')
flags.DEFINE_integer('batch_size', 1,
                     'number of images to send for inference at once.')
flags.DEFINE_boolean(
    'resize', False,
    'Whether to resize the image here. For some models, the resizing is done '
    'on the model size so there is no need to resize in the poller.')
flags.DEFINE_integer('resize_height', 1920, 'Height of the resized image.')
flags.DEFINE_integer('resize_width', 1920, 'Width of the resized image.')

flags.mark_flags_as_required(['watch_path', 'output_file'])

_MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MiB
_IMAGE_TYPE = tf.uint8
_POLLER_TIMEOUT_SEC = 0.5
_CLASS_ID_TO_LABEL = ('COTS',)
_MAX_DETECTION_FPS = 10

# https://hub.docker.com/r/helmuthva/jetson-xavier-tensorflow-serving
# https://velog.io/@canlion/tensorflow-2.x-od-api-tensorRT

file_queue = multiprocessing.Queue()

image_shape = None


def data_gen():
  try:
    yield file_queue.get(timeout=_POLLER_TIMEOUT_SEC)
  except queue.Empty:
    pass  # Allow empty dataset to be passed to prevent process from blocking


def parse_image(filename):
  """Reads an image."""
  original_image = tf.io.read_file(filename)
  original_image = tf.io.decode_jpeg(original_image, try_recover_truncated=True)
  if FLAGS.resize:
    modified_image = tf.image.resize_with_pad(original_image, FLAGS.resize_height,
                                     FLAGS.resize_width)
  else:
    modified_image = original_image
  if _IMAGE_TYPE == tf.float32:
    image = tf.image.convert_image_dtype(modified_image, tf.float32)
  elif FLAGS.resize:  # only convert back to uint8 when the mage is resized.
    image = tf.cast(tf.round(modified_image), _IMAGE_TYPE)
  else:
    image = modified_image
  return filename, image, original_image


def get_ordered_filename_to_detections(inference_response, original_filenames):
  result = collections.defaultdict(lambda: [])
  result.update({os.path.basename(name): [] for name in original_filenames})
  for detection in inference_response.detections:
    basename = os.path.basename(detection.file_path)
    result[basename].append(detection)
  return collections.OrderedDict(sorted(result.items()))


def format_tracker_response(filename, tracks):
  """Formats tracker response in csv format."""
  result = filename
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
      image = tf.io.decode_jpeg(image)
      image_shape = image.numpy().shape
      logging.info(f'Using image shape {image_shape}')
    file_queue.put(event_path)


def dispatch_inference_and_track(data, tracker, stub):
  """Dispatches a single inference request and runs tracker."""
  inference_start = time.time()
  try:
    filepaths = [entry.decode("utf-8") for entry in list(data[0].numpy())]
    images = [entry for entry in list(data[2].numpy())]
    for idx in  range(len(filepaths)):
      filename_to_image[os.path.basename(filepaths[idx])] = images[idx]
    request = service_pb2.InferenceRequest(
        file_paths=list(data[0].numpy()),
        data=tf.io.serialize_tensor(data[1]).numpy(),
        original_image_height=image_shape[0],
        original_image_width=image_shape[1],
    )
    logging.info(f'Sending inference request.')
    response = stub.Inference(request)
  except grpc.RpcError as e:
    logging.error('gRPC error: %s', str(e))
    return
  finally:
    inference_ms = int((time.time() - inference_start) * 1000)
    logging.info(f'Inference request took {inference_ms}ms')

  filename_to_detections = get_ordered_filename_to_detections(
      response, request.file_paths)
  output_lines = []

  for filename, detections in filename_to_detections.items():
    # TODO: Read this from the jpeg file.
    current_timestamp = time.time()
    detections_for_tracker = []
    for detection in detections:
      detections_for_tracker.append(Detection(class_id=0, score=detection.score, 
        x0=detection.left, y0=detection.top, 
        x1=detection.left + detection.width, 
        y1=detection.top + detection.height))

    # Always call tracker to propagate previous detections.
    tracks = tracker.update(filename_to_image[filename], detections_for_tracker,
                            current_timestamp)
    del filename_to_image[filename]

    if not detections_for_tracker:
      output_lines.append(filename + ',')
    else:
      output_lines.append(format_tracker_response(filename, tracks))
  output_lines.append('')

  try:
    with open(FLAGS.output_file, 'a') as output_file:
      output_file.write('\n'.join(output_lines))
  except (OSError, IOError) as e:
    logging.error('Error writing to file %s', e.strerror)


def create_filename_to_image_map():
  global filename_to_image
  if 'filename_to_image' not in globals():
    filename_to_image = {}

def poller():
  """Runs main poller loop that fetches files and run inference."""
  create_filename_to_image_map()
  ds_counter = tf.data.Dataset.from_generator(
      data_gen,
      output_types=tf.string,
      output_shapes=(),
  )
  image_ds = ds_counter.map(parse_image)

  tracker = cots_tracker_v2.OpticalFlowTracker(tid=1)
  image_count = 0
  elapsed_sec = 0

  with grpc.insecure_channel(
      'localhost:50051',
      options=[
          ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
          ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH),
      ]) as channel:
    stub = service_pb2_grpc.DetectorStub(channel)
    while True:
      start = time.time()
      for data in image_ds.repeat().batch(FLAGS.batch_size):
        dispatch_inference_and_track(data, tracker, stub)
        elapsed_sec += time.time() - start
        image_count += data[0].numpy().size
        logging.info('Total inference: %d, FPS: %.2f', image_count,
                     image_count / elapsed_sec)
        start = time.time()


def main(unused_argv):
  event_handler = Handler()
  observer = observers.Observer()
  observer.schedule(event_handler, FLAGS.watch_path, recursive=True)
  observer.start()

  try:
    poller()
  except KeyboardInterrupt:
    observer.stop()
  observer.join()


if __name__ == '__main__':
  app.run(main)
