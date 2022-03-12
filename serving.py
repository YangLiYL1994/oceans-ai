"""Inference server. Currently works with TF OD API."""
from concurrent import futures
import time

from absl import app
from absl import flags
from absl import logging

import grpc
import numpy as np
import service_pb2
import service_pb2_grpc
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', None, 'Path to inference SavedModel.')
flags.DEFINE_string('model_signature', 'serving_default',
                    'Signature of the model to run.')
flags.DEFINE_float('detection_threshold', 0.4,
                   'Detection confidence threshold to return.')
flags.DEFINE_integer('batch_size', 4,
                     'Batch size that the model expects.')

_OUTPUT_NAMES = (
    'detection_anchor_indices',
    'detection_boxes',
    'detection_classes',
    'detection_multiclass_scores',
    'detection_scores',
    'num_detections',
    'raw_detection_boxes',
    'raw_detection_scores',
)

_MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MiB
_IMAGE_TYPE = tf.uint8


class Detector(service_pb2_grpc.Detector):
  """Detector service. gets serizliaed tensors and returns detecion results."""

  def __init__(self, model):
    super().__init__()
    self._model = model

    try:
      serving_fn = model.signatures[FLAGS.model_signature]
    except KeyError:
      raise KeyError(f'Model does not have signautre {FLAGS.model_signature}. '
                     f'Available signatures: {list(model.signatures)}')

    @tf.function(
        input_signature=[tf.TensorSpec((None, None, None, 3), _IMAGE_TYPE)])
    def model_fn(data):
      return serving_fn(data)

    self._model_fn = model_fn

    logging.info('Warming up..')

    # Warm-up.
    # TODO: Read the input size from the model.
    for i in range(10):
      self._model_fn(tf.zeros((4, 736, 1280, 3), dtype=tf.uint8))

  def Inference(self, request, context):
    images = tf.io.parse_tensor(request.data, _IMAGE_TYPE)
    start = time.time()
    logging.info(f'Inference request with tensor shape: {images.shape}')

    if images.shape[0] < FLAGS.batch_size:
      images = tf.pad(images, [[0, FLAGS.batch_size - images.shape[0]], [0, 0], [0, 0], [0, 0]])

    detections = self._model_fn(images)
    result = service_pb2.InferenceReply()
    img_h, img_w = images.shape[1:3]

    num_detections = detections['num_detections'].numpy().astype(np.int32)
    detection_boxes = detections['detection_boxes'].numpy()
    detection_classes = detections['detection_classes'].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'].numpy()

    # Temporarily disable padding compensation because the current model will
    # handle this.
    #
    # model_y_padding = (
    #     (request.original_image_width - request.original_image_height) / 2 /
    #     request.original_image_width)
    model_y_padding = 0

    # print('Detected:', num_detections)
    for file_idx, file_path in enumerate(request.file_paths):
      scores = detection_scores[file_idx]
      valid_indices = detection_scores[file_idx, :] >= FLAGS.detection_threshold
      scores = scores[valid_indices]
      classes = detection_classes[file_idx, valid_indices]
      bbox = detection_boxes[file_idx, valid_indices, :]

      for i, pos in enumerate(bbox):
        box_x1 = pos[1]
        box_y1 = (pos[0] - model_y_padding) / (1 - 2 * model_y_padding)
        box_x2 = pos[3]
        box_y2 = (pos[2] - model_y_padding) / (1 - 2 * model_y_padding)
        detection = service_pb2.BoundingBox(
            file_path=file_path,
            class_id=classes[i],
            score=scores[i],
            left=box_x1 * img_w,
            top=box_y1 * img_h,
            width=(box_x2 - box_x1) * img_w,
            height=(box_y2 - box_y1) * img_h,
        )
        result.detections.append(detection)

    inference_ms = int(1000 * (time.time() - start))
    logging.info(f'Inference request done in {inference_ms}ms')
    return result


def serve():
  """Starts gRPC service."""
  # These are not available in TF 2.5.
  # options = tf.saved_model.LoadOptions(
  #     allow_partial_checkpoint=True, experimental_skip_checkpoint=True)
  start = time.time()
  model = tf.saved_model.load(FLAGS.model_path)

  server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=1),
      options=[
          ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH),
          ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH),
      ])
  service_pb2_grpc.add_DetectorServicer_to_server(Detector(model), server)
  logging.info('Model loading done in %.2fs. Inference server is ready.',
               time.time() - start)

  server.add_insecure_port('[::]:50051')
  server.start()
  server.wait_for_termination()


def main(unused_argv):
  tf.config.optimizer.set_jit(True)
  tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
  serve()


if __name__ == '__main__':
  app.run(main)
