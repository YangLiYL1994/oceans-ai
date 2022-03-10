"""Library to track COTS using detected bounding boxes."""

import abc
import cv2
import numpy as np
import service_pb2

from typing import Any, Dict
from collections import defaultdict
from object_detection.utils import np_box_ops

class Detection(abc.ABC):

  def __init__(self, bounding_box, file_name, is_detected=True):
    self.bounding_box = bounding_box
    self.is_detected = is_detected
    self.file_name = file_name

class RecentFrames(abc.ABC):
  def __init__(self, max_frames):
    self.frames = []
    self.max_frames = max_frames

  def insert_frame(self, frame_id):
    if (len(self.frames)) > self.max_frames:
      # Remove the oldest entry.
      self.frames.pop(0)
    self.frames.append(frame_id)

  def get_frames(self):
      return self.frames

  def get_max_frames(self):
    return self.max_frames


class CotsTracker(abc.ABC):
  
  def __init__(self):
    # See get_tracked_persons() for the format of the contents of this
    # dictionary.
    self.sequence_id_to_bbox = defaultdict(lambda: [])
    self.sequence_id = 0
    # Map from sequence id to number of objects linked to the sequence.
    self.sequence_id_to_length = defaultdict(int)
    self.min_iou = 0.4
    # Keep track of last n processed frames.
    self.recent_frames = RecentFrames(5)
    self.reference_image = None
    self.flow = None
    self.image_width = 0
    self.image_height = 0

  def _increment_sequence_length(self, sequence_id):
    self.sequence_id_to_length[sequence_id] += 1

  def _generate_next_sequence_id(self):
  # Returns an ID for newly tracked person."""
    sequence_id = self.sequence_id
    self.sequence_id += 1
    return sequence_id

  def get_iou(self, ground_truth, prediction):
    return np_box_ops.iou(ground_truth, prediction)

  def _track_object(self, bounding_box):
    # Returns existing sequence id if object is tracked else returns -1
    for sequence_id, sequence_list in self.sequence_id_to_bbox.items():
      previous_bbox = sequence_list[-1].bounding_box
      left = np.reshape(previous_bbox, (1, 4))
      right = np.reshape(bounding_box, (1, 4))
      if self.get_iou(left, right) > self.min_iou:
        return sequence_id
    return -1

  def remove_obsolete_sequences(self):
    if (len(self.recent_frames.get_frames())
        < self.recent_frames.get_max_frames()):
      # Not enough frames have been processed.
      return
    recent_frames = set(self.recent_frames.get_frames())
    sequence_ids_to_delete = []
    for sequence_id, sequence_list in self.sequence_id_to_bbox.items():
      recent_frame_matched_to_sequence = False
      # Process detections in reverse order - most recent to oldest
      for entry in reversed(sequence_list):
        if entry.file_name in recent_frames and entry.is_detected:
          recent_frame_matched_to_sequence = True
          break
      # No detections found for recent frames - stop propagating by deleting
      # this sequence.
      if not recent_frame_matched_to_sequence:
        sequence_ids_to_delete.append(sequence_id)
    for sequence_id in sequence_ids_to_delete:
      print('Deleting sequence id:', sequence_id)
      del self.sequence_id_to_bbox[sequence_id]
      del self.sequence_id_to_length[sequence_id]

  def get_projected_estimate_new_bounding_box(self, bbox, filename):
    bbox_as_list = bbox.bounding_box.tolist();
    point_y1 = bbox_as_list[0]
    point_x1 = bbox_as_list[1]
    point_y2 = bbox_as_list[2]
    point_x2 = bbox_as_list[3]

    fy1 = int(np.rint(np.clip(point_y1, 0, self.flow.shape[1])))
    fx1 = int(np.rint(np.clip(point_x1, 0, self.flow.shape[0])))
    fy2 = int(np.rint(np.clip(point_y2, 0, self.flow.shape[1])))
    fx2 = int(np.rint(np.clip(point_x2, 0, self.flow.shape[0])))

    point_y1 = point_y1 + abs(self.flow[fy1, fx1, 0])
    point_x1 = point_x1 + abs(self.flow[fy1, fx1, 1])
    point_y2 = point_y2 + abs(self.flow[fy2, fx2, 0])
    point_x2 = point_x2 + abs(self.flow[fy2, fx2, 1])

    # Prevent projected bbox from going outside image corners.
    point_y1 = np.clip(point_y1, 0, self.image_height)
    point_x1 = np.clip(point_x1, 0, self.image_width)
    point_y2 = np.clip(point_y2, 0, self.image_height)
    point_x2 = np.clip(point_x2, 0, self.image_width)

    projected_bbox = np.array([point_y1, point_x1, point_y2, point_x2])
    return Detection(projected_bbox, file_name=filename, is_detected=False)

  def propagate_previous_detections(self, filename):
    if not self.sequence_id_to_bbox:
      return
    id_to_projected_box = {}
    for sequece_id, detections in self.sequence_id_to_bbox.items():
      estimated_box = self.get_projected_estimate_new_bounding_box(
          detections[-1], filename)
      id_to_projected_box[sequece_id] = estimated_box
    for sequece_id in self.sequence_id_to_bbox.keys():
      self.sequence_id_to_bbox[sequece_id].append(
          id_to_projected_box[sequece_id])
    return

  def process_frame(self, filename, detections, image, image_height,
                    image_width):
    self.remove_obsolete_sequences()
    self.recent_frames.insert_frame(filename)
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if self.reference_image is not None:
      self.flow = cv2.calcOpticalFlowFarneback(prev=self.reference_image,
                                               next=frame_gray, flow=self.flow,
                                               pyr_scale=0.8, levels=3,
                                               winsize=15,
                                               iterations=3, poly_n=5,
                                               poly_sigma=1.2, flags=0)
      self.propagate_previous_detections(filename)
    results = service_pb2.TrackerResults()
    results.file_path = filename
    if not detections:
      return results
    # Store mapping from sequence id to bounding boxes tracked in a single
    # image. Assumption: Two detections within the same file cannot belong to
    # the same sequence. Merge sequences detected in a file after tracking for
    # objects detected in a image is complete.
    new_sequence_id_to_bbox = {}
    existing_sequence_id_to_bbox = {}

    for detection in detections:
      tracker_result = results.tracker_results.add()
      tracker_result.detection.CopyFrom(detection)
      bounding_box = np.array([detection.top, detection.left,
                               detection.top + detection.height,
                               detection.left + detection.width])
      current_detection = Detection(bounding_box, filename, True)
      detection_sequence_id = self._track_object(bounding_box)
      if detection_sequence_id == -1:
        # Unable to link detection to existing sequence.
        # Generate new sequence id and append detection.
        detection_sequence_id = self._generate_next_sequence_id()
        new_sequence_id_to_bbox[detection_sequence_id] = current_detection
      else:
        existing_sequence_id_to_bbox[detection_sequence_id] = current_detection
      tracker_result.sequence_id = detection_sequence_id
      self.sequence_id_to_length[detection_sequence_id] += 1
      tracker_result.sequence_length = (
          self.sequence_id_to_length[detection_sequence_id])

    # Merge detections from this file into object map.
    for key, value in new_sequence_id_to_bbox.items():
      self.sequence_id_to_bbox[key].append(value)
    for key, value in existing_sequence_id_to_bbox.items():
      # Detection linked to an existing sequence. Replace last projected
      # detection for identified sequence.
      existing_detections = self.sequence_id_to_bbox[key]
      existing_detections[-1] = value
      self.sequence_id_to_bbox[key] = existing_detections
    self.reference_image = frame_gray
    self.image_height = image_height
    self.image_width = image_width
    return results
