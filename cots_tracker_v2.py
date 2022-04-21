import copy
import time

from absl import logging
import cv2
import numpy as np


class Tracklet():
    def __init__(self, timestamp, detection):
        self.timestamp = timestamp
        # Store a copy here to make sure the coordinates will not be updated
        # when the optical flow propagation runs using another reference to this
        # detection.
        self.detection = copy.deepcopy(detection)

    def __repr__(self):
        return f'Time {self.timestamp}, ' + self.detection.__repr__()


class Track():
    def __init__(self, id, detection):
        self.id = id
        self.linked_dets = []
        self.det = detection

    def __repr__(self):
        result = f'Track {self.id}'
        for linked_det in self.linked_dets:
            result += '\n' + linked_det.__repr__()
        return result


class OpticalFlowTracker():
    def __init__(self, tid, ft=3.0, iou=0.5, tt=2.0, bb=32, size=64, its=20,
                 eps=0.03, levels=3):
        self.track_id = tid
        # How long to apply optical flow tracking without getting positive 
        # detections (sec).
        self.track_flow_time = ft * 1000
        # Required IoU overlap to link a detection to a track.
        self.overlap_threshold = iou
        # Used to detect if detector needs to be reset.
        self.time_threshold = tt * 1000
        self.border = bb
        # Size of optical flow region.
        self.of_size = (size, size)
        self.of_criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, its, 
                            eps)
        self.of_levels= levels

        self.tracks = []
        self.prev_image = None
        self.prev_time = -1

    def update(self, image_bgr, detections, timestamp):
        start = time.time()
        num_optical_flow_calls = 0

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        image_w = image.shape[1]
        image_h = image.shape[0]

        # Assume tracker is invalid if too much time has passed!
        if (self.prev_time > 0 and 
                timestamp - self.prev_time > self.time_threshold):
            logging.warning(
                'Too much time since last update, resetting tracker.')
            self.tracks = []

        # Remove tracks which are:
        # - Touching the image edge.
        # - Have existed for a long time without linking a real detection.
        active_tracks = []
        for track in self.tracks:
            if (track.det.x0 < self.border or track.det.y0 < self.border or
                    track.det.x1 >= (image_w - self.border) or 
                    track.det.y1 >= (image_h - self.border)):
                logging.info(f'Removing track {track.id} because it\'s near the border')
                continue

            time_since_last_detection = timestamp - track.linked_dets[-1].timestamp
            if (time_since_last_detection > self.track_flow_time):
                logging.info(f'Removing track {track.id} because it\'s too old '
                             f'({time_since_last_detection:.02f}s)')
                continue

            active_tracks.append(track)
        
        self.tracks = active_tracks

        # Run optical flow to update existing tracks.
        if self.prev_time > 0:
            # print('Running optical flow propagation.')
            of_params = {
                'winSize': self.of_size,
                'maxLevel': self.of_levels,
                'criteria': self.of_criteria
            }
            for track in self.tracks:
                input_points = np.float32([[[(track.det.x0 + track.det.x1) / 2, 
                                             (track.det.y0 + track.det.y1) / 2]]])
                output_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_image, image, input_points, None, **of_params)
                num_optical_flow_calls += 1
                w = track.det.x1 - track.det.x0
                h = track.det.y1 - track.det.y0
                # print(f'Detection before flow update: {track.det}')
                track.det.x0 = output_points[0][0][0] - w * 0.5
                track.det.y0 = output_points[0][0][1] - h * 0.5
                track.det.x1 = output_points[0][0][0] + w * 0.5
                track.det.y1 = output_points[0][0][1] + h * 0.5
                # print(f'Detection after flow update: {track.det}')

        # Insert new detections.
        for detection in detections:
            if (detection.x0 < self.border or detection.y0 < self.border or
                    detection.x1 >= image_w - self.border or
                    detection.y1 >= image_h - self.border):
                # print('Skipping detection because it\'s close to the border.')
                continue

            # See if detection can be linked to an existing track.
            linked = False
            overlap_index = 0
            overlap_max = -1000
            for track_index, track in enumerate(self.tracks):
                # print(f'Testing track {track_index}')
                if track.det.class_id != detection.class_id:
                    continue
                overlap = detection.iou(track.det)
                if overlap > overlap_max:
                    overlap_index = track_index
                    overlap_max = overlap

            # Link to existing track with maximal IoU.
            if overlap_max > self.overlap_threshold:
                track = self.tracks[overlap_index]
                track.det = detection
                track.linked_dets.append(Tracklet(timestamp, detection))
                linked = True

            if not linked:
                logging.info(f'Creating new track with ID {self.track_id}')
                new_track = Track(self.track_id, detection)
                new_track.linked_dets.append(Tracklet(timestamp, detection))
                self.tracks.append(new_track)
                self.track_id += 1
        
        self.prev_image = image
        self.prev_time = timestamp

        if num_optical_flow_calls > 0:
            tracking_ms = int(1000 * (time.time() - start))
            logging.info(f'Tracking took {tracking_ms}ms, '
                         f'{num_optical_flow_calls} optical flow calls')

        return self.tracks
