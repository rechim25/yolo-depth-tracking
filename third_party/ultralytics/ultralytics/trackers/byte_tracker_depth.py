import torch
import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYZAH

from depth_anything_v2.dpt import DepthAnythingV2


class STrack(BaseTrack):
    """
    Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYZAH): Shared Kalman filter that is used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYZAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.
        depth (float): Depth value of the object.

    Methods:
        predict(): Predict the next state of the object using Kalman filter.
        multi_predict(stracks): Predict the next states for multiple tracks.
        multi_gmc(stracks, H): Update multiple track states using a homography matrix.
        activate(kalman_filter, frame_id): Activate a new tracklet.
        re_activate(new_track, frame_id, new_id): Reactivate a previously lost tracklet.
        update(new_track, frame_id): Update the state of a matched track.
        convert_coords(tlwh): Convert bounding box to x-y-z-aspect-height format.
        tlwh_to_xyzah(tlwh, depth): Convert tlwh bounding box to xyzah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person", depth=5.0)
        >>> track.activate(kalman_filter=KalmanFilterXYZAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYZAH()

    def __init__(self, xywh, score, cls, depth):
        """
        Initialize a new STrack instance.

        Args:
            xywh (List[float]): Bounding box coordinates and dimensions in the format (x, y, w, h, [a], idx), where
                (x, y) is the center, (w, h) are width and height, [a] is optional aspect ratio, and idx is the id.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.
            depth (float): Depth value of the detection.

        Examples:
            >>> xywh = [100.0, 150.0, 50.0, 75.0, 1]
            >>> score = 0.9
            >>> cls = "person"
            >>> depth = 5.0
            >>> track = STrack(xywh, score, cls, depth)
        """
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None
        self.depth = depth  # Add depth attribute

    def predict(self):
        """Predicts the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[9] = 0  # Adjusted for the new state vector size
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][9] = 0  # Adjusted for the new state vector size
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R10x10 = np.kron(np.eye(5, dtype=float), R)  # Adjusted for the new state vector size
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R10x10.dot(mean)
                mean[:2] += t
                cov = R10x10.dot(cov).dot(R10x10.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a previously lost track using new detection data and updates its state and attributes."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx
        self.depth = new_track.depth  # Update depth

    def update(self, new_track, frame_id):
        """
        Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        Examples:
            Update the state of a track with new detection information
            >>> track = STrack([100, 200, 50, 80, 0.9, 1], depth=5.0)
            >>> new_track = STrack([105, 205, 55, 85, 0.95, 1], depth=5.1)
            >>> track.update(new_track, 2)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx
        self.depth = new_track.depth  # Update depth

    def convert_coords(self, tlwh):
        """Convert a bounding box's top-left-width-height format to its x-y-z-aspect-height equivalent."""
        return self.tlwh_to_xyzah(tlwh, self.depth)

    @property
    def tlwh(self):
        """Returns the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:5].copy()  # Include depth
        x, y, z, a, h = ret
        w = a * h  # Width is aspect ratio times height
        tl_x = x - w / 2
        tl_y = y - h / 2
        return np.array([tl_x, tl_y, w, h])


    @property
    def xyxy(self):
        """Converts bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyzah(tlwh, depth):
        """Convert bounding box from tlwh format to center-x-center-y-depth-aspect-height (xyzah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2  # Center x, y
        aspect_ratio = ret[2] / ret[3]
        return np.array([ret[0], ret[1], depth, aspect_ratio, ret[3]])

    @property
    def xyz(self):
        """Returns the current position in 3D space (x, y, depth)."""
        if self.mean is None:
            return np.array([self._tlwh[0] + self._tlwh[2] / 2, self._tlwh[1] + self._tlwh[3] / 2, self.depth])
        return self.mean[:3].copy()

    @property
    def xyzah(self):
        """Returns the current position in (center x, center y, depth, aspect_ratio, height) format."""
        if self.mean is None:
            return self.tlwh_to_xyzah(self._tlwh, self.depth)
        return self.mean[:5].copy()

    @property
    def xywh(self):
        """Returns the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """Returns position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning("WARNING ⚠️ angle attr not found, returning xywh instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """Returns the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """Returns a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETrackerDepth:
    """
    BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    Responsible for initializing, updating, and managing the tracks for detected objects in a video sequence.
    It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman filtering for predicting
    the new object locations, and performs data association.

    Attributes:
        tracked_stracks (List[STrack]): List of successfully activated tracks.
        lost_stracks (List[STrack]): List of lost tracks.
        removed_stracks (List[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYZAH): Kalman Filter object.
        lambda_depth (float): Weight for depth difference in the distance metric.

    Methods:
        update(results, img=None, depth_map=None): Updates object tracker with new detections.
        get_kalmanfilter(): Returns a Kalman filter object for tracking bounding boxes.
        init_track(dets, scores, cls, depths, img=None): Initialize object tracking with detections.
        get_dists(tracks, detections): Calculates the distance between tracks and detections.
        multi_predict(tracks): Predicts the location of tracks.
        reset_id(): Resets the ID counter of STrack.
        joint_stracks(tlista, tlistb): Combines two lists of stracks.
        sub_stracks(tlista, tlistb): Filters out the stracks present in the second list from the first list.
        remove_duplicate_stracks(stracksa, stracksb): Removes duplicate stracks based on IoU.

    Examples:
        Initialize BYTETracker and update with detection results
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> depth_map = depth_model.predict(image)
        >>> tracked_objects = tracker.update(results, img=image, depth_map=depth_map)
    """

    def __init__(self, args, frame_rate=30):
        """
        Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.

        Examples:
            Initialize BYTETracker with command-line arguments and a frame rate of 30
            >>> args = Namespace(track_buffer=30, lambda_depth=1.0)
            >>> tracker = BYTETracker(args, frame_rate=30)
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
        self.lambda_depth = getattr(self.args, 'lambda_depth', 1.0)  # Depth weight

        # Initialize the DepthAnythingV2 model
        self.device = args.device if args.device else "cuda"
        self.depth_model = self._initialize_depth_model()

    def _initialize_depth_model(self):
        """
        Initializes the DepthAnythingV2 model.

        Returns:
            depth_model: An instance of DepthAnythingV2 loaded onto the specified device.
        """
        # Configure the model based on your requirements
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder_type = getattr(self.args, 'encoder', 'vits')  # Default to 'vitl' if not specified
        model_config = model_configs.get(encoder_type)

        depth_model = DepthAnythingV2(**model_config)
        depth_model.load_state_dict(torch.load(
            getattr(self.args, 'ckpt_path', ''), map_location='cpu'))
        depth_model = depth_model.to(self.device).eval()
        return depth_model

    def update(self, results, img=None, depth_map=None):
        """Updates the tracker with new detections and returns the current list of tracked objects."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        # Run depth inference on the image
        depth_map = self.depth_model.infer_image(img, input_size=getattr(self.args, 'input_size', 518))
        depth_map = depth_map.squeeze()  # Remove batch dimension if present
        depth_map = depth_map.cpu().numpy() if isinstance(depth_map, torch.Tensor) else depth_map
        # Do not normalize the depth map to maintain consistency 

        # Extract depth values for all detections
        depth_values = []
        for bbox in bboxes:
            x_center, y_center, w, h, idx = bbox  # assuming bbox has 5 elements
            x1 = int(x_center - w / 4)
            y1 = int(y_center - h / 4)
            x2 = int(x_center + w / 4)
            y2 = int(y_center + h / 4)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(depth_map.shape[1] - 1, x2)
            y2 = min(depth_map.shape[0] - 1, y2)

            depth_values_in_bbox = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(depth_values_in_bbox)
            # avg_depth = depth_map[int(y_center), int(x_center)]
            depth_values.append(avg_depth)

        depth_values = np.array(depth_values)

        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]
        depth_values_keep = depth_values[remain_inds]
        depth_values_second = depth_values[inds_second]

        detections = self.init_track(dets, scores_keep, cls_keep, depth_values_keep, img)
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
        detections_second = self.init_track(dets_second, scores_second, cls_second, depth_values_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # TODO
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self):
        """Returns a Kalman filter object for tracking bounding boxes using KalmanFilterXYZAH."""
        return KalmanFilterXYZAH()  # Modified Kalman filter to include depth

    def init_track(self, dets, scores, cls, depths, img=None):
        """Initializes object tracking with given detections, scores, class labels, and depth values using the STrack algorithm."""
        return [STrack(xyxy, s, c, d) for (xyxy, s, c, d) in zip(dets, scores, cls, depths)] if len(dets) else []  # detections

    def get_dists(self, tracks, detections):
        """Calculates the distance between tracks and detections using IoU and depth difference."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        iou_dists = matching.iou_distance(tracks, detections)
        track_depths = np.array([track.depth for track in tracks])
        det_depths = np.array([det.depth for det in detections])
        depth_diffs = np.abs(track_depths[:, np.newaxis] - det_depths[np.newaxis, :])

        # Normalize depth differences
        max_depth_diff = np.max(depth_diffs) if np.max(depth_diffs) > 0 else 1.0
        depth_diffs_norm = depth_diffs / max_depth_diff

        # Combine distances
        dists = iou_dists + self.lambda_depth * depth_diffs_norm
        print(f"iou_dists: {iou_dists}")
        print(f"lambda: {self.lambda_depth}")
        print(f"depth_diffs_norm: {depth_diffs_norm}")
        print(f"max_depth_diff: {max_depth_diff}")
        print(f"track_depths: {track_depths}")
        return dists

    def multi_predict(self, tracks):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Resets the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Resets the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """Combines two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """Filters out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """Removes duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
