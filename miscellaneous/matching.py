import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def ious_old(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    #print("atlbrs: " + str(atlbrs))
    #print("btlbrs: " + str(btlbrs))
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious



distortion_coeffs = np.array([0.55678373, -0.57769629, 0.05860027, 0.05745619], dtype=np.float32)
intrinsic_matrix = np.array([[296.8978837, 0., 640.08274749], [0., 314.52113708, 399.7948668], [0., 0., 1.]], dtype=np.float32)


def undistort_and_project_point(distortion_coeffs, intrinsic_matrix, point):
    # Create an undistorted intrinsic matrix
    K_undistorted = intrinsic_matrix.copy()
    K_undistorted[(0, 1), (0, 1)] = 0.1 * K_undistorted[(0, 1), (0, 1)]

    # Convert the input point to np.float32
    point = np.array([point], dtype=np.float32)

    # Ensure the point is in the correct shape for fisheye undistortPoints
    point = point.reshape(-1, 1, 2)

    # Undistort the point
    undistorted_points = cv2.fisheye.undistortPoints(point, intrinsic_matrix, distortion_coeffs, P=K_undistorted)

    return undistorted_points[0][0].astype(float)

def find_missing_corners(top_left, bottom_right):
    if len(top_left) != 2 or len(bottom_right) != 2:
        return "Invalid input format. Please provide top-left and bottom-right points as arrays of length 2."
    
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    top_right_corner = []
    bottom_left_corner = []
    
    if x1 != x2 and y1 != y2:
        top_right_corner = [x2, y1]
        bottom_left_corner = [x1, y2]
    else:
        return "The provided points do not form a rectangle."
    
    return top_right_corner, bottom_left_corner

def get_middle_points(top_left_corner, bottom_left_corner, bottom_right_corner, top_right_corner):
    if len(top_left_corner) != 2 or len(bottom_left_corner) != 2 or len(bottom_right_corner) != 2 or len(top_right_corner) != 2:
        return "Invalid input format. Please provide four corners as arrays of length 2."
    
    x1, y1 = top_left_corner
    x2, y2 = bottom_left_corner
    x3, y3 = bottom_right_corner
    x4, y4 = top_right_corner
    
    middle_point_top = [(x1 + x4) / 2, (y1 + y4) / 2]
    middle_point_left = [(x2 + x1) / 2, (y2 + y1) / 2]
    middle_point_bottom = [(x3 + x2) / 2, (y3 + y2) / 2]
    middle_point_right = [(x4 + x3) / 2, (y4 + y3) / 2]
    
    return middle_point_top, middle_point_left, middle_point_bottom, middle_point_right

def get_rectangle_corners(middle_point_top, middle_point_left, middle_point_bottom, middle_point_right):
    top_left_x = min(middle_point_top[0], middle_point_left[0], middle_point_bottom[0], middle_point_right[0])
    top_left_y = min(middle_point_top[1], middle_point_left[1], middle_point_bottom[1], middle_point_right[1])
    bottom_right_x = max(middle_point_top[0], middle_point_left[0], middle_point_bottom[0], middle_point_right[0])
    bottom_right_y = max(middle_point_top[1], middle_point_left[1], middle_point_bottom[1], middle_point_right[1])
    
    top_left_corner = [top_left_x, top_left_y]
    bottom_right_corner = [bottom_right_x, bottom_right_y]
    
    return top_left_corner, bottom_right_corner

def fish_ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    fish_atlbrs=[]
    for a_bb in atlbrs:
      bb_tl = [a_bb[0],a_bb[1]]
      bb_br = [a_bb[2],a_bb[3]]
      bb_tr, bb_bl = find_missing_corners(bb_tl, bb_br)
      udis_bb_tl = undistort_and_project_point(distortion_coeffs, intrinsic_matrix, bb_tl)
      udis_bb_br = undistort_and_project_point(distortion_coeffs, intrinsic_matrix, bb_br)
      udis_bb_tr = undistort_and_project_point(distortion_coeffs, intrinsic_matrix, bb_tr)
      udis_bb_bl = undistort_and_project_point(distortion_coeffs, intrinsic_matrix, bb_bl)
      mp_top, mp_left, mp_bot, mp_right =  get_middle_points(udis_bb_tl, udis_bb_bl, udis_bb_br, udis_bb_tr)
      new_bb_tl,new_bb_br = get_rectangle_corners(mp_top, mp_left, mp_bot, mp_right)
      fish_bb = np.concatenate((new_bb_tl,new_bb_br),axis=None)
      #fish_bb = np.concatenate((udis_bb_tl, udis_bb_br), axis=None)
      fish_atlbrs.append(fish_bb)

    fish_btlbrs=[]
    for b_bb in btlbrs:
      bb_tl = [b_bb[0],b_bb[1]]
      bb_br = [b_bb[2],b_bb[3]]
      bb_tr, bb_bl = find_missing_corners(bb_tl, bb_br)
      udis_bb_tl = undistort_and_project_point(distortion_coeffs, intrinsic_matrix, bb_tl)
      udis_bb_br = undistort_and_project_point(distortion_coeffs, intrinsic_matrix, bb_br)
      udis_bb_tr = undistort_and_project_point(distortion_coeffs, intrinsic_matrix, bb_tr)
      udis_bb_bl = undistort_and_project_point(distortion_coeffs, intrinsic_matrix, bb_bl)
      mp_top, mp_left, mp_bot, mp_right =  get_middle_points(udis_bb_tl, udis_bb_bl, udis_bb_br, udis_bb_tr)
      new_bb_tl,new_bb_br = get_rectangle_corners(mp_top, mp_left, mp_bot, mp_right)
      fish_bb = np.concatenate((new_bb_tl,new_bb_br),axis=None)
      #fish_bb = np.concatenate((udis_bb_tl, udis_bb_br), axis=None)
      fish_btlbrs.append(fish_bb)
      
    ious = bbox_ious(
        np.ascontiguousarray(fish_atlbrs, dtype=float),
        np.ascontiguousarray(fish_btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    #print("atracks: " + str(atracks))
    #print("btracks: " + str(btracks))
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        print("Here, I am")
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        #print("atlbrs: " + str(atlbrs))
        #print("btlbrs: " + str(btlbrs))
    _ious = fish_ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost