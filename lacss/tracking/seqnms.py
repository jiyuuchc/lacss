# -*- coding: utf-8 -*-
# Slightly modified from https://github.com/tmoopenn/seq-nms

import copy

import numpy as np

from lacss.ops import box_intersection

__ALL__=["seq_nms"]

# from compute_overlap import compute_overlap_areas_given, compute_area
def compute_area(boxes):
    return (boxes[:, 2] - boxes[:,0]) * (boxes[:, 3] - boxes[:,1])

def compute_overlap_areas_given(boxes1, boxes2, area2):
    its = box_intersection(boxes1, boxes2)
    area1 = compute_area(boxes1)
    # area2 = compute_area(boxes2)
    iou = its / (area1[:, None] + area2 - its + 1e-8)
    return iou

def seq_nms(boxes, scores, labels=[], linkage_threshold=0.6, nms_threshold=0.4, score_metric='avg'):
    ''' Filter detections using the seq-nms algorithm. Boxes and classifications should be organized sequentially along the first dimension 
    corresponding to the input frame.  
    Args 
        boxes                 : Tensor of shape (num_frames, num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        scores                : Tensor of shape (num_frames, num_boxes) containing the confidence score for each box.
        linkage_threshold     : Threshold used to link two boxes in adjacent frames 
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed with regards to a best sequence.
    '''
    # use filtered boxes and scores to create nms graph across frames 
    box_graph = build_box_sequences(boxes, scores, labels, linkage_threshold=linkage_threshold)
    print("BOX GRAPH SHAPE", box_graph.shape)
    _seq_nms(box_graph, boxes, scores, nms_threshold, score_metric=score_metric)

def build_box_sequences(boxes, scores, labels=[], linkage_threshold=0.5):
    ''' Build bounding box sequences across frames. A sequence is a set of boxes that are linked in a video
    where we define a linkage as boxes in adjacent frames (of the same class) with IoU above linkage_threshold (0.5 by default).
    Args
        boxes                  : Tensor of shape (num_frames, num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format. 
        scores                : Tensor of shape (num_frames, num_boxes) containing the confidence score for each box.
        linkage_threshold      : Threshold for the IoU value to determine if two boxes in neighboring frames are linked 
    Returns 
        A list of shape (num_frames - 1, num_boxes, k, 1) where k is the number of edges to boxes in neighboring frame (s.t. 0 <= k <= num_boxes at f+1)
        and last dimension gives the index of that neighboring box. 
    '''
    box_graph = []
    # iterate over neighboring frames 
    for f in range(boxes.shape[0] - 1):
        boxes_f, scores_f = boxes[f,:,:], scores[f,:]
        boxes_f1, scores_f1 = boxes[f+1,:,:], scores[f+1,:]
        if f == 0:
            areas_f = compute_area(boxes_f.astype(np.double)) #(boxes_f[:,2] - boxes_f[:,0] + 1) * (boxes_f[:,3] - boxes_f[:,1] + 1)
        else: 
            areas_f = areas_f1

        # calculate areas for boxes in next frame
        areas_f1 = compute_area(boxes_f1.astype(np.double)) #(boxes_f1[:,2] - boxes_f1[:,0] + 1) * (boxes_f1[:,3] - boxes_f1[:,1] + 1)

        adjacency_matrix = []
        for i, box in enumerate(boxes_f):
            overlaps = compute_overlap_areas_given(np.expand_dims(box,axis=0).astype(np.double), boxes_f1.astype(np.double), areas_f1.astype(np.double) )[0]

            # add linkage if IoU greater than threshold and boxes have same labels i.e class  
            if len(labels) == 0 :
                edges = [ovr_idx for ovr_idx, IoU in enumerate(overlaps) if IoU >= linkage_threshold]
            else:
                edges = [ovr_idx for ovr_idx, IoU in enumerate(overlaps) if IoU >= linkage_threshold and labels[f,i] == labels[f+1,ovr_idx]]
            adjacency_matrix.append(edges)
        box_graph.append(adjacency_matrix)
    return np.array(box_graph, dtype="object")

def find_best_sequence(box_graph, scores):
    ''' Given graph of all linked boxes, find the best sequence in the graph. The best sequence 
    is defined as the sequence with the maximum score across an arbitrary number of frames.
    We build the sequences back to front from the last frame to easily capture start of new sequences/
    Condition to start of new sequence: 
        if there are no edges from boxes in frame t-1, then the box in frame t must be the start of a new sequence.
        This assumption is valid since all scores are positive so we can always improve a sequence by increasing its length. 
        Therefore if there are links to a box from previous frames, we can always build a better path by extending it s.t. 
        the box cannot be the start of a new best sequence. 
    Args
        box_graph             : list of shape (num_frames - 1, num_boxes, k) returned from build_box_sequences that contains box sequences 
        scores                : Tensor of shape (num_frames, num_boxes) containing the label for the corresponding box. 
    Returns 
        None
    '''
    # list of tuples storing (score up to current frame, path up to current frame)
    # we dynamically build up best paths through graph starting from the end frame
    # s.t we can determine the beginning of sequences i.e. if there are no links 
    # to a box from previous frames, then it is a candidate for starting a sequence 
    max_scores_paths = [] 

    # list of all independent sequences where a given row corresponds to starting frame
    sequence_roots = []

    # starting from the last frame, build base paths i.e paths consisting of a single node 
    max_scores_paths.append([(score, [idx]) for idx, score in enumerate(scores[-1])])

    for reverse_idx, frame_edges in enumerate(box_graph[::-1]): # list of edges between neigboring frames i.e frame dimension 
        max_paths_f = []
        used_in_sequence = np.zeros(len(max_scores_paths[-1]), int)
        frame_idx = len(box_graph) - reverse_idx - 1
        for box_idx, box_edges in enumerate(frame_edges): # list of edges for each box in frame i.e. box dimension
            if not box_edges: # no edges for current box so consider it a max path consisting of a single node 
                max_paths_f.append((scores[frame_idx][box_idx], [box_idx]))
            else: # extend previous max paths 
                # here we use box_edges list to index used_in_sequence list and mark boxes in corresponding frame t+1 
                # as part of a sequence since we have links to them and can always make a better max path by making it longer (no negative scores)
                used_in_sequence[box_edges] = 1
                prev_idx = np.argmax([max_scores_paths[-1][bidx][0] for bidx in box_edges])
                score_so_far = max_scores_paths[-1][box_edges[prev_idx]][0]
                path_so_far = copy.copy(max_scores_paths[-1][box_edges[prev_idx]][1])
                path_so_far.append(box_idx)
                max_paths_f.append((scores[frame_idx][box_idx] + score_so_far, path_so_far))
        
        # create new sequence roots for boxes in frame at frame_idx + 1 that did not have links from boxes in frame_idx
        new_sequence_roots = [max_scores_paths[-1][idx] for idx, flag in enumerate(used_in_sequence) if flag == 0]

        sequence_roots.append(new_sequence_roots) 
        max_scores_paths.append(max_paths_f)
    
    # add sequences starting in begining frame as roots 
    sequence_roots.append(max_scores_paths[-1])

    # reverse sequence roots since built sequences from back to front 
    sequence_roots = sequence_roots[::-1]

    # iterate sequence roots to find sequence with max score 
    best_score = 0 
    best_sequence = [] 
    sequence_frame_index = 0
    for index, frame_sequences in enumerate(sequence_roots):
        if not frame_sequences: continue 
        max_index = np.argmax([sequence[0] for sequence in frame_sequences])
        if frame_sequences[max_index][0] > best_score:
            best_score = frame_sequences[max_index][0]
            best_sequence = frame_sequences[max_index][1][::-1] # reverse path 
            sequence_frame_index = index
    return sequence_frame_index, best_sequence, best_score


def rescore_sequence(sequence, scores, sequence_frame_index, max_sum, score_metric='avg'):
    ''' Given a sequence, rescore the confidence scores according to the score_metric.
    Args
        sequence                    : The best sequence containing indices of boxes 
        scores                      : Tensor of shape (num_frames, num_boxes) containing the label for the corresponding box. 
        sequence_frame_index        : The index of the frame where the best sequence begins 
        best_score                  : The summed score of boxes in the sequence 
    Returns 
        None   
    '''
    if score_metric == 'avg':
        avg_score=max_sum/len(sequence)
        for i,box_ind in enumerate(sequence):
            scores[sequence_frame_index+i][box_ind]= avg_score
    elif score_metric == 'max':
        max_score = 0.0
        for i, box_ind in enumerate(sequence):
            if scores[sequence_frame_index + i][box_ind] > max_score: max_score = scores[sequence_frame_index+i][box_ind]
        for i, box_ind in enumerate(sequence):
            scores[sequence_frame_index + i][box_ind] = max_score
    else:
        raise ValueError("Invalid score metric")

def delete_sequence(sequence_to_delete, sequence_frame_index, scores, boxes, box_graph, suppress_threshold=0.3):
    ''' Given a sequence, remove its connections in box graph (create graph of linked boxes across frames).
    Args
        sequence_to_delete          : The best sequence containing indices of boxes to be deleted
        sequence_frame_index        : The index of the frame where the best sequence begins 
        scores                      : Tensor of shape (num_frames, num_boxes) containing the label for the corresponding box. 
        boxes                       : Tensor of shape (num_frames, num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        box_graph                   : list of shape (num_frames - 1, num_boxes, k) returned from build_box_sequences that contains box sequences 
        suppress_threshold          : Threshold for suprresing boxes that have an IoU with sequence boxes greater than the threshold 
    Returns 
        None  
    '''
    for i,box_idx in enumerate(sequence_to_delete):
        other_boxes = boxes[sequence_frame_index + i]
        box_areas = compute_area(other_boxes.astype(np.double))
        seq_box_area = box_areas[box_idx]
        seq_box = boxes[sequence_frame_index+i][box_idx]

        overlaps = compute_overlap_areas_given(np.expand_dims(seq_box,axis=0).astype(np.double), boxes[sequence_frame_index+i,:].astype(np.double), box_areas.astype(np.double))[0]
        deletes=[ovr_idx for ovr_idx,IoU in enumerate(overlaps) if IoU >= suppress_threshold]

        if sequence_frame_index + i < len(box_graph): 
            
            for delete_idx in deletes:
                box_graph[sequence_frame_index+i][delete_idx]=[]

        if i > 0 or sequence_frame_index > 0:
            # remove connections to current sequence node from previous frame nodes
            for prior_box in box_graph[sequence_frame_index+i-1]: 
                for delete_idx in deletes:
                    if delete_idx in prior_box:
                        prior_box.remove(delete_idx)
    
        deletes.remove(box_idx)
        scores[sequence_frame_index+i, deletes] = -1


def _seq_nms(box_graph, boxes, scores, nms_threshold, score_metric='avg'):
    ''' Iteratively executes the seq-nms algorithm given a box graph.
    Args
        box_graph                   : list of shape (num_frames - 1, num_boxes, k) returned from build_box_sequences that contains box sequences 
        boxes                       : Tensor of shape (num_frames, num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        scores                      : Tensor of shape (num_frames, num_boxes) containing the label for the corresponding box. 
        nms_threshold               : Threshold for the IoU value to determine when a box should be suppressed with regards to a best sequence.
    Returns 
        None
    '''
    while True: 
        sequence_frame_index, best_sequence, best_score = find_best_sequence(box_graph, scores)
        # print(best_sequence, best_score)
        if len(best_sequence) <= 1:
            break 
        rescore_sequence(best_sequence, scores, sequence_frame_index, best_score, score_metric=score_metric)
        delete_sequence(best_sequence, sequence_frame_index, scores, boxes, box_graph, suppress_threshold=nms_threshold)
