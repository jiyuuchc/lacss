import tensorflow as tf
import tensorflow.keras.layers as layers

def proposal_locations(pred, max_output_size=500, distance_threshold=3.0, topk=2000, score_threshold=None, padded=False):
    '''
    Produce a list of proposal locations based on predication map, remove redundency with non_max_suppression
    Args:
      pred: [batch_size, height, width] binary prediction for cell gt_locations
      max_output_size: int
      distance_threshold: non_max_suppression threshold
      topk: if not 0, only the topk pixels are analyzed
      score_thresold: default to None
      padded: whether output non_ragged tensor (padded with -1), default to False
    Returns:
      scores: [batch_size, None], Ragged or padded tensor sorted scores
      indices: [batch_size, None, 2], Ragged or padded tensor, candidate locations
    '''

    if score_threshold is None:
        score_threshold = float('-inf')

    batch_size, height, width, _ = tf.shape(pred)
    pred = tf.reshape(pred, [batch_size, -1])

    if topk > 0:
        scores, indices = tf.math.top_k(pred,topk)
    else:
        indices = tf.argsort(pred, direction='DESCENDING')
        scores = tf.gather(pred, indices, batch_dims=1)

    indices = tf.unravel_index(tf.reshape(indices, [-1]), [height, width])
    indices = tf.transpose(tf.reshape(indices, [batch_size, 2, -1]), [1,2,0])

    sqdist = tf.reduce_sum(tf.math.square(indices[:,None,:,:] - indices[:,:,None,:]), axis=-1)
    sqdist = tf.cast(sqdist, tf.float32)
    sq_th = distance_threshold * distance_threshold
    dist_matrix = tf.cast(tf.cast(sqdist, tf.float32) <= sq_th, tf.float32)

    # nms is developed for boxes (4d data) but this function works on overlap matrix directly so could also apply to
    # out case (2d data)
    def nms_one(element):
        score, dist, ind = element
        sel = tf.image.non_max_suppression_overlaps(dist, score, max_output_size, score_threshold=score_threshold)
        score_out = tf.gather_nd(score, sel[...,None])
        ind_out = tf.gather_nd(ind, sel[..., None])

        return score_out, ind_out

    scores, indices =  tf.map_fn(
        nms_one, (scores, dist_matrix, indices),
        fn_output_signature = (tf.RaggedTensorSpec([None], scores.dtype, 0), tf.RaggedTensorSpec([None, 2], indices.dtype, 0)),
    )

    if padded:
        scores = scores.to_tensor(-1)
        indices = indices.to_tensor(-1)

    return scores, indices
