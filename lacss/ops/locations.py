import tensorflow as tf
import tensorflow.keras.layers as layers

def topk_proposals(scores, regressions=None, topk=2000, scale_factor=8):
    batch_size = tf.shape(scores)[0]
    height = tf.shape(scores)[1]
    width = tf.shape(scores)[2]
    pred = tf.reshape(scores, [batch_size, -1])

    if topk >=  height*width:
        topk = height*width
    scores, indices = tf.math.top_k(pred,topk)
    indices = tf.unravel_index(tf.reshape(indices, [-1]), [height, width])
    indices = tf.transpose(indices)
    indices = tf.reshape(indices, [batch_size, -1, 2])

    proposed_locations = tf.cast(indices, tf.float32) * scale_factor + scale_factor / 2

    if regression is not None:
        proposed_locations += tf.gather_nd(regressions, indices, batch_dims=1)

def proposal_locations(model_out, max_output_size=500, distance_threshold=1.01, topk=2000, score_threshold=None, padded=False):
    '''
    Produce a list of proposal locations based on predication map, remove redundency with non_max_suppression
    Args:
      model_out: (score_out, regression_out)
          score_out: [batch_size, height, width, 1] score prediction for cell gt_locations
          regression_out: [batch_size, height, width, 2] regression prediction
      max_output_size: int
      distance_threshold: non_max_suppression threshold
      topk: if not 0, only the topk pixels are analyzed
      score_thresold: default to None
      padded: whether output non_ragged tensor (padded with -1), default to False
    Returns:
      scores: [batch_size, None], Ragged or padded tensor sorted scores
      locations: [batch_size, None, 2], Ragged or padded tensor, candidate locations
    '''

    if score_threshold is None:
        score_threshold = float('-inf')

    score_out, regression_out = model_out
    batch_size = tf.shape(score_out)[0]
    height = tf.shape(score_out)[1]
    width = tf.shape(score_out)[2]
    score_out = tf.reshape(score_out, [batch_size, -1])

    if topk < 0 or topk > height*width:
        topk = height*width
    scores, indices = tf.math.top_k(score_out, topk)

    indices = tf.unravel_index(tf.reshape(indices, [-1]), [height, width])
    indices = tf.transpose(indices)
    indices = tf.reshape(indices, [batch_size, -1, 2])
    locations = tf.cast(indices, tf.float32) + 0.5
    regressions = tf.gather_nd(regression_out, indices, batch_dims=1)
    locations = locations + regressions

    sqdist = tf.reduce_sum(tf.math.square(locations[:,None,:,:] - locations[:,:,None,:]), axis=-1)
    sq_th = distance_threshold * distance_threshold
    dist_matrix = tf.cast(tf.cast(sqdist, tf.float32) <= sq_th, tf.float32)

    # nms is developed for boxes (4d data) but this function works on overlap matrix directly so could also apply to
    # out case (2d data)
    def nms_one(element):
        score, dist, loc = element
        sel = tf.image.non_max_suppression_overlaps(dist, score, max_output_size, score_threshold=score_threshold)
        score_out = tf.gather(score, sel)
        ind_out = tf.gather(loc, sel)

        return score_out, ind_out

    nms_scores, nms_locations =  tf.map_fn(
        nms_one, (scores, dist_matrix, locations),
        fn_output_signature = (tf.RaggedTensorSpec([None], scores.dtype, 0), tf.RaggedTensorSpec([None, 2], locations.dtype, 0)),
    )

    if padded:
        nms_scores = nms_scores.to_tensor(-1)
        nms_indices = nms_locations.to_tensor(-1)

    return nms_scores, nms_locations
