import tensorflow as tf

''' Loss Function Definition'''


'''Dice Loss, depth 3, Check'''


def dice_loss_function(prediction, ground_truth):
    ground_truth = tf.one_hot(indices=ground_truth, depth=3)
    dice = 0
    for i in range(3):
        # reduce_mean calculation
        intersection = tf.reduce_mean(prediction[:, :, :, :, i] * ground_truth[:, :, :, :, i])
        union_prediction = tf.reduce_sum(prediction[:, :, :, :, i] * prediction[:, :, :, :, i])
        union_ground_truth = tf.reduce_sum(ground_truth[:, :, :, :, i] * ground_truth[:, :, :, :, i])
        union = union_ground_truth + union_prediction
        dice = dice + 2 * intersection / union
    return -dice


'''SoftMax Loss, Check'''


def softmax_loss_function(logits, labels):
    # loss = weighted * - target * log(softmax(logits))
    # weighted?
    prediction = logits
    softmax_prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=labels, depth=3)
    loss = 0
    for i in range(3):
        class_i_ground_truth = ground_truth[:, :, :, :, i]
        class_i_prediction = softmax_prediction[:, :, :, :, i]
        weighted = 1 - (tf.reduce_sum(class_i_ground_truth) / tf.reduce_sum(ground_truth))
        loss = loss - tf.reduce_mean(weighted * class_i_ground_truth * tf.log(
            tf.clip_by_value(t=class_i_prediction, clip_value_min=0.005, clip_value_max=1)))
        # Clips tensor values to a specified min and max.
    return loss
