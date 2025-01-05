import tensorflow as tf


class CustomSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, ignore_class=-1):
        super().__init__()
        self.from_logits = from_logits
        self.ignore_class = ignore_class

    def call(self, y_true, y_pred):
        # Ensure inputs are tensors
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        # Generate a mask that is False where y_true equals ignore_class and True elsewhere
        mask = tf.not_equal(y_true, self.ignore_class)

        # Use this mask to filter out ignored values from y_true and y_pred
        y_true_filtered = tf.boolean_mask(y_true, mask)
        y_pred_filtered = tf.boolean_mask(y_pred, mask)

        # Compute the sparse categorical crossentropy on filtered targets and predictions
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true_filtered, y_pred_filtered, from_logits=self.from_logits
        )

        # Return the mean loss value
        return tf.reduce_mean(loss)


def masked_loss(y_true, y_pred):
    """
    Calculate the masked sparse categorical cross-entropy loss.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    loss (tensor): Calculated loss.
    """

    # Calculate the loss for each item in the batch. Remember to pass the right arguments, as discussed above!
    loss_fn = CustomSparseCategoricalCrossentropy(from_logits=True, ignore_class=-1)
    # Use the previous defined function to compute the loss
    loss = loss_fn(y_true, y_pred)

    return loss


def masked_accuracy(y_true, y_pred):
    """
    Calculate masked accuracy for predicted labels.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    accuracy (tensor): Masked accuracy.
    """

    # Calculate the loss for each item in the batch.
    # We must always cast the tensors to the same type in order to use them in training. Since we will make divisions, it is safe to use tf.float32 data type.
    y_true = tf.cast(y_true, tf.float32)
    # Create the mask, i.e., the values that will be ignored
    mask = tf.not_equal(y_true, -1.0)

    mask = tf.cast(mask, tf.float32)

    # Perform argmax to get the predicted values
    y_pred_class = tf.math.argmax(y_pred, axis=-1)
    y_pred_class = tf.cast(y_pred_class, tf.float32)
    # Compare the true values with the predicted ones
    matches_true_pred = tf.equal(y_true, y_pred_class)
    matches_true_pred = tf.cast(matches_true_pred, tf.float32)
    # Multiply the acc tensor with the masks
    matches_true_pred *= mask

    # Compute masked accuracy (quotient between the total matches and the total valid values, i.e., the amount of non-masked values)
    masked_acc = tf.reduce_sum(matches_true_pred) / tf.reduce_sum(mask)

    return masked_acc


def entity_accuracy(y_true, y_pred):
    """
    Calculate the accuracy based on the entities. Which mean that correct `O` tags will not be taken into account.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    accuracy (tensor): Tag accuracy.
    """

    y_true = tf.cast(y_true, tf.float32)
    # We ignore the padding and the O tag
    mask = y_true > 0
    mask = tf.cast(mask, tf.float32)

    y_pred_class = tf.math.argmax(y_pred, axis=-1)
    y_pred_class = tf.cast(y_pred_class, tf.float32)

    matches_true_pred = tf.equal(y_true, y_pred_class)
    matches_true_pred = tf.cast(matches_true_pred, tf.float32)

    matches_true_pred *= mask

    masked_acc = tf.reduce_sum(matches_true_pred) / tf.reduce_sum(mask)

    return masked_acc
