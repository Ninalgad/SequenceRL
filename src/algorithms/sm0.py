import tensorflow as tf
from algorithm import Algorithm


TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(5, 10, 10, 2), dtype=tf.float32, name='board'),
    tf.TensorSpec(shape=(5, 178), dtype=tf.float32, name='vec'),
    tf.TensorSpec(shape=(5, 10, 10), dtype=tf.float32, name='action'),
    tf.TensorSpec(shape=(5, 1), dtype=tf.float32, name='tar')
]

POLICY_SIGNATURE = [
    tf.TensorSpec(shape=(None, 10, 10, 2), dtype=tf.float32, name='board'),
    tf.TensorSpec(shape=(None, 10, 10), dtype=tf.float32, name='action'),
    tf.TensorSpec(shape=(None, 178), dtype=tf.float32, name='vec')
]


def reg_loss_fun(y_true, y_pred):
    assert y_true.shape == y_pred.shape, (y_true.shape, y_pred.shape)
    return tf.keras.losses.MSE(y_true, y_pred)


def cat_loss_fun(y_true, y_pred, from_logits=False):
    # if not from_logits:
    #     assert (tf.reduce_min(y_pred).numpy() >=0) and (tf.reduce_max(y_pred).numpy() <= 1), y_pred
    assert y_true.shape == y_pred.shape, (y_true.shape, y_pred.shape)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    return cce(y_true, y_pred)


@tf.custom_gradient
def quantize_chance_logits(logits):
    chance_tar = tf.one_hot(tf.math.argmax(logits, axis=-1), tf.shape(logits)[-1])

    def straight_through_gradient(dy):
        # Pass the gradient through directly (straight-through)
        return dy
    return chance_tar, straight_through_gradient


class StochasticMuZeroAlgorithm(Algorithm):
    def __init__(self, config, init_model, learning_rate=1e-5):
        super(StochasticMuZeroAlgorithm, self).__init__(init_model=init_model, learning_rate=learning_rate)
        self.training_step = 0
        self.config = config

    def build(self):
        self.model.build()

    def step_rollout_loss(self, t, state, targets):
        # state , c_tar, q_tar, a_tar
        targets = {k: tf.expand_dims(v[t], axis=0) for k, v in targets.items()}

        p, v = self.model._predictions(state)
        a_s = self.model._afterstate_dynamics(state, targets['a_tar'])
        sig, q = self.model._afterstate_predictions(a_s)
        c = quantize_chance_logits(sig)
        state_next, r = self.model._dynamics(a_s, c)

        loss =  reg_loss_fun(targets['q_tar'], v) + cat_loss_fun(targets['a_tar'], p)
        loss += reg_loss_fun(targets['q_tar'], q) + cat_loss_fun(c, sig) + 0.1*reg_loss_fun(targets['c_tar'], sig)

        return loss, state_next

    @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
    def train_step(self, board, vec, action, tar):
        self.training_step += 1

        a = tf.keras.layers.Flatten()(action)
        with tf.GradientTape() as tape:
            c_logits = self.model._encoder(board, vec)
            c_tar = quantize_chance_logits(c_logits)
            targets = {"c_tar": c_tar, "q_tar": tar, "a_tar": a}

            state = self.model._representation(board, vec)[:1]
            loss_ = 0.0
            for i in range(5):
                step_loss, state = self.step_rollout_loss(i, state, targets)
                loss_ += step_loss
            loss_ /= 5

        params = self.model.trainable_variables
        grads = tape.gradient(loss_, params)
        self.optimizer.apply_gradients(zip(grads, params))
        self.train_loss(loss_)
        return loss_

    @tf.function(input_signature=POLICY_SIGNATURE)
    def policy(self, board, action, vec):
        q = self.model(board, action, vec)
        return q
