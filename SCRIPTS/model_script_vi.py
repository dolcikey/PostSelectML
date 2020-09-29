import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import metrics
from tensorflow.python.training.rmsprop import RMSPropOptimizer
import os

HEIGHT = 500
WIDTH = 333
DEPTH = 3
NUM_CLASSES = 2
BATCH_SIZE = 32
INPUT_TENSOR_NAME = "inputs_input" # According to Amazon, needs to match the name of the 
                                   # first layer + "_input"
                                   # Workaround for actual known bugs

def keras_model_fn(hyperparameters):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(HEIGHT, WIDTH, DEPTH), activation="relu", name="inputs",padding="same"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, kernel_size=(3, 3), activation="sigmoid", padding="same"))
    model.add(MaxPooling2D())
    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="sigmoid"))
    model.add(Dense(2, activation="softmax"))

    opt = RMSPropOptimizer(learning_rate=hyperparameters['learning_rate'], decay=hyperparameters['decay'])

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[metrics.binary_accuracy, 
                'accuracy'],_tuning_objective_metric  = ['recall', 'f1_score'])
    return model


def serving_input_fn(hyperparameters):
    tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_dir=training_dir)


def _input(mode, batch_size, data_dir):
    assert os.path.exists(data_dir), ("Unable to find images resources for input, are you sure you downloaded them ?")

    if mode == tf.estimator.ModeKeys.TRAIN:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            zoom_range=0.3,
            horizontal_flip=True # This portion horizontally flips all the evaluation images adding 
                                 # these agumented images to help remedy the class imbalance problem. 
        )
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(data_dir, target_size=(HEIGHT, WIDTH), batch_size=batch_size)
    images, labels = generator.next()

    return {INPUT_TENSOR_NAME: images}, labels


# This script is based off the script created by Paul Breton (Medium Article: Keras in the cloud with Amazon SageMaker)


def _call_model_fn_eval_distributed(self, input_fn, config):
    """Call model_fn in distribution mode and handle return values."""

    iterator, input_hooks = self._get_iterator_from_input_fn(
        input_fn, model_fn_lib.ModeKeys.EVAL, self._eval_distribution)

    is_tpu_strategy = (
        self._eval_distribution.__class__.__name__ == 'TPUStrategy')

    if is_tpu_strategy:
      steps_per_run_variable = training.get_or_create_steps_per_run_variable()
      def step_fn(ctx, inputs):
        """Runs one step of the eval computation and captures outputs."""
        if isinstance(inputs, tuple):
          features, labels = inputs
        else:
          features = inputs
          labels = None
        estimator_spec = self._eval_distribution.extended.call_for_each_replica(
            self._call_model_fn,
            args=(features, labels, model_fn_lib.ModeKeys.EVAL, config))
        eval_metric_ops = _verify_and_create_loss_metric(
            estimator_spec.eval_metric_ops, estimator_spec.loss,
            self._eval_distribution)
        update_op, eval_dict = _extract_metric_update_ops(
            eval_metric_ops, self._eval_distribution)
        ctx.set_non_tensor_output(name='estimator_spec', output=estimator_spec)
        ctx.set_non_tensor_output(name='eval_dict', output=eval_dict)
        return update_op

      # TODO(priyag): Fix eval step hook to account for steps_per_run.
      ctx = self._eval_distribution.extended.experimental_run_steps_on_iterator(
          step_fn, iterator, iterations=steps_per_run_variable)
      update_op = ctx.run_op
      eval_dict = ctx.non_tensor_outputs['eval_dict']
      grouped_estimator_spec = ctx.non_tensor_outputs['estimator_spec']
    else:
      features, labels = estimator_util.parse_iterator_result(
          iterator.get_next())
      grouped_estimator_spec = (
          self._eval_distribution.extended.call_for_each_replica(
              self._call_model_fn,
              args=(features, labels, model_fn_lib.ModeKeys.EVAL, config)))
      eval_metric_ops = _verify_and_create_loss_metric(
          grouped_estimator_spec.eval_metric_ops, grouped_estimator_spec.loss,
          self._eval_distribution)
      update_op, eval_dict = _extract_metric_update_ops(
          eval_metric_ops, self._eval_distribution)

    scaffold = _combine_distributed_scaffold(
        grouped_estimator_spec.scaffold, self._eval_distribution)
    evaluation_hooks = self._eval_distribution.unwrap(
        grouped_estimator_spec.evaluation_hooks)[0]
    evaluation_hooks = evaluation_hooks + (
        estimator_util.StrategyInitFinalizeHook(
            self._eval_distribution.initialize,
            self._eval_distribution.finalize),)

    return (scaffold, evaluation_hooks, input_hooks, update_op, eval_dict)

  def _evaluate_run(self, checkpoint_path, scaffold, update_op, eval_dict,
                    all_hooks, output_dir):
    """Run evaluation."""
    eval_results = evaluation._evaluate_once(  # pylint: disable=protected-access
        checkpoint_path=checkpoint_path,
        master=self._config.evaluation_master,
        scaffold=scaffold,
        eval_ops=update_op,
        final_ops=eval_dict,
        hooks=all_hooks,
        config=self._session_config)

    current_global_step = eval_results[ops.GraphKeys.GLOBAL_STEP]

    _write_dict_to_summary(
        output_dir=output_dir,
        dictionary=eval_results,
        current_global_step=current_global_step)

    if checkpoint_path:
      _write_checkpoint_path_to_summary(
          output_dir=output_dir,
          checkpoint_path=checkpoint_path,
          current_global_step=current_global_step)

    return eval_results