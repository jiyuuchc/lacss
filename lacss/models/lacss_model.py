# from typing import Union, List, Optional
# import jax
# import elegy as eg

# from ..metrics import MeanAP
# from ..ops import *

# from elegy.model.model_base import _MODEL_CONTEXT
# from elegy import data
# from elegy.callbacks import Callback, CallbackList
# from elegy.callbacks.sigint import SigIntMode

# jnp = jax.numpy

# class LacssModel(eg.Model):
#     def evaluate(
#             self,
#             x=None, y=None,
#             verbose: int = 1,
#             steps: Optional[int] = None,
#             callbacks: Union[List[Callback], CallbackList, None] = None,
#             **kwargs,
#     ):
#         ''' override low-level evalute() because we need some non-jax-able aggregation 
#         '''
#         batch_size = 1 # always one sample per device

#         with _MODEL_CONTEXT.callbacks_context(self if _MODEL_CONTEXT.model is None else None):
#             data_handler = data.DataHandler(
#                 x=x,
#                 y=y,
#                 epochs=1,
#                 steps_per_epoch=steps,
#                 verbose=1,
#                 training=False,
#             )

#             # Container that configures and calls `tf.keras.Callback`s.
#             if not isinstance(callbacks, CallbackList):
#                 callbacks = CallbackList(
#                     callbacks,
#                     add_history=True,
#                     add_progbar=verbose!=0,
#                     sigint_mode=SigIntMode.TEST,
#                     model=self,
#                     verbose=verbose,
#                     epochs=1,
#                     steps=data_handler.inferred_steps,
#                 )
#             callbacks.on_test_begin()

#             self.eval(inplace=True)
#             for _, iterator in data_handler.enumerate_epochs():
#                 m = MeanAP([.5, .75])
#                 with data_handler.catch_stop_iteration():
#                     for step in data_handler.steps():
#                         callbacks.on_test_batch_begin(step)
#                         batch = next(iterator)
#                         image, label , _ = data.unpack_x_y_sample_weight(batch)

#                         preds = self.predict_on_batch(image)

#                         for k in range(image.shape[0]):
#                             valid = preds['pred_location_scores'][k] >= 0
#                             cur_pred = {key: v[k] for key,v in preds.items()}
#                             ious = iou_patches_and_labels(cur_pred, label[k])
#                             m.update_state(ious, cur_pred['pred_location_scores'])

#                         logs = {"size": data_handler.batch_size}
#                         callbacks.on_test_batch_end(step, logs)

#                         if self.stop_training:
#                             break

#                 logs = dict(zip(['ap50', 'ap75'], m.result()))
#                 if self.stop_training:
#                     break

#             callbacks.on_test_end()

#             return logs
