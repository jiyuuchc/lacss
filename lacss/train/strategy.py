import treex as tx
import typing as tp
import jax
from functools import partial
jnp = jax.numpy

Model = tx.Module
Logs = tp.Dict[str, tp.Any]

class Eager:
    @classmethod
    def predict(cls, model, inputs):
        print("JIT Predict")
        inputs_obj = tx.Inputs.from_value(inputs)
        preds = model(*inputs_obj.args, **inputs_obj.kwargs) 
        return preds

    @classmethod
    def _loss_fn(
        cls,
        params: tx.Tree,
        model: tx.Module,
        loss_logs: tx.LossAndLogs,
        inputs: tp.Any,
        labels: tp.Any,
    ) -> tp.Tuple[jnp.ndarray, tp.Tuple[Model, tx.LossAndLogs, Logs]]:
        model = model.merge(params) # so the compiler can track the effect of params
        preds = cls.predict(model, inputs)
        loss, losses_logs, metrics_logs = loss_logs.batch_loss_epoch_logs(
            preds=preds,
            model=model,
            inputs=inputs,
            aux_losses=model.loss_logs(),
            aux_metrics=model.metric_logs(),
            **labels
        )
        logs = {**losses_logs, **metrics_logs}
        return loss, (model, loss_logs, logs)

    @classmethod
    def init_step(
        cls,
        seed: int,
        model: Model,
        optmz: tx.Optimizer,
        inputs: tp.Any,
    ) -> tp.Tuple[Model, tx.Optimizer]:
        # print('JIT init_step')
        if model is not None and not model.initialized:
            model = model.init(seed, inputs=inputs)
        if optmz is not None and not optmz.initialized:
            optmz = optmz.init(model.trainable_parameters())

        return model, optmz

    @classmethod
    def train_step(
        cls,
        model: Model,
        optimizer: tx.Optimizer,
        loss_logs: tx.LossAndLogs,
        inputs: tp.Any,
        labels: tp.Any,
    ) -> tp.Tuple[Model, tx.Optimizer, tx.LossAndLogs, Logs]:
        # print('JIT train_step')

        params = model.trainable_parameters()

        grads, (model, loss_logs, logs) = jax.grad(cls._loss_fn, has_aux=True)(
            params, model.train(), loss_logs, inputs, labels
        )

        params = optimizer.update(grads, params)
        model = model.merge(params)

        return model, optimizer, loss_logs, logs

    @classmethod
    def test_step(
        cls,
        model: Model,
        metrics: tx.Metric,
        inputs: tp.Any,
        labels: tp.Any,
    ):
        # print('JIT test_step')
        preds = cls.predict(model, inputs)
        metrics.update(
            preds=preds,
            model=model,
            inputs=inputs,
            **labels,
        )
        return metrics

class Core(Eager):
    predict = jax.jit(Eager.predict)

class JIT(Core):
    train_step = jax.jit(Core.train_step)
    test_step = jax.jit(Core.test_step)

class Distributed(Core):
    @classmethod
    def _train_step(
        cls,
        model: Model,
        optimizer: tx.Optimizer,
        loss_logs: tx.LossAndLogs,
        inputs: tp.Any,
        labels: tp.Any,
    ):
        print("JITTTTING")
        # copies of models have differnt rng key
        axis_index = jax.lax.axis_index("device")
        model.map(lambda key: jax.random.split(key, jax.device_count())[axis_index], tx.Rng, inplace=True)

        params = model.trainable_parameters()
        batch_loss_logs = tx.copy(loss_logs)
        grads, (model, batch_loss_logs, logs) = jax.grad(cls._loss_fn, has_aux=True)(
            params, model.train(), batch_loss_logs, inputs, labels
        )

        grads = jax.lax.pmean(grads, axis_name="device")
        params = optimizer.update(grads, params)
        model = model.merge(params)

        # sync batch statistics
        model.map(partial(jax.lax.pmean, axis_name="device"), tx.BatchStat, inplace=True)

        # update metrics from batch_loss_logs
        batch_loss_logs = jax.tree_map(lambda x,y: y-x, loss_logs.MetricState(), batch_loss_logs.MetricState())
        batch_loss_logs.map(partial(jax.lax.psum, axis_name='device'), tx.MetricState, inplace=True)
        loss_logs = loss_logs.merge(jax.tree_map(lambda x,y: x+y, loss_logs.MetricState(), batch_loss_logs.MetricState()))

        # aggregate logs
        logs = jax.tree_map(partial(jax.lax.pmean, axis_name='device'), logs)

        return model, optimizer, loss_logs, logs

    @classmethod
    def _test_step(
        cls,
        model: Model,
        metrics: tx.Metric,
        inputs: tp.Any,
        labels: tp.Any,
    ):
        preds = cls.predict(inputs)

        labels, preds = jax.lax.all_gather((labels, preds), axis_name='device')
        labels, preds = jax.tree_map(
            lambda v: v.reshape([-1]+v.shape[2:]),
            (labels, preds),
        )

        metrics.update(
            preds=preds,
            model=model,
            inputs=inputs,
            **labels,
        )

        return metrics

Distributed.train_step = jax.pmap(
    Distributed._train_step,
    in_axes =(None, None, None, 0, 0),
    out_axes=None, 
    axis_name="device"
    )

Distributed.test_step = jax.pmap(
    Distributed._test_step,
    in_axes = (None, None, 0, 0),
    out_axes = None,
    axis_name="device"
    )
