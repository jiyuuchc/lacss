import typing as tp
import inspect
import jax, optax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from optax import GradientTransformation
from dataclasses import asdict

from .pytree import Pytree, static_field
from .loss import Loss

class WrappedModule(Pytree, mutable=True):
    _variables: dict
    _flax_module: nn.Module = static_field()
    initialized:bool = static_field()

    def __init__(self, obj):
        if not isinstance(obj, nn.Module):
            raise ValueError('Can only wrap flax modules')

        self._flax_module = obj
        self._variables = {}
        self.initialized = False
        # self.__doc__ = obj.__doc__
        # self.__annotations__.update(obj.__annotations__)

    def init(self, *args, **kwargs):
        _ = self.init_with_output(*args, **kwargs)

    def init_with_output(self, *args, **kwargs):
        output, self._variables = self._flax_module.init_with_output(*args, **kwargs)
        self.init_with_output = True
        return output
    
    def update_variables(self, vars):
        old_vars = unfreeze(self._variables)
        old_vars.update(vars)
        self._variables = freeze(old_vars)            
    
    def get_variables(self, col='params'):
        return self._variables[col]

    def __str__(self, *args, **kwargs):
        return f'Pytree({self._flax_module.__str__(*args, **kwargs)}))'

    def __repr__(self, *args, **kwargs):
        return f'Pytree({self._flax_module.__repr__(*args, **kwargs)})'

    def __call__(self, *args, **kwargs):
        if 'mutable' in kwargs:
            raise ValueError('expected kwargs "mutable".')
        output, self._variables = self._flax_module.apply(self._variables, *args, mutable=True, **kwargs)
        return output

    def __getattr__(self, attr):
        if attr == 'apply':
            raise AttributeError('Do not call apply() in a pytree-ized moduel. Use __call__() instead')
        if attr == 'vars':
            return self._variables

        return getattr((self._flax_module), attr)
    
    def get_config(self):
        return asdict(self._flax_module)
    
    # def parameter_flags(self):
    #     all_nodes = jtu.tree_map(lambda x: x, self)
    #     all_nodes._variables = unfreeze(all_nodes._variables)
    #     params_ids = jtu.tree_leaves(jtu.tree_map(id, all_nodes._variables['params']))
    #     flags = jtu.tree_map(lambda x: id(x) in params_ids, all_nodes)
    #     flags._variables = freeze(flags._variables)
    #     return flags
        
class WrappedGT(Pytree, mutable=True):
    _states: tp.Any
    # _tx_core: GradientTransformation = static_field()
    _tx: GradientTransformation = static_field()
    initialized:bool = static_field()

    def __init__(self, obj):
        if not isinstance(obj, GradientTransformation):
            raise ValueError('Can only wrap optax optimizes')
        self._tx = obj 
        self._states = None
        self.initialized = False

    def init(self, tree):
        # partitions = {True: self._tx_core, False: optax.set_to_zero()}
        # flags = jtu.tree_leaves(tree.parameter_flags())
        # self._tx = optax.multi_transform(partitions, flags)
        # self._state = self._tx.init(jtu.tree_leaves(tree))
        self._state = self._tx.init(jtu.tree_leaves(tree))
        self.initialized = True
    
    def update(self, grad, tree):
        tree_lfs, st1 = jtu.tree_flatten(tree)
        grad_lfs, st2 = jtu.tree_flatten(grad)
        assert st1 == st2
        updates, self._state = self._tx.update(grad_lfs, self._state, tree_lfs)
        tree_lfs = optax.apply_updates(tree_lfs, updates)
        tree = jtu.tree_unflatten(st1, tree_lfs)
        return tree

class LossLog(Pytree, mutable=True):
    _losses: tp.Sequence[Loss] = static_field()

    def __init__(self, losses):
        if isinstance(losses, Loss):
            self._losses = [losses]
        else:
            self._losses = losses

        self.reset()

    def reset(self):
        self._cnts = {}
        self._sums = {}
        for loss in self._losses:
            self._cnts.update({loss.name: jnp.asarray(0.)})
            self._sums.update({loss.name: jnp.asarray(0.)})        
    
    def update(self, **kwargs):
        loss_log = {loss_fn.name: loss_fn(**kwargs) for loss_fn in self._losses}

        for k, v in loss_log.items():
            self._cnts[k] += 1
            self._sums[k] += v

        return sum(loss_log.values()), loss_log
    
    def compute(self):
        results = {k: self._sums[k] / self._cnts[k] for k in self._cnts}
        return results
