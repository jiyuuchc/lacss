| class | description |
|---|---|
| lacss.train.Core | No model JIT compiling, i.e., for debugging |
| lacss.train.JIT | JIT compile the model. Default strategy |
| lacss.train.VMapped | Transform the model with vmap. This allows defining a model on unbatched data but train with batched data. |
| lacss.train.Distributed | Transform the model with pmap. This allows training the model on multiple devices. |

::: lacss.train.TFDatasetAdapter
      options:
        show_root_heading: true

::: lacss.train.TorchDataLoaderAdapter
      options:
        show_root_heading: true

::: lacss.train.Trainer
      options:
        show_root_heading: true

::: lacss.train.LacssTrainer
      options:
        show_root_heading: true
