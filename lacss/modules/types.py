
class ModuleConfig():
    def __init__(self):
        self._config_dict = {}

    def get_config(self):
        return self._config_dict

    def __getattr__(self, name):
        if name != '_config_dict' and name in self._config_dict:
            return self._config_dict[name]
        else:
            # raise AttributeError()
            return super().__getattr__(name)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
