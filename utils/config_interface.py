class ConfigInterface():
    def __init__(self, **config_kwargs):
        self.config = config_kwargs

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} {"[VALID]" if self.is_valid() else "[INVALID]"}: {self.config}'

    def is_valid(self):
        return True
