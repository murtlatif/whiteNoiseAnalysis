from utils.config_interface import ConfigInterface


class TrainingConfig(ConfigInterface):
    def __init__(self, epochs: int, loss_fnc, save_path, save_name: str = 'model', starting_epoch=0, log_frequency=100) -> None:
        super().__init__(
            epochs=epochs,
            loss_fnc=loss_fnc,
            save_path=save_path,
            save_name=save_name,
            starting_epoch=starting_epoch,
            log_frequency=log_frequency
        )

    def is_valid(self):
        cfg = self.config

        if cfg['epochs'] <= 0:
            return False

        if not callable(cfg['loss_fnc']):
            return False

        if not isinstance(cfg['save_name'], str):
            return False

        if cfg['starting_epoch'] < 0:
            return False

        if cfg['log_frequency'] < 0:
            return False

        return True
