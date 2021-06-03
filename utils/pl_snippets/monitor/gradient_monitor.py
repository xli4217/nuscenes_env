from .data_monitor_base import DataMonitorBase


class GradientMonitor(DataMonitorBase):

    GROUP_NAME = "gradient"

    def __init__(self, log_every_n_steps: int = None):
        """
        Callback that logs the histogram of gradients passed to `gradient`.

        Args:
            log_every_n_steps: The interval at which histograms should be logged. This defaults to the
                interval defined in the Trainer. Use this to override the Trainer default.

        Example:

            .. code-block:: python

                # log histogram of training data passed to `LightningModule.gradient`
                trainer = Trainer(callbacks=[GradientMonitor()])
        """
        super().__init__(log_every_n_steps=log_every_n_steps)

    def on_fit_start(self, trainer, pl_module, batch, *args, **kwargs):
        super().on_fit_start(trainer, pl_module, batch, *args, **kwargs)

        for k, v in pl_module.named_parameters():
            #self.log_histograms(batch, group=self.GROUP_NAME)
            self.log_histogram(v.grad, self.GROUP_NAME+"/"+k)