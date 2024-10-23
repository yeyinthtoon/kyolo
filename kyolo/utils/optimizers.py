import re

from keras import ops, optimizers, saving


@saving.register_keras_serializable()
class MultiLRAdam(optimizers.Adam):
    def __init__(self, learning_rates, learning_rate=0.001, *args, **kwargs):
        super().__init__(learning_rate=learning_rate, *args, **kwargs)
        self.learning_rates = learning_rates
        self._lr_regex = {re.compile(k): v for k, v in learning_rates.items()}

    def _get_learning_rate_for_var(self, var):
        """Get learning rate for variable."""
        for regex, lr in self._lr_regex.items():
            if regex.search(var.name):
                if isinstance(lr, optimizers.schedules.LearningRateSchedule):
                    return lr(self.iterations)
                elif callable(lr):
                    return lr()
                return lr
        lr = self.learning_rate
        if isinstance(lr, optimizers.schedules.LearningRateSchedule):
            return lr(self.iterations)
        elif callable(lr):
            return lr()

        return lr

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return
        for variable in variables:
            if self._use_weight_decay(variable):
                lr = self._get_learning_rate_for_var(variable)
                lr = ops.cast(lr, variable.dtype)
                wd = ops.cast(self.weight_decay, variable.dtype)
                variable.assign(variable - variable * wd * lr)

    def update_step(self, gradient, variable, _):
        """Update step given gradient and the associated model variable."""
        lr = self._get_learning_rate_for_var(variable)
        lr = ops.cast(lr, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(ops.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, variable.dtype), local_step)

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign_add(m, ops.multiply(ops.subtract(gradient, m), 1 - self.beta_1))
        self.assign_add(
            v,
            ops.multiply(ops.subtract(ops.square(gradient), v), 1 - self.beta_2),
        )
        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(v_hat, ops.maximum(v_hat, v))
            v = v_hat
        self.assign_sub(
            variable,
            ops.divide(ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"learning_rates": self.learning_rates})
        return config
