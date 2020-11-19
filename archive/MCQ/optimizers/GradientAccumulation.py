import tensorflow as tf
from tensorflow.python.training import training_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops, math_ops, state_ops, control_flow_ops

class GradientAccumulation(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, accuSteps: int, name: str = 'GradientAccumulation', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('accuSteps', accuSteps)
        self._optimizer = optimizer

    def get_config(self):
        config = super().get_config()
        config["optimizer"] = self._optimizer
        config["accuSteps"] = self._serialize_hyperparameter("accuSteps")
        return config

    def _create_slots(self, var_list):
        self._numVars = len(var_list)
        for var in var_list:
            self.add_slot(var, 'accuGrads')
        self._optimizer._create_slots(var_list)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        self._optimizer._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["accuSteps"] = tf.identity(self._get_hyper("accuSteps", tf.int64))


    def _resource_apply_dense(self, grad, var, apply_state):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype))

        accuGrads = self.get_slot(var, 'accuGrads')
        accuGrads.assign_add(grad)

        def _wrapUseAccu(accuGrads, grad, var, apply_state):
            result = self._optimizer._resource_apply_dense(accuGrads, var, apply_state)
            accuGrads.assign(tf.broadcast_to(0.0, tf.shape(accuGrads)))
            return result

        def _wrapWOAccu(accuGrads, grad, var, apply_state):
            return self._optimizer._resource_apply_dense(grad * 0.0, var, apply_state)

        # Warning: Any Tensors or Operations created outside of true_fn and false_fn will be executed regardless of which branch is selected at runtime.
        # https://www.tensorflow.org/api_docs/python/tf/cond
        return tf.cond((self.iterations + 1) % coefficients["accuSteps"] == 0, lambda: _wrapUseAccu(accuGrads, grad, var, apply_state), lambda: _wrapWOAccu(accuGrads, grad, var, apply_state))

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype))

        accuGrads = self.get_slot(var, 'accuGrads')
        accuGrads.assign_add(grad)

        def _wrapUseAccu(accuGrads, grad, var, indices, apply_state):
            result = self._optimizer._resource_apply_sparse(accuGrads, var, indices, apply_state)
            accuGrads.assign(tf.broadcast_to(0.0, tf.shape(accuGrads)))
            return result

        def _wrapWOAccu(accuGrads, grad, var, indices, apply_state):
            return self._optimizer._resource_apply_sparse(grad * 0.0, var, indices, apply_state)

        # Warning: Any Tensors or Operations created outside of true_fn and false_fn will be executed regardless of which branch is selected at runtime.
        # https://www.tensorflow.org/api_docs/python/tf/cond
        return tf.cond((self.iterations + 1) % coefficients["accuSteps"] == 0, lambda: _wrapUseAccu(accuGrads, grad, var, indices, apply_state), lambda: _wrapWOAccu(accuGrads, grad, var, indices, apply_state))



class GradientAccumulatedAdam(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, accuSteps: int=100, name: str = 'GradientAccumulatedAdam', **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('accuSteps', accuSteps)
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'accuSteps': self._serialize_hyperparameter('accuSteps')
        })
        return config

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')
        self._numVars = len(var_list)
        for var in var_list:
            self.add_slot(var, 'accuGrads')


    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super().set_weights(weights)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        accuSteps = tf.identity(self._get_hyper("accuSteps", tf.int64))
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] * (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(dict(lr=lr, epsilon=tf.convert_to_tensor(self.epsilon, var_dtype), beta_1_t=beta_1_t, beta_1_power=beta_1_power, one_minus_beta_1_t=1 - beta_1_t, beta_2_t=beta_2_t, beta_2_power=beta_2_power, one_minus_beta_2_t=1 - beta_2_t, accuSteps=accuSteps))


    def _resource_apply_dense(self, grad, var, apply_state):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype))

        accuGrads = self.get_slot(var, 'accuGrads')
        accuGrads.assign_add(grad)

        def _wrapUseAccu(accuGrads, grad, var, apply_state):
            m = self.get_slot(var, 'm')
            v = self.get_slot(var, 'v')

            if not self.amsgrad:
                result = training_ops.resource_apply_adam(var.handle, m.handle, v.handle, coefficients['beta_1_power'], coefficients['beta_2_power'], coefficients['lr_t'], coefficients['beta_1_t'], coefficients['beta_2_t'], coefficients['epsilon'], grad, use_locking=self._use_locking)
            else:
                vhat = self.get_slot(var, 'vhat')
                result = training_ops.resource_apply_adam_with_amsgrad(var.handle, m.handle, v.handle, vhat.handle, coefficients['beta_1_power'], coefficients['beta_2_power'], coefficients['lr_t'], coefficients['beta_1_t'], coefficients['beta_2_t'], coefficients['epsilon'], grad, use_locking=self._use_locking)
            accuGrads.assign(tf.broadcast_to(0.0, tf.shape(accuGrads)))
            return result

        def _wrapWOAccu(accuGrads, grad, var, apply_state):
            return training_ops.resource_apply_gradient_descent(var.handle, 0.0, grad * 0.0, use_locking=self._use_locking)

        # Warning: Any Tensors or Operations created outside of true_fn and false_fn will be executed regardless of which branch is selected at runtime.
        # https://www.tensorflow.org/api_docs/python/tf/cond
        return tf.cond((self.iterations + 1) % coefficients["accuSteps"] == 0, lambda: _wrapUseAccu(accuGrads, grad, var, apply_state), lambda: _wrapWOAccu(accuGrads, grad, var, apply_state))

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype))

        accuGrads = self.get_slot(var, 'accuGrads')
        accuGrads.assign_add(grad)

        def _wrapUseAccu(accuGrads, grad, var, indices, apply_state):
            m = self.get_slot(var, 'm')
            m_scaled_g_values = accuGrads * coefficients['one_minus_beta_1_t']
            m_t = state_ops.assign(m, m * coefficients['beta_1_t'], use_locking=self._use_locking)
            with ops.control_dependencies([m_t]):
                m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v = self.get_slot(var, 'v')
            v_scaled_g_values = (accuGrads * accuGrads) * coefficients['one_minus_beta_2_t']
            v_t = state_ops.assign(v, v * coefficients['beta_2_t'], use_locking=self._use_locking)
            with ops.control_dependencies([v_t]):
                v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

            if not self.amsgrad:
                v_sqrt = math_ops.sqrt(v_t)
                var_update = state_ops.assign_sub(var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']), use_locking=self._use_locking)
                result = control_flow_ops.group(*[var_update, m_t, v_t])
            else:
                v_hat = self.get_slot(var, 'vhat')
                v_hat_t = math_ops.maximum(v_hat, v_t)
                with ops.control_dependencies([v_hat_t]):
                    v_hat_t = state_ops.assign(v_hat, v_hat_t, use_locking=self._use_locking)
                v_hat_sqrt = math_ops.sqrt(v_hat_t)
                var_update = state_ops.assign_sub(var, coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']), use_locking=self._use_locking)
                result = control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

            accuGrads.assign(tf.broadcast_to(0.0, tf.shape(accuGrads)))
            return result

        def _wrapWOAccu(accuGrads, grad, var, indices, apply_state):
            return resource_variable_ops.resource_scatter_add(var.handle, indices, grad * 0.0)

        # Warning: Any Tensors or Operations created outside of true_fn and false_fn will be executed regardless of which branch is selected at runtime.
        # https://www.tensorflow.org/api_docs/python/tf/cond
        return tf.cond((self.iterations + 1) % coefficients["accuSteps"] == 0, lambda: _wrapUseAccu(accuGrads, grad, var, indices, apply_state), lambda: _wrapWOAccu(accuGrads, grad, var, indices, apply_state))
