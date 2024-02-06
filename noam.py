class Noam():
    '''Noam wrapper'''

    def __init__(self, optimiser, d_model, n_warmup_steps):
        self._optimiser = optimiser
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimiser"
        self._update_learning_rate()
        self._optimiser.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimiser"
        self._optimiser.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self._get_lr_scale()

        for param_group in self._optimiser.param_groups:
            param_group['lr'] = lr