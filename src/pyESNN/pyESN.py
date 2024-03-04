import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric argument s, broadcasts it
       to the specified length if possible.

    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s

    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


def identity(x):
    return x

class ESN(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
                 input_scaling=None, teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=torch.tanh, inverse_out_activation=torch.tanh,
                 random_state=None, silent=True):
        super(ESN, self).__init__()

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = input_shift
        self.input_scaling = input_scaling

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.initweights()

    def initweights(self):
        W = torch.tensor(self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5)
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        radius = torch.max(torch.abs(torch.eig(W)[0]))
        self.W = W * (self.spectral_radius / radius)

        self.W_in = torch.tensor(self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1)
        self.W_feedb = torch.tensor(self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1)

    def _update(self, state, input_pattern, output_pattern):
        if self.teacher_forcing:
            preactivation = (torch.matmul(self.W, state)
                             + torch.matmul(self.W_in, input_pattern)
                             + torch.matmul(self.W_feedb, output_pattern))
        else:
            preactivation = (torch.matmul(self.W, state)
                             + torch.matmul(self.W_in, input_pattern))
        return (torch.tanh(preactivation)
                + self.noise * (torch.rand(self.n_reservoir) - 0.5))

    def _scale_inputs(self, inputs):
        if self.input_scaling is not None:
            inputs = torch.matmul(inputs, torch.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):
        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, inspect=False):
        if inputs.ndim < 2:
            inputs = inputs.unsqueeze(1)
        if outputs.ndim < 2:
            outputs = outputs.unsqueeze(1)

        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)

        if not self.silent:
            print("harvesting states...")

        states = torch.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :],
                                        teachers_scaled[n - 1, :])

        if not self.silent:
            print("fitting...")

        transient = min(int(inputs.shape[1] / 10), 100)
        extended_states = torch.cat((states, inputs_scaled), dim=1)
        self.W_out = torch.matmul(torch.pinverse(extended_states[transient:, :]),
                                  self.inverse_out_activation(teachers_scaled[transient:, :])).T

        self.laststate = states[-1, :]
        self.lastinput = inputs[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        if inspect:
            pass

        if not self.silent:
            print("training error:")

        pred_train = self._unscale_teacher(self.out_activation(
            torch.matmul(extended_states, self.W_out.T)))
        if not self.silent:
            print(torch.sqrt(torch.mean((pred_train - outputs) ** 2)))
        return pred_train

    def predict(self, inputs, continuation=True):
        if inputs.ndim < 2:
            inputs = inputs.unsqueeze(1)
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = torch.zeros(self.n_reservoir)
            lastinput = torch.zeros(self.n_inputs)
            lastoutput = torch.zeros(self.n_outputs)

        inputs = torch.cat((lastinput, self._scale_inputs(inputs)))
        states = torch.cat((laststate, torch.zeros((n_samples, self.n_reservoir))))
        outputs = torch.cat((lastoutput, torch.zeros((n_samples, self.n_outputs))))

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(torch.matmul(self.W_out,
                                                                 torch.cat((states[n + 1, :], inputs[n + 1, :]))))

        return self._unscale_teacher(self.out_activation(outputs[1:]))
