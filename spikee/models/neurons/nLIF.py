import torch
from copy import deepcopy

class nLIF:
    def __init__(self, nodes, dim, input_size, tau, maxspan, seed):
        '''
        nLIF neuron model.

        nodes: number nodes in the graph
        dim: embedding dimension
        input_size: dimension of static input pool
        tau: time constant
        maxspan: time interval
        seed: random seed
        '''
        torch.manual_seed(seed)
        # network size
        self.nodes = nodes
        self.dim = dim
        self.input_size = input_size

        # neuron parameters
        self.taus = tau
        self.threshold = 1.
        self.winit = 0.2
        self.maxspan = maxspan/2

        # create input spikes
        self.input_spikes = (torch.rand(input_size)-0.5)*maxspan
        self.input_spikes = self.input_spikes.sort().values
        self.input_exp = torch.exp(self.input_spikes/self.taus)
        self.spike_seq = torch.zeros((self.input_size, self.input_size))
        for i in range(self.input_size):
            for j in range(self.input_size):
                if i >= j:
                    self.spike_seq[i][j] = self.input_exp[j]
        self.spike_mask = (self.spike_seq > 0)*1.

        # create input -> population weights
        self.weights = self.init_weights()
        self.spike_times = self.get_spike_times()

    def init_weights(self):
        '''
        Initialize weights using normal distribution.

        Output: weights
        '''
        weights = torch.nn.Embedding(self.nodes, self.dim*self.input_size)

        # guarantee that all populations spike once
        for i in range(self.nodes):
            while bool((weights.weight.view(-1,self.dim,self.input_size)[i].sum(-1) < 1).any()) == True:
                weights.weight.data[i] = torch.normal(mean = self.winit, std = 1, size = (1, self.dim*self.input_size))

        return weights

    def get_spike_times(self):
        '''
        Calculate spike times analytically. Used for training.

        Output: spike times
        '''
        # needs to go through all causal sets
        spike_times = torch.zeros((self.nodes, self.dim))+self.maxspan+0.5
        weights = self.weights.weight.view(-1, self.dim, self.input_size)

        for j in range(self.input_size):
            wsumexp = torch.matmul(weights, self.spike_seq[j])
            wsum = torch.matmul(weights, self.spike_mask[j])
            wsumdiff = wsum - self.threshold
            wquotient = wsumexp/(wsumdiff+ 1e-10) + 1e-10 # for stability
            times = self.taus * torch.log(wquotient)
            if j < self.input_size-1:
                # check condition for spiking
                # 1. has not spiked yet
                # 2. weights sum over threshold
                # 3. next input spike does not hinder firing
                new_spikes = (spike_times == self.maxspan+0.5)*(wsum > self.threshold)*(wquotient < self.input_exp[j+1])
            else:
                # for last spike
                new_spikes =  (spike_times == self.maxspan+0.5)*(wsum > self.threshold)
            spike_times[new_spikes] = times[new_spikes]

        return spike_times

    def update_embeddings(self):
        '''
        Spike embeddings are stored after calculation to reduce
        compute time when evaluating several times without
        training updates.
        '''
        self.spike_times = self.get_spike_times()

    def embeddings(self, s_embs, o_embs):
        '''
        Read out embeddings.

        Input: list of subjects, list of objects
        Output: list of subject embeddings, list of object embeddings
        '''
        s_embs = torch.tensor(s_embs).long()
        o_embs = torch.tensor(o_embs).long()

        return self.spike_times[s_embs], self.spike_times[o_embs]

    def weight_loss(self):
        '''
        Regularization term that increases weights when their sum is below the threshold value.

        Output: regularization term
        '''
        weight_norm = self.weights.weight.view(-1, self.dim, self.input_size).sum(-1)
        return ((self.threshold-weight_norm)*(weight_norm < self.threshold)).sum()

    def _integrate_model_using_Euler(self):
        '''
        Euler method to obtain spike times. Used to cross-check the analytical solution.

        Output: spike times, membrane potentials over time
        '''
        results = []
        spike_times = torch.zeros((self.nodes, self.dim))+1.5
        voltage = torch.zeros((self.nodes, self.dim))
        weights = self.weights.weight.view(-1, self.dim, self.input_size)

        # Euler integration from t = -maxspan to maxspan to solve ODE
        # return spike times + voltage traces
        t = -self.maxspan
        dt = 0.01

        results.append(deepcopy(voltage.detach().numpy()))
        while t <= self.maxspan:
            voltage.data = torch.matmul(weights, (1-torch.exp(-(t-self.input_spikes)/self.taus)) * (t > self.input_spikes))
            t += dt
            results.append(deepcopy(voltage.detach().numpy()))

            new_spikes = (t-dt-1.5)*(voltage > self.threshold)+1.5
            spiked = spike_times == 1.5
            spike_times = torch.logical_not(spiked)*spike_times + spiked*new_spikes

        return spike_times, results
