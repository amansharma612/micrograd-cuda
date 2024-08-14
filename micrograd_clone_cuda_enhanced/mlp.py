from engine import Tensor
from ops import Operations
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = Operations(p.device).matrix_zeros(p.shape[0], p.shape[1])


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = Tensor([[random.uniform(-1, 1) for i in range (nin)] for j in range(nout)], shape = (nout, nin))
        self.neurons.to("gpu")

    def __call__(self, x):
        out = self.neurons @ x
        return out

    def parameters(self):
        return [self.neurons]
    
    def __repr__(self):
        return f"Layer of dim: {self.neurons.shape}"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], non_lin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of {', '.join([str(layer) for layer in self.layers])}"