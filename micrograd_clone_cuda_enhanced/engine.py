import math
from typing import Any
from ops import Operations, reshape, flatten_matrix

class Tensor:
    

    def __init__(self, data, children = (), op  = '', shape = (), device = "cpu"):
        self.data = None
        self.data = data
        self.shape = shape # specify shape everytime you create a tensor
        self.grad = Operations(device).matrix_zeros(self.shape[0], self.shape[1])
        
        self._backward = lambda: None
        self._prev = set(children)
        self.op = op
        self.device = device


    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, (len(other), len(other[0])), device = self.device)
        out = Tensor(Operations(self.device).matmul(self.data, other.data, self.shape[0], self.shape[1], other.shape[1]), (self, other), "*", shape = (self.shape[0], other.shape[1]), device = self.device)

        # For explanation, see https://math.stackexchange.com/questions/2755654/partial-derivative-of-matrix-product-in-neural-network
        def backward():
            self_transpose_data = Operations(self.device).transpose(self.data, self.shape[0], self.shape[1])
            other_transpose_data = Operations(self.device).transpose(other.data, other.shape[0], other.shape[1])
            
               
            self_grad_data = Operations(self.device).matmul(out.grad, other_transpose_data, out.shape[0], out.shape[1], other.shape[0])
            other_grad_data = Operations(self.device).matmul(self_transpose_data, out.grad, self.shape[1], self.shape[0], out.shape[1])

            self.grad = Operations(self.device).add(self.grad, self_grad_data, self.shape[0], self.shape[1])
            other.grad = Operations(self.device).add(other.grad, other_grad_data, self.shape[0], self.shape[1])


        out._backward = backward

        
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, (len(other), len(other[0])), device = self.device)   
        out = Tensor(Operations(self.device).add(self.data, other.data, self.shape[0], self.shape[1]), children= (self, other), op = "+", shape = (self.shape[0], self.shape[1]), device = self.device)

        def backward():
            self_grad_data = Operations(self.device).matrix_ones(self.shape[0], self.shape[1])
            self_grad_data = Operations(self.device).mul(self_grad_data, out.grad, self.shape[0], self.shape[1])

            other_grad_data = Operations(self.device).matrix_ones(self.shape[0], self.shape[1])
            other_grad_data = Operations(self.device).mul(other_grad_data, out.grad, self.shape[0], self.shape[1])
            
            self.grad = Operations(self.device).add(self.grad, self_grad_data, self.shape[0], self.shape[1])
            other.grad = Operations(self.device).add(other.grad, other_grad_data, self.shape[0], self.shape[1])

        out._backward = backward

        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))

        out = Tensor(Operations(self.device).pow(self.data, other, self.shape[0], self.shape[1]), shape = (self.shape[0], self.shape[1]), children=(self,), op =  "+", device = self.device)

        def backward():
            self_grad_data = Operations(self.device).pow(self.data, other - 1, self.shape[0], self.shape[1])
            self_grad_data = Operations(self.device).scalar_mul(self_grad_data, other, self.shape[0], self.shape[1])
            self_grad_data = Operations(self.device).mul(self_grad_data, out.grad, self.shape[0], self.shape[1])

            self.grad = Operations(self.device).add(self.grad, self_grad_data, self.shape[0], self.shape[1])
            




        out._backward = backward

        return out
    
    def relu(self):

        out = Tensor(Operations(self.device).relu(self.data, self.shape[0], self.shape[1]), shape = (self.shape[0], self.shape[1]), children=(self,), op = "relu", device = self.device)
        
        def backward():
            self_grad_data = Operations(self.device).relugrad(self.data, self.shape[0], self.shape[1])
            self_grad_data = Operations(self.device).mul(self_grad_data, out.grad, self.shape[0], self.shape[1])

            self.grad = Operations(self.device).add(self.grad, self_grad_data, self.shape[0], self.shape[1])

        out._backward = backward

        return out

    def Sigmoid(self):
        def sigmoid(Tensor):
            '''helper function for sigmoid computation'''
            return 1 / (1 + math.exp(-1 * Tensor))
        

        out = Tensor(sigmoid(self.data), (self,), "Sigmoid")

        def backward():
            self.grad += sigmoid(self.data) * (1 - sigmoid(self.data)) * out.grad

        out._backward = backward

        return out
    
    def to(self, device):
        self.device = device
        self.data = Operations(device).to(self.data, self.shape[0], self.shape[1])
        self.grad = Operations(device).to(self.grad, self.shape[0], self.shape[1])
    

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for node in v._prev:
                    build_topo(node)
                topo.append(v)

        build_topo(self)

        '''Setting the output gradient Tensor for backpropogation'''
        self.grad = Operations(self.device).matrix_ones(self.shape[0], self.shape[1])

        for node in reversed(topo):
            node._backward()
    
    def step(self, lr):
        # Performs Gradient Descent depending on the learning rate
        data_delta = Operations(self.device).scalar_mul(self.grad, -1 * lr, self.shape[0], self.shape[1])
        self.data = Operations(self.device).add(self.data, data_delta, self.shape[0], self.shape[1])

    def zero_grad(self):
        self.grad = Operations(self.device).matrix_zeros(self.shape[0], self.shape[1])        
    

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** - 1)
    
    def __rtruediv__(self, other):
        other * (self ** -1)

    def __repr__(self):
        return f"Tensor : {self.data}"
    

