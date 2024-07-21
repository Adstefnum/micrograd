import numpy as np

class Value:
    """
    A data structure to hold a value and its gradient.
    """
    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.children = set(children)
    
    def backward(self):
        
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0 # seed gradient, we set the gradient of the output node to 1
        for v in reversed(topo): #we run in reverse as the topo list saves the first nodes first but we need to backpropagate from the output node
            v._backward()

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
        out = Value(t, children=(self,))
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(np.exp(x), children=(self,))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f'Value(data={self.data}, gradient={self.grad})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other))
        def _backward():
            self.grad += 1 * out.grad #local grad * global grad (chain rule)
            other.grad += 1 * out.grad #it's += because we can have multiple paths to the same node and we need to note their contributions
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "The exponent must be an integer or a float."
        out = Value(self.data ** other, children=(self,))
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1 

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**-1