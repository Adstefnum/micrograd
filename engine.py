class Value:
    """
    A data structure to hold a value and its gradient.
    """
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
    
    def backward(self):
        pass

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
        out = Value(t)
        def backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = backward
        return out

    def __repr__(self):
        return f'Value(data={self.data}, gradient={self.grad})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        def backward():
            self.grad += 1 * out.grad #local grad * global grad (chain rule)
            other.grad += 1 * out.grad #it's += because we can have multiple paths to the same node and we need to note their contributions
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data**-1)

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data)
        def backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
        out._backward = backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return Value(-self.data)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**-1