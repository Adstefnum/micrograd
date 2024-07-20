from nn import MLP

no_of_epochs = 20
learning_rate = 0.1  # should be positive, negative here is likely a typo
loss_tol = 1e-5

def pretty_print_list(lst):
    print("\n".join(map(str, lst)))
    
def train():
    m = MLP(3, [4, 5, 1])
    
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets
    print("Training model...")
    
    epoch = 0
    while True:
        # forward pass
        ypred = [m(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        # backward pass
        for p in m.parameters():
            p.grad = 0.0
        loss.backward()
        
        # update
        for p in m.parameters():
            p.data += learning_rate * p.grad
        
        print(epoch, loss.data)
        
        # Check if loss is below tolerance or max epochs reached
        if loss.data < loss_tol:
            break
        
        epoch += 1
    
    print("Training done.")
    print(pretty_print_list(ypred))
    # print parameters and len of parameters
    print(pretty_print_list(m.parameters()))
    print("The number of parameters in the model is", len(m.parameters()))

if __name__ == "__main__":
    train()

