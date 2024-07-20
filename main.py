from nn import MLP
no_of_epochs = 20

def train():
    x = [1, 2, 3]
    m = MLP(3, [4, 5, 1])
    print(m(x))

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets
    print("Training model...")
    for k in range(no_of_epochs):
  
    # forward pass
        ypred = [m(x) for x in xs]
        print(ypred)
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        # backward pass
        for p in m.parameters():
            p.grad = 0.0
        loss.backward()
        
        # update
        for p in m.parameters():
            p.data += -0.1 * p.grad
        
        print(k, loss.data)
    print("Training done.")
    print(ypred)
    # print parameters and len of parameters
    print(m.parameters())
    print(len(m.parameters()))


if __name__ == "__main__":
    train()
