from nn import MLP
no_of_epochs = 20
learning_rate = -0.1 #why negative? because we are minimizing the loss function
loss_tol = 1e-5

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
    for k in range(no_of_epochs):
  
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
        
        print(k, loss.data)
        return loss.data, ypred, m
    

def main():
    loss, ypred, m = train()
    while loss > loss_tol:
        loss, ypred, m = train()
        
    print("Training done.")
    print(ypred)
    # print parameters and len of parameters
    print(m.parameters())
    print(len(m.parameters()))
    
if __name__ == "__main__":
    main()
