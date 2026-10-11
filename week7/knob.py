state = 2


command = 4
w = 0

lr = 0.05
for step in range(10):

    prediction = w * state

    loss = (prediction - command) ** 2
    slope = 2 * (prediction - command) * state
    print(step, prediction, loss,w)

    w = w - lr * slope 
