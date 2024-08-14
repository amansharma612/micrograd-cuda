from engine import Tensor
from mlp import MLP


model = MLP(4, [20, 1])


out = Tensor([[3]], shape=(1, 1))

out.to("gpu")

for i in range(10000):
    x = Tensor([[1], [1], [1], [12]], shape = (4, 1))
    x.to("gpu")
    y = model(x)

    loss = (out + y) ** 2
    loss.to("cpu")
    print(loss)
    loss.to("gpu")
    
    # y.to("cpu")
    # print(y)
    # y.to("gpu")

    loss.backward()

    for p in model.parameters():
        p.step(1e-4)   
        
    for p in model.parameters():
        p.zero_grad()


x = Tensor([[1], [1], [1], [12]], shape = (4, 1))
x.to("gpu")

y = model(x)
out.to("cpu")

y.to("cpu")
print("Output: ", y, "Target: ", out)






