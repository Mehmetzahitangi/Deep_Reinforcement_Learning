import torch
import numpy as np
import torch.nn as nn

# Tensors
a = torch.FloatTensor(3, 2)
print(a)
print(a.zero_())

print(torch.FloatTensor([[1,2,3],[3,2,1]]))

n = np.zeros(shape=(3, 2))
print(n)
b = torch.tensor(n)
print(b)

n = np.zeros(shape=(3, 2), dtype=np.float32)
print(torch.tensor(n))

# Scalar Tensors
print("---------------Scalar Tensors")
a = torch.tensor([1,2,3])
print(a)

s = a.sum()
print(s)
print(s.item())
print(torch.tensor(1))

# Tensor operations torch.stack(), torch.transpose(), and torch.cat()

# GPU Tensors
print("---------------GPU Tensors")
a = torch.FloatTensor([2,3])
print(a)

#ca = a.to("cuda")
#print(ca)

print(a+1)

#print(ca+1)
#print(ca.device)
# Gradients and tensors
print("---------------Gradients")
v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])

v_sum = v1 + v2
v_res = (v_sum*2).sum()
print(v_res)
print("leaf", v1.is_leaf, v2.is_leaf)
print("leaf", v_sum.is_leaf, v_res.is_leaf)
print("Grad", v1.requires_grad,  v2.requires_grad, v_sum.requires_grad, v_res.requires_grad )

v_res.backward()
print(v1.grad)

# NN blocks
print("---------------NN blocks")
l = nn.Linear(2, 5) # two input 5 output
v = torch.FloatTensor([1, 2])
print(l(v))

# Custom Layers
print("---------------Custom Layers")
class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

def forward(self, x):
        return self.pipe(x)

if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(net)
    print(out)