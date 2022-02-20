import torch

x = torch.rand(3, requires_grad=True) #per calcular el gradient
print(x)

y = x + 2
print(y)
z = y*y*2
z = z.mean()
print(z)

z.backward() #dz/dx per calcular el gradient
print(x.grad)

#3 opcions per treure el grad
# x.requires_grad_(False)
#x.detach()
#with torch.no_grad():
#sempre que volguem resetejar el grad, x.grad.zero_()
