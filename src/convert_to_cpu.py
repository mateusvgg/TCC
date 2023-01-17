import torch

X = torch.load('./X_tensor.pt')
Y = torch.load('./y_tensor.pt')

for i in range(len(X)):
    for j in range(len(X[i])):
        X[i][j] = X[i][j].to(torch.device('cpu'))

print(X[0][1].device)
print(Y)

torch.save(X, 'X_tensor_cpu.pt')
