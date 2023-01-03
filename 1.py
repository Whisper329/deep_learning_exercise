import torch

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
m = torch.mm(tensor,tensor)
m2numpy = m.numpy()
print(f'the result is {m}.')
print(f'the result is {m2numpy}.')

