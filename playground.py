import torch 
a = torch.tensor([[0,2,3],[2,3,1]])
b = torch.tensor([7,8,9,10,11,12])

c = a.index_fill_(dim=0,index=b,)
print(c)

