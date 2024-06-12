# import the  torch module 
import torch 
  
color_image = torch.rand(256, 256,3).to('cuda').requires_grad_()
mask = color_image.mean(dim=-1)
mask = 1 - torch.exp(-100*mask)
mask.retain_grad()

# generate another random mask
random_mask = torch.randn(256, 256).to('cuda').requires_grad_()

loss = torch.nn.functional.mse_loss(mask, random_mask)

# backpropagate
loss.backward()

# print the gradients
print(color_image.grad)
print(mask.grad)



  
