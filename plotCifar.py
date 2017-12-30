# plot cifar 
import torch
import torchvision.utils as tutils
import matplotlib.pyplot as pl

transform=transforms.Compose([transforms.ToTensor(),])
trainset = cifarDataset.CIFAR100(root='/home/wuxiaodong/data', train=True, download=False, coarse=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=2)

testset = cifarDataset.CIFAR100(root='/home/wuxiaodong/data', train=False, download=False, coarse=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

data, fine_target, coarse_target = next(iter(testloader))

transformed_images = []
for i in range(20):
    transformed_images += [data[i]]

show(tutils.make_grid(transformed_images))
plt.show()
