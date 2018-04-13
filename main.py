import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import argparse
from model import Generator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("type", default="DCGAN",help="Options to choose from DCGAN, WGAN, WGAN+GP, LSGAN")
parser.add_argument("image_size",default=64, help="Image size")
parser.add_argument("is_cuda",default=False, help="Is graphic card present?")
parser.add_argument("batch_size",default=64, help="batch_size")
parser.add_argument("epochs",default=2, help="Number of epochs to Train")

args = parser.parse_args()
print(args.type)

batch_size=64#args.batch_size
img_size = 128#args.image_size
is_cuda=False#args.is_cuda
NUM_OF_EPOCHS=2#args.epochs
lr=1e-3#args.lr
#Load Data
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

#Load Models
if not is_cuda:
	G=Generator()
	D=Discriminator()
else:
	G=Generator().cuda()
	D=Discriminator().cuda()

G.weight_init(0,0.02)
D.weight_init(0,0.02)

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(NUM_OF_EPOCHS):
	train_discriminator(D,G,train_loader,D_optimizer,args)


