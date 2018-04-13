import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd import Variable as V
import torch.optim as optim
import argparse
from model import Generator, Discriminator
from tensorboardX import SummaryWriter
import tqdm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="DCGAN",help="Options to choose from DCGAN, WGAN, WGAN+GP, LSGAN")
parser.add_argument("--image_size",default=64, help="Image size")
parser.add_argument("--is_cuda",default=False, help="Is graphic card present?")
parser.add_argument("--batch_size",default=64, help="batch_size")
parser.add_argument("--epochs",default=2, help="Number of epochs to Train")
parser.add_argument("--lr",default=1e-3, help="learning_rate")

args = parser.parse_args()
print(args)


writer = SummaryWriter()
batch_size=args.batch_size
img_size = args.image_size
is_cuda=args.is_cuda
NUM_OF_EPOCHS=args.epochs
lr=args.lr
# args={'batch_size':64,'image_size':img_size,"is_cuda":False,"NUM_OF_EPOCHS":2,"lr":1e-3}
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
	ind=0
	l=len(train_loader)
	for batch,labels in tqdm.tqdm(train_loader):
		z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
		#train_real
		if is_cuda:
			real_batch,noise=V(batch.cuda()),V(z_.cuda())
		else:
			real_batch,noise=V(batch),V(z_)	
		data={}
		data['real_batch']=real_batch
		data['noise']=noise	
		args['epoch']=epoch*l+ind
		ind=ind+1
		train_discriminator(D,G,data,D_optimizer,args,writer)
		train_generator(D,G,data,G_optimizer,args,writer)
	if epoch%10==0:
		validate(G,writer,epoch)	



