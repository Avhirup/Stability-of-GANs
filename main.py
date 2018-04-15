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
parser.add_argument("-type", default="DCGAN",help="Options to choose from DCGAN, WGAN, WGAN+GP, LSGAN",type=str)
parser.add_argument("-image_size",default=64, help="Image size",type=int)
parser.add_argument("-is_cuda",default=True, help="Is graphic card present?",type=bool)
parser.add_argument("-batch_size",default=64, help="batch_size",type=int)
parser.add_argument("-epochs",default=10, help="Number of epochs to Train",type=int)
parser.add_argument("-lr",default=2e-4, help="learning_rate",type=float)
parser.add_argument("-LAMBDA",default=1e-3, help="lambda",type=float)
parser.add_argument("-is_GP",default=False, help="To use Gradient Penalty",type=bool)
parser.add_argument("-n_critic",default=5, help="number of critic iterations per Generator iterations",type=int)
parser.add_argument("-clamp_upper",default=1e-2, help="weight clipping upper limit",type=float)
parser.add_argument("-clamp_lower",default=-1e-2, help="weight clipping lower limit",type=float)

args = parser.parse_args()
writer = SummaryWriter()
batch_size=args.batch_size
img_size = args.image_size
is_cuda=args.is_cuda
NUM_OF_EPOCHS=args.epochs
lr=args.lr
print (args.type,batch_size,img_size,is_cuda,NUM_OF_EPOCHS,lr)
#Load Data
print ("--------------------------------",img_size)
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
		z_ = torch.randn((batch.size()[0], 100)).view(-1, 100, 1, 1)
		#train_real
		if is_cuda:
			real_batch,noise=V(batch.cuda()),V(z_.cuda())
		else:
			real_batch,noise=V(batch),V(z_)	
		data={}
		data['real_batch']=real_batch
		data['noise']=noise	
		# break
		args.epoch=epoch*l+ind
		ind=ind+1
		train_discriminator(D,G,data,D_optimizer,args,writer,args.type)
		train_generator(D,G,data,G_optimizer,args,writer,args.type)
	# break
	if epoch%1==0:
		validate(G,writer,epoch,args,args.type)	



