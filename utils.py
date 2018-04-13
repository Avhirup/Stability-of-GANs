import torch.nn as nn
import tqdm
from torch.autograd import Variable as V

def reset_grad(M):
	if not isinstance(M,list):
		M=[M]
	for m in M:	
		m.zero_grad()

def train_DCGAN_discriminator(D,G,train_loader,D_optimizer):
	y_real_ = torch.ones(batch_size)
	y_fake_ = torch.zeros(batch_size)
	BCE_loss=nn.BCELoss()
	for batch,labels in tqdm.tqdm(train_loader):
		z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
		#train_real
		if is_cuda:
			real_batch,y_real_,y_fake_,noise=V(batch.cuda()),V(y_real_.cuda()),V(y_fake_.cuda()),V(z_.cuda())
		else:
			real_batch,y_real_,y_fake_,noise=V(batch),V(y_real_),V(y_fake_),V(z_)
		D_real_loss=BCE_loss(D(real_batch),y_real_)
		G_result=G(noise)
		D_fake_loss=BCE_loss(D(G_result),y_fake_)

		D_loss=D_real_loss+D_fake_loss
		D_loss.backward()
		D_optimizer.step()
		reset_grad([D,G])
		print (1)
		break

def train_DCGAN_generator(D,G,train_loader):
	y_real_ = torch.ones(batch_size)
	BCE_loss=nn.BCELoss()
	for batch,labels in tqdm(train_loader):
		z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
		#train_real
		if is_cuda:
			real_batch,y_real_,y_fake_,noise=V(batch.cuda()),V(y_real_.cuda()),V(y_fake_.cuda()),V(z_.cuda())
		else:
			real_batch,y_real_,y_fake_,noise=V(batch),V(y_real_),V(y_fake_),V(z_)		
		G_result=G(noise)
		G_loss=BCE_loss(D(G_result),y_real_)
		G_loss.backward()
		G_optimizer.step()
		reset_grad([D,G])





def train_discriminator(D,G,train_loader,type="DCGAN"):
	#train on 
