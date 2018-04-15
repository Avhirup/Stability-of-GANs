import torch.nn as nn
import tqdm
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch 
import random
def reset_grad(M):
	if not isinstance(M,list):
		M=[M]
	for m in M:	
		m.zero_grad()

def train_DCGAN_discriminator(D,G,data,D_optimizer,args,writer):
	y_real_ = V(torch.ones(data['real_batch'].size()[0]))
	y_fake_ = V(torch.zeros(data['noise'].size()[0]))
	BCE_loss=nn.BCELoss()
	if args.is_cuda:
		y_real_,y_fake_=y_real_.cuda(),y_fake_.cuda()
	D_real_loss=BCE_loss(D(data['real_batch']).squeeze(),y_real_)
	G_result=G(data['noise'])
	# print ("D_real_loss",D_real_loss)
	D_fake_loss=BCE_loss(D(G_result).squeeze(),y_fake_)
	# print ("D_fake_loss",D_fake_loss)
	D_loss=D_real_loss+D_fake_loss
	D_loss.backward()
	D_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Discriminator/Error",D_loss,args.epoch)
	writer.export_scalars_to_json("./all_scalars.json")

def train_DCGAN_generator(D,G,data,G_optimizer,args,writer):
	BCE_loss=nn.BCELoss()
	y_real_ = V(torch.ones(data['noise'].size()[0]))
	if args.is_cuda:
		y_real_=y_real_.cuda()
	G_result=G(data['noise'])
	G_loss=BCE_loss(D(G_result).squeeze(),y_real_)
	# print ("G_real_loss",G_loss)
	G_loss.backward()
	G_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Generator/Error",G_loss,args.epoch)
	writer.export_scalars_to_json("./all_scalars.json")



def train_WGAN_discriminator(D,G,data,D_optimizer,args,writer):
	for critic_repetitions in range(args.n_critic):
		real_embed=D(data['real_batch']).mean()
		G_result=G(data['noise'])
		fake_embed=D(G_result).mean()
		if args.is_cuda:
			Penalty=V(torch.zeros(1)).cuda()
		else:
			Penalty=V(torch.zeros(1))

		if args.is_GP:
			alpha=random.random()
			x_hat = (alpha*G_result+(1-alpha)*data['real_batch']).detach()
			x_hat.requires_grad = True
			loss_D = D(x_hat).sum()
			loss_D.backward()
			x_hat.grad.volatile = False			
			Penalty=(((x_hat.grad -1)**2 ).mean())* args.LAMBDA

		D_loss=-(real_embed-fake_embed).mean()+Penalty
		D_loss.backward()
		D_optimizer.step()
		if not args.is_GP:
			for p in D.parameters():
				p.data.clamp_(args.clamp_lower,args.clamp_upper)

		reset_grad([D,G])
		writer.add_scalar("Discriminator/Error",D_loss,args.epoch)
		writer.export_scalars_to_json("./all_scalars.json")


def train_WGAN_generator(D,G,data,G_optimizer,args,writer):
	G_result=G(data['noise'])
	fake_embed=D(G_result)
	G_loss=-fake_embed.mean()
	G_loss.backward()
	G_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Generator/Error",G_loss,args.epoch)
	writer.export_scalars_to_json("./all_scalars.json")



def train_LSGAN_discriminator(D,G,data,D_optimizer,args,writer):
	real_embed=D(data['real_batch'])
	G_result=G(data['noise'])
	fake_embed=D(G_result)

	D_loss=((real_embed-1)**2).mean()+args.LAMBDA*((fake_embed)**2).mean()
	D_loss.backward()
	D_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Discriminator/Error",D_loss,args.epoch)
	writer.export_scalars_to_json("./all_scalars.json")	



def train_LSGAN_generator(D,G,data,G_optimizer,args,writer):
	G_result=G(data['noise'])
	fake_embed=D(G_result)
	G_loss=args.LAMBDA*((fake_embed-1)**2).mean()
	G_loss.backward()
	G_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Generator/Error",G_loss,args.epoch)
	writer.export_scalars_to_json("./all_scalars.json")


def train_discriminator(D,G,data,D_optimizer,args,writer,type="DCGAN"):
	if type=='DCGAN':
		train_DCGAN_discriminator(D,G,data,D_optimizer,args,writer)
	elif type=='WGAN':
		train_WGAN_discriminator(D,G,data,D_optimizer,args,writer)
	elif type=='LSGAN':
		train_LSGAN_discriminator(D,G,data,D_optimizer,args,writer)
	else:
		print("Implementation Remaining")
		pass


def train_generator(D,G,data,G_optimizer,args,writer,type="DCGAN"):
	if type=='DCGAN':
		train_DCGAN_generator(D,G,data,G_optimizer,args,writer)
	elif type=='WGAN':
		train_WGAN_generator(D,G,data,G_optimizer,args,writer)		
	elif type=='LSGAN':
		train_LSGAN_generator(D,G,data,G_optimizer,args,writer)		
	else:
		print("Implementation Remaining")
		pass

def validate(G,writer,epoch,args,type='DCGAN'):
	z_ = torch.randn((5, 100)).view(-1, 100, 1, 1)
	if args.is_cuda:
		G_result=G(V(z_).cuda())
	else:
		G_result=G(V(z_))
	writer.add_image("GeneratedImage/"+type,G_result,epoch)

