import torch.nn as nn
import tqdm
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch 
def reset_grad(M):
	if not isinstance(M,list):
		M=[M]
	for m in M:	
		m.zero_grad()

def train_DCGAN_discriminator(D,G,data,D_optimizer,args,writer):
	y_real_ = V(torch.ones(args['batch_size']))
	y_fake_ = V(torch.zeros(args['batch_size']))
	BCE_loss=nn.BCELoss()
	if args['is_cuda']:
		y_real_,y_fake_=y_real_.cuda(),y_fake_.cuda()
	D_real_loss=BCE_loss(F.sigmoid(D(data['real_batch']).squeeze()),y_real_)
	G_result=G(data['noise'])
	D_fake_loss=BCE_loss(F.sigmoid(D(G_result).squeeze()),y_fake_)

	D_loss=D_real_loss+D_fake_loss
	D_loss.backward()
	D_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Discriminator/Error",D_loss,args['epoch'])
	writer.export_scalars_to_json("./all_scalars.json")

def train_DCGAN_generator(D,G,data,G_optimizer,args,writer):
	BCE_loss=nn.BCELoss()
	y_real_ = V(torch.ones(args['batch_size']))
	if args['is_cuda']:
		y_real_=y_real_.cuda()
	G_result=G(data['noise'])
	G_loss=BCE_loss(F.sigmoid(D(G_result).squeeze()),y_real_)
	G_loss.backward()
	G_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Generator/Error",G_loss,args['epoch'])
	writer.export_scalars_to_json("./all_scalars.json")



def train_WGAN_discriminator(D,G,data,D_optimizer,args,writer):
	for critic_repetitions in range(args['n_critic']):
		real_embed=D(data['real_batch'])
		G_result=G(data['noise'])
		fake_embed=D(G_result)
		Penalty=V(torch.zeros(1))

		if args['is_GP']:
			alpha=random.random()
			x_hat = (alpha*G_result+(1-alpha)*V(img_r)).detach()
			x_hat.requires_grad = True
			loss_D = D(x_hat).sum()
			loss_D.backward()
			x_hat.grad.volatile = False			
			Penalty=((x_hat.grad -1)**2 * args['LAMBDA']).mean()

		D_loss=(real_embed-fake_embed).mean()+Penalty
		D_loss.backward()
		D_optimizer.step()
		reset_grad([D,G])
		if not args['is_GP']:
			for p in D.parameters():
				p.data.clamp_(args["clamp_lower"],args["clamp_upper"])

		writer.add_scalar("Discriminator/Error",D_loss,args['epoch'])
		writer.export_scalars_to_json("./all_scalars.json")


def train_WGAN_generator(D,G,data,G_optimizer,args,writer):
	G_result=G(data['noise'])
	fake_embed=D(G_result)
	G_loss=-fake_embed.mean()
	G_loss.backward()
	G_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Generator/Error",G_loss,args['epoch'])
	writer.export_scalars_to_json("./all_scalars.json")



def train_LSGAN_discriminator(D,G,data,D_optimizer,args,writer):
	real_embed=D(data['real_batch'])
	G_result=G(data['noise'])
	fake_embed=D(G_result)

	D_loss=args['LAMBDA']*((real_embed-1)**2).mean()+((fake_embed)**2).mean()
	D_loss.backward()
	D_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Discriminator/Error",D_loss,args['epoch'])
	writer.export_scalars_to_json("./all_scalars.json")	



def train_LSGAN_generator(D,G,data,G_optimizer,args,writer):
	G_result=G(data['noise'])
	fake_embed=D(G_result)
	G_loss=args['LAMBDA']*args['LAMBDA']*((fake_embed-1)**2).mean()
	G_loss.backward()
	G_optimizer.step()
	reset_grad([D,G])
	writer.add_scalar("Generator/Error",G_loss,args['epoch'])
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

def validate(G,writer,epoch,type='DCGAN'):
	z_ = torch.randn((5, 100)).view(-1, 100, 1, 1)
	G_result=G(V(z_))
	writer.add_image("GeneratedImage/"+type,G_result,epoch)

