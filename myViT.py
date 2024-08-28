import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)

#Turn the image into patches 
def patchify(images, n_patches):
	n, c, h, w = images.shape

	assert h == w, "Patchify method is implemented for square images only"

	patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
	patch_size = h // n_patches

	for idx, image in enumerate(images):
		for i in range(n_patches):
			for j in range(n_patches):
				patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
				patches[idx, i * n_patches + j] = patch.flatten()
	return patches

#Multi-head self-attention
class MyMSA(nn.Module):
	def __init__(self, d, n_heads=2):
		super(MyMSA, self).__init__()
		self.d = int(d/2)
		self.n_heads = n_heads		#This number of heads specifies how many heads per each input type

		assert int(d/2) % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

		d_head = int(self.d / n_heads)
		self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
		self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
		self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
		self.d_head = d_head
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, sequences):
		# Sequences has shape (N, seq_length, token_dim)
		# We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
		# And come back to    (N, seq_length, item_dim)  (through concatenation)
		
		img_sequences, mask_sequences = torch.split(sequences, int(np.shape(sequences)[-1] / 2), 2)
		
		#Image Heads	
		img_result = []
		for sequence in img_sequences:
			seq_result = []
			for head in range(self.n_heads):
				q_mapping = self.q_mappings[head]
				k_mapping = self.k_mappings[head]
				v_mapping = self.v_mappings[head]

				seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
				q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

				attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
				seq_result.append(attention @ v)
				
				
			img_result.append(torch.hstack(seq_result))
		
		#Mask Heads	
		mask_result = []
		for sequence in mask_sequences:
			seq_result = []
			for head in range(self.n_heads):
				q_mapping = self.q_mappings[head]
				k_mapping = self.k_mappings[head]
				v_mapping = self.v_mappings[head]

				seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
				q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

				attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
				seq_result.append(attention @ v)
				
				
			mask_result.append(torch.hstack(seq_result))

		#Concatenate head results
		img_concs = torch.cat([torch.unsqueeze(r, dim=0) for r in img_result])
		mask_concs = torch.cat([torch.unsqueeze(r, dim=0) for r in mask_result])
			
		return torch.cat((img_concs, mask_concs), 2)


class MyViTBlock(nn.Module):
	def __init__(self, hidden_d, n_heads, mlp_ratio=4):
		super(MyViTBlock, self).__init__()
		self.hidden_d = hidden_d
		self.n_heads = n_heads

		self.norm1_i = nn.LayerNorm( int(hidden_d / 2) )
		self.norm1_m = nn.LayerNorm( int(hidden_d / 2) )
		
		self.mhsa = MyMSA(hidden_d, n_heads)
		
		self.norm2_i = nn.LayerNorm( int(hidden_d / 2) )
		self.norm2_m = nn.LayerNorm( int(hidden_d / 2) )
		
		self.mlp_i = nn.Sequential(nn.Linear(int(hidden_d / 2), mlp_ratio * int(hidden_d / 2)), nn.GELU(), nn.Linear(mlp_ratio * int(hidden_d / 2), int(hidden_d / 2)))
		self.mlp_m = nn.Sequential(nn.Linear(int(hidden_d / 2), mlp_ratio * int(hidden_d / 2)), nn.GELU(), nn.Linear(mlp_ratio * int(hidden_d / 2), int(hidden_d / 2)))

	def forward(self, x):
	
		#Split image and mask inputs
		x_i, x_m = torch.split(x, int(self.hidden_d / 2), 2)
		
		#Normalize input types (images and masks) separately
		norm_i = self.norm1_i(x_i)
		norm_m = self.norm1_m(x_m)
		
		#Apply multi head self attention
		out = self.mhsa( torch.cat((norm_i, norm_m), 2) )
		out_i, out_m = torch.split(out, int(self.hidden_d / 2), 2)
		
		#Add residual connections
		out_i = out_i + x_i
		out_m = out_m = x_m
		
		#Normalize again
		norm2_i = self.norm2_i(out_i)
		norm2_m = self.norm2_m(out_m)
	
		#Multilayer perception and residual connections
		out_i = out_i + self.mlp_i(norm2_i)
		out_m = out_m + self.mlp_m(norm2_m)
		
		#Return
		return torch.cat((out_i, out_m), 2)


class ViT(nn.Module):
	def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=32, n_heads=2, out_d=10):
		# Super constructor
		super(ViT, self).__init__()

		# Attributes
		self.chw = chw # ( C , H , W )
		self.n_patches = n_patches
		self.n_blocks = n_blocks
		self.n_heads = n_heads
		self.hidden_d = hidden_d

		# Input and patches sizes
		assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
		assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
		self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

		# 1) Linear mapper
		self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
		self.linear_mapper = nn.Linear(self.input_d, int(self.hidden_d / 2) )		#Linear(in_features_length, out_features_length, [...])
		self.linear_mapper_mask = nn.Linear(self.input_d, int(self.hidden_d / 2) )	

		# 2) Learnable classification token
		self.class_token = nn.Parameter(torch.rand(1, int(self.hidden_d / 2) ))	#Generate one extra token for each of the enbeddings that comes out the Linear Mapper

		# 3) Positional embedding - uses Torch.nn's register_buffer as it does not require training
		self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, int(hidden_d / 2)), persistent=False)
		#It does the same as:
		#self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d / 2)))
		#self.pos_embed.requires_grad = False

		# 4) Transformer encoder blocks
		self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

		# 5) Classification MLPk
		self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))
	
	def forward(self, inputs):
	
		images = inputs[0]
		masks = inputs[1]
	
		# Dividing image inputs into patches
		n, c, h, w = images.shape
		img_patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
		mask_patches = patchify(masks, self.n_patches).to(self.positional_embeddings.device)
		
		#print('Patches shape', np.shape(img_patches))

		# Running linear layer tokenization
		# Map the vector corresponding to each patch to the hidden size dimension
		img_tokens = self.linear_mapper(img_patches)
		mask_tokens = self.linear_mapper_mask(mask_patches)
		
		#print('Mapped patches', np.shape(img_tokens))

		# Adding classification token to the tokens
		img_tokens = torch.cat((self.class_token.expand(n, 1, -1), img_tokens), dim=1)
		mask_tokens = torch.cat((self.class_token.expand(n, 1, -1), mask_tokens), dim=1)

		#print('Patches with token shape', np.shape(img_tokens))

		# Adding positional embedding
		img_out = img_tokens + self.positional_embeddings.repeat(n, 1, 1)
		masks_out = mask_tokens + self.positional_embeddings.repeat(n, 1, 1)
		out = torch.cat((img_out, masks_out), 2)
				
		#print('Concatenated', np.shape(out))
				
		# Transformer Blocks
		for block in self.blocks:
			out = block(out)

		o_i, o_m = torch.split(out, int(self.hidden_d / 2), 2)

		# Getting the classification tokens only
		img_out = o_i[:, 0]
		masks_out = o_m[:, 0]
		
		out = torch.cat((img_out, masks_out), 1)

		return self.mlp(out) # Map to output dimension, output category distribution
    

#Sequence length is the length of a sequence; d is the dimension received as input, which came from the linear mapper
def get_positional_embeddings(sequence_length, d):
	result = torch.ones(sequence_length, d)
	for i in range(sequence_length):
		for j in range(d):
			result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
	return result


def main():
	return
	

if __name__ == '__main__':
	main()


