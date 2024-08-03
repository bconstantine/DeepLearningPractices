import torch
import math

class SinusoidalTimeEmbeddings(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        assert dimension % 2 == 0 # dimension must be even

    @torch.no_grad()
    def facebookresearch_approach_calculation(self, time):
        """
        From:
        https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py
        """
        device = time.device
        half_dimension = self.dimension // 2 #half going to be sin, half going to be cos
        embeddings = math.log(10000) / (half_dimension - 1)
        embeddings = torch.exp(torch.arange(half_dimension, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
    @torch.no_grad()
    def original_transformer_calculation(self, time):
        """
        From: 
        https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/models/blocks.py
        """
        device = time.device
        half_dimension = self.dimension // 2 #half going to be sin, half going to be cos
        # factor = 10000^(2i/d_model), where i correspond to the a range of 0 to d_model/2
        factor = 10000 ** ((torch.arange(half_dimension, dtype=torch.float32, device=device) / half_dimension)
        )
        
        # pos / factor
        # timesteps B -> B, 1 -> B, temb_dim
        t_emb = time[:, None].repeat(1, half_dimension) / factor
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        return t_emb