class AttentionBlockSettings():
    def __init__(self, group_norm_channel_dim:int, attn_num_heads:int, attn_embed_dim:int, attn_batch_first: bool):
        """Settings for block consists of repetition:
            GroupNorm --> MultiHead Attention
            *Won't have parameter to indicate how many layers before connecting identity, handled in Downsample/Upsample block

        Params:
            group_norm_channel: int --> # of channels to group norm

            attn_num_heads: int --> # of heads for the multihead attention (i.e. # of scaled dot product attention)
                            (Each head (scaled-dot product attention) will be done in embedding dim / num_heads channel)
            attn_embed_dim: int --> # of embedding dimension for the multihead attention 
                            (typically, since each embedding dim is the same with input dim to avoid another linear layer to reshape)
            attn_batch_first: bool --> Whether the input is batch first or not.
                            If batch first, the Attention is in shape [B, L (sequence length), C (channel dim/embedding dim)]
                            If not batch first, the Attention is in shape [L, B, C]
        """
        self.group_norm_channel_dim = group_norm_channel_dim
        self.attn_num_heads = attn_num_heads
        self.attn_embed_dim = attn_embed_dim
        self.batch_first = attn_batch_first

class TimeEmbeddingSettings():
    def __init__(self, time_embedding_dim:int):
        """Settings for Time embedding processing block consists of 
            SiLU --> Linear
            
        Params:
            time_embedding_dim: int --> # of dimension of time embedding encoding
        """
        self.time_embedding_dim = time_embedding_dim

class ResnetSettings():
    """Settings for block consists of repetition:
        GroupNorm --> SiLU --> Conv
        *Won't have parameter to indicate how many layers before connecting identity, handled in Downsample/Upsample block

    Params:
        group_norm_channel_dim: int --> # of channels to group norm
    """
    def __init__(self, group_norm_channel_dim:int):
        self.group_norm_channel_dim = group_norm_channel_dim


class LatentConfig():
    """
    The latent channel dimension of
    """
    def __init__(self, latent_codebook_size: int, latent_channel_dim: int):
        self.latent_codebook_size = latent_codebook_size
        self.latent_channel_dim = latent_channel_dim


class DownsampleBlockConfig():
    def __init__(self, down_channels_list:list, layers_in_each_block:list, 
                 resnet_settings:ResnetSettings, 
                 time_embedding_settings:TimeEmbeddingSettings, 
                 attention_block_settings: AttentionBlockSettings):
        self.down_channels_list = down_channels_list # channels of downblock in each repetition
        self.layers_in_each_block = layers_in_each_block # layers of resnet+self attn repetition in each down block
        self.resnet_settings = resnet_settings
        self.time_embedding_settings = time_embedding_settings
        self.attention_block_settings = attention_block_settings
        assert len(self.down_channels_list) == len(self.layers_in_each_block) + 1 # because down_channels_list is a pair

class MidBlockConfig():
    def __init__(self, mid_channels_list:list, layers_in_each_block:list, 
                 resnet_settings:ResnetSettings, time_embedding_settings:TimeEmbeddingSettings, 
                 attention_block_settings: AttentionBlockSettings):
        self.mid_channels_list = mid_channels_list # channels of downblock in each repetition
        self.layers_in_each_block = layers_in_each_block # layers of resnet+self attn repetition in each down block
        self.resnet_settings = resnet_settings
        self.time_embedding_settings = time_embedding_settings
        self.attention_block_settings = attention_block_settings
        assert len(self.mid_channels_list) == len(self.layers_in_each_block) + 1

class UpsampleBlockConfig():
    def __init__(self, up_channels_list:list, layers_in_each_block:list,
                 resnet_settings:ResnetSettings, time_embedding_settings:TimeEmbeddingSettings, 
                 attention_block_settings: AttentionBlockSettings):
        self.up_channels_list = up_channels_list # channels of downblock in each repetition
        self.layers_in_each_block = layers_in_each_block # layers of resnet+self attn repetition in each down block
        self.resnet_settings = resnet_settings
        self.time_embedding_settings = time_embedding_settings
        self.attention_block_settings = attention_block_settings
        assert len(self.up_channels_list) == len(self.layers_in_each_block) + 1



class DiscriminatorConfig():
    def __init__(self, conv_channels_list:list, kernel_size_list: list, stride_list: list, padding_list: list):
        self.conv_channels_list = conv_channels_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.padding_list = padding_list