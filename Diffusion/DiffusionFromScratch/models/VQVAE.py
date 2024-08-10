import torch
from . import components
from .settings import DownsampleBlockConfig, MidBlockConfig, UpsampleBlockConfig, \
            LatentConfig


class VQVAE(torch.nn.Module):
    def __init__(self, 
                 down_block_config: DownsampleBlockConfig, 
                 mid_block_config: MidBlockConfig,
                 up_block_config: UpsampleBlockConfig,
                 latent_config: LatentConfig,
                 enc_dec_final_groupnorm_channel_dim: int):
        
        """
        Model consists of:
        Image --> Encoder --> Latent (Z) --> Decoder --> Output
        - Encoder section: 
            Conv --> Downblock --> MidBlock --> GroupNorm --> SiLU --> Conv --> Quantizer --> Latent (Z)

        - Decoder section: 
            Conv --> MidBlock --> Upblock --> GroupNorm --> SiLU --> Conv
        """
        super(VQVAE, self).__init__()

        #assertion of block sizes identical for encoder decoder
        assert len(down_block_config.down_channels_list) == len(up_block_config.up_channels_list)
        for idx in range(len(down_block_config.down_channels_list)):
            assert down_block_config.down_channels_list[idx] == up_block_config.up_channels_list[-idx-1]
        
        #assert connection channel size same
        assert down_block_config.down_channels_list[-1] == mid_block_config.mid_channels_list[0]
        assert up_block_config.up_channels_list[0] == mid_block_config.mid_channels_list[-1]

        self.encoder_first_conv = torch.nn.Conv2d(3, down_block_config.down_channels_list[0], kernel_size=3, padding=1, stride = 1)
        
    
        self.encoder_down_layers = torch.nn.ModuleList(
            [components.DownsampleBlock(in_channels = down_block_config.down_channels_list[idx], 
                                        out_channels= down_block_config.down_channels_list[idx+1], 
                                        layers_repetition = down_block_config.layers_in_each_block[idx], 
                                        do_downsample = True,
                                        resnet_settings= down_block_config.resnet_settings, 
                                        time_embedding_settings= down_block_config.time_embedding_settings,
                                        attention_block_settings = down_block_config.attention_block_settings)
            for idx in range(len(down_block_config.layers_in_each_block))]
        )


        self.encoder_mid_layers = torch.nn.ModuleList(
             [components.MidBlock(mid_block_config.mid_channels_list[idx], 
                                out_channels=mid_block_config.mid_channels_list[idx+1], 
                                layers_repetition = mid_block_config.layers_in_each_block[idx],
                                resnet_settings=mid_block_config.resnet_settings,
                                attention_block_settings=mid_block_config.attention_block_settings, 
                                time_embedding_settings=mid_block_config.time_embedding_settings)
            for idx in range(len(mid_block_config.layers_in_each_block))]
        )

        self.encoder_before_latent_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(enc_dec_final_groupnorm_channel_dim, down_block_config.down_channels_list[-1]),
            torch.nn.SiLU(), 
            torch.nn.Conv2d(down_block_config.down_channels_list[-1], latent_config.latent_channel_dim, 
                            kernel_size=3, padding=1, stride=1), 
            torch.nn.Conv2d(latent_config.latent_channel_dim, latent_config.latent_channel_dim, 
                            kernel_size=1)
        )


        self.decoder_first_conv = torch.nn.Sequential(
            torch.nn.Conv2d(latent_config.latent_channel_dim, 
                            latent_config.latent_channel_dim,
                            kernel_size = 1, padding=1, stride = 1),
            torch.nn.Conv2d(latent_config.latent_channel_dim, 
                            mid_block_config.mid_channels_list[-1], 
                            kernel_size=3, padding=1, stride = 1)   
        )

        self.decoder_mid_layers = torch.nn.ModuleList(
            [components.MidBlock(mid_block_config.mid_channels_list[idx], 
                                out_channels=mid_block_config.mid_channels_list[idx+1], 
                                layers_repetition = mid_block_config.layers_in_each_block[idx],
                                resnet_settings=mid_block_config.resnet_settings,
                                attention_block_settings=mid_block_config.attention_block_settings, 
                                time_embedding_settings=mid_block_config.time_embedding_settings)
            for idx in range(len(mid_block_config.layers_in_each_block))]
        )

        #although sequential is capable, we use ModuleList because we need to put multiple input in the forward
        self.decoder_up_layers = torch.nn.ModuleList(
            [components.UpsampleBlock(in_channels = up_block_config.up_channels_list[idx], 
                                        out_channels= up_block_config.up_channels_list[idx+1], 
                                        layers_repetition = up_block_config.layers_in_each_block[idx], 
                                        do_upsample = True,
                                        resnet_settings= up_block_config.resnet_settings, 
                                        time_embedding_settings= up_block_config.time_embedding_settings,
                                        attention_block_settings = up_block_config.attention_block_settings)
            for idx in range(len(up_block_config.layers_in_each_block))]
        )

        self.decoder_before_output = torch.nn.Sequential(
            torch.nn.GroupNorm(enc_dec_final_groupnorm_channel_dim, up_block_config.up_channels_list[-1]),
            torch.nn.SiLU(), 
            torch.nn.Conv2d(up_block_config.up_channels_list[-1], 3, 
                            kernel_size=3, padding=1, stride=1), #final conv, return to image size
        )

        self.learnable_embeddings = torch.nn.Embedding(latent_config.latent_codebook_size, latent_config.latent_channel_dim)


    def quantize(self, x):
        #squeeze layer from B C H W to B H*W C, to be compatible with embedding L C dim
        B, C, H, W = x.shape
        layerOutput = x.permute(0,2,3,1)
        layerOutput = layerOutput.reshape(layerOutput.size(0), -1, layerOutput.size(-1))
        
        #embedding dimension in the beginning is L C, make it B(1) L C
        learnable_embeddings_expanded = self.learnable_embeddings.weight[None,:]

        #find distance between each embedding (1 L C) and each B H*W C
        # Note that the 1 will be broadcasted, and have the same effect with using .repeat()
        # cdist --> if input is B×P×M and BxRxM, the output will have shape B×P×R, where 
        #   [b,p,r] is the distance value of point p and r in batch b .
        distance_matrix = torch.cdist(layerOutput, learnable_embeddings_expanded)
        
        #find the diminim
        #since we want to find among learnable_embeddings with minimum distance,
        #we do it on last layer (r), causing shape B*H*W(keep_dims is off)                                                                                     
        indices_minimum_distance = torch.argmin(distance_matrix, dim = -1)

        #select embedding weights, this will create [B*H*W, C]
        quantized_output = torch.index_select(self.learnable_embeddings, 0, indices_minimum_distance.view(-1))
        
        #to avoid uncertainty during learning, when we are updating one of the weights, the others as standard is frozen
        #e.g. cookbook_losses = updating the embedding cookbook vectors, which will frozen the encoder output
        #flatened_x should be in the same dims (B*H*W, C) with quantized_output
        layerOutput = layerOutput.reshape(-1, layerOutput.size(-1))
        cookbook_losses = torch.mean((quantized_output.detach() - layerOutput)**2)
        commitment_losses = torch.mean((quantized_output - layerOutput.detach())**2)   

        #make the quantized_output to be a constant so that x can be updated later with learning
        quantized_output = layerOutput + (quantized_output - layerOutput).detach()
        #reshape back to B C H W shape from B*H*W C
        quantized_output = quantized_output.reshape(B,H,W,C).permute(0,3,1,2)

        return quantized_output, cookbook_losses, commitment_losses 


    def encoder(self, x, time_embed = None):
        layerOutput = self.encoder_first_conv(x)
        for layer in self.encoder_down_layers:
            layerOutput = layer(layerOutput, time_embed)
        for layer in self.encoder_mid_layers:
            layerOutput = layer(layerOutput, time_embed)
        layerOutput = self.encoder_before_latent_layers(layerOutput)
        quantizedLatent, cookbook_losses, commitment_losses = self.quantize(self, layerOutput)
        return quantizedLatent, cookbook_losses,commitment_losses
    
    def decoder(self, x, time_embed = None):
        layerOutput = self.decoder_first_conv(x)
        for layer in self.decoder_mid_layers:
            layerOutput = layer(layerOutput, time_embed)
        for layer in self.decoder_up_layers:
            layerOutput = layer(layerOutput, time_embed)

        layerOutput = self.decoder_before_output(layerOutput)
        return layerOutput

    def forward(self, x, time_embed = None):
        encoder_quantized_latent, cookbook_losses,commitment_losses = self.encoder(x, time_embed)
        decoder_output = self.decoder(encoder_quantized_latent, time_embed)
        return decoder_output, cookbook_losses, commitment_losses
    

