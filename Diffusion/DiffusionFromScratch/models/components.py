import torch

from .settings import ResnetSettings, TimeEmbeddingSettings, AttentionBlockSettings

class DownsampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers_repetition,
                 do_downsample: bool, 
                 resnet_settings: ResnetSettings, 
                 time_embedding_settings: TimeEmbeddingSettings = None,
                 attention_block_settings: AttentionBlockSettings = None,):
        """Down sample block for VAE / VQVAE, Input should be in shape of (B, C, H, W)
        Downsampling only happens at the end of the block, the rest of repetition will retain the same (h,w) dimension
        (Channel transformation at the first resnet block)
        Params: 
            in_channels: int --> Number of input channels
            out_channels: int --> Number of output channels
            layers_repetition: int --> Number of layers in the block 
                (Repetition of: resnet --> [Attention])
                Note that the block enclosed by [] is optional
                Finally, enclosed by 1x Downsample conv layer
            do_downsample: bool --> whether downsample is done or not (compress h w dimension)
            resnet_settings: ResnetSettings --> Settings for the resnet block
            time_embedding_settings: TimeEmbeddingSettings --> Settings for the time embedding block
            attention_block_settings: AttentionBlockSettings --> Settings for the attention block
        """
        super(DownsampleBlock, self).__init__()

        #assertion
        assert resnet_settings.group_norm_channel_dim <= in_channels #GroupNorm channel dimension should be less than input channel dimension
        if attention_block_settings:
            assert attention_block_settings.attn_num_heads <= attention_block_settings.attn_embed_dim #Make the same to avoid adding another linear layer
            assert attention_block_settings.attn_embed_dim == out_channels #Make the same to avoid adding another linear layer
            assert attention_block_settings.batch_first == True #Shape consistency between components
        
        self.layers_repetition = layers_repetition


        self.resnet_layers_before_timeembedding = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.GroupNorm(resnet_settings.group_norm_channel_dim, in_channels if idx == 0 else out_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(in_channels if idx ==0 else out_channels, out_channels, 
                                    kernel_size=3, 
                                    padding=1,
                                    stride=1)
                )
                for idx in range(layers_repetition)
            ]
        )

        self.time_embedding_layers = None
        if time_embedding_settings != None:
            self.time_embedding_layers = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Linear(time_embedding_settings.time_embedding_dim, out_channels)
                    )
                    for _ in range(layers_repetition)
                ]
            )
        
        self.resnet_layers_after_timeembedding = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.GroupNorm(resnet_settings.group_norm_channel_dim, out_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=3, 
                                    padding=1,
                                    stride=1)
                )
                for _ in range(layers_repetition)
            ]
        )

        #For projecting possible mismatch of channel for residual input (especially first layer)
        self.residual_input_channel_transform = self.residual_input_conv = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, kernel_size=1)
                for idx in range(layers_repetition)
            ]
        )

        self.attention_multiheads = None
        self.attention_norms = None
        if attention_block_settings != None:
            #attention norms and multi head attention needs to be separate as their input shape is different
            #attention_norms input should be in (B, C, H*W)
            #attention multi head input should be in (B, L (sequence length), F(feature vector)) when batch_first
            #   whereas in case of image, L = H*W and F = Channels (feature vector of each pixel)
            self.attention_norms = torch.nn.ModuleList(
                [
                    torch.nn.GroupNorm(attention_block_settings.group_norm_channel_dim, out_channels)
                    for _ in range (layers_repetition)
                ]
            )
            self.attention_multiheads = torch.nn.ModuleList(
                [
                    torch.nn.MultiheadAttention(embed_dim=attention_block_settings.attn_embed_dim, 
                                                    num_heads=attention_block_settings.attn_num_heads, 
                                                    batch_first=attention_block_settings.batch_first)
                    for _ in range(layers_repetition)
                ]
            )
        
        #downsample the hxw dimension
        self.downsample_conv = torch.nn.Identity()
        if do_downsample:
            self.downsample_conv = torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding = 1)

    def forward(self, x, time_embedding=None):
        layer_output = x
        for layerIdx in range(self.layers_repetition):
            residual_input = layer_output
            layer_output = self.resnet_layers_before_timeembedding[layerIdx](layer_output)

            #add time embedding
            if self.time_embedding_layers and not time_embedding:
                time_embed_output = self.time_embedding_layers[layerIdx](time_embedding)
                layer_output = layer_output + time_embed_output[:,:,None, None]
            layer_output = self.resnet_layers_after_timeembedding[layerIdx](layer_output)

            #Add identity
            layer_output = layer_output + self.residual_input_channel_transform[layerIdx](residual_input)

            if self.attention_multiheads and self.attention_norms:
                #attention block is 3D, so h and w is usually flattened (h*w)
                batch_size, channels, h, w = layer_output.shape
                attnInput = layer_output.reshape(batch_size, channels, h*w)
                attnInput = self.attention_norms[layerIdx](attnInput)
                attnInput = attnInput.transpose(1,2)
                attnOutput, _ = self.attention_multiheads[layerIdx](attnInput, attnInput, attnInput)
                attnOutput = attnOutput.transpose(1,2).reshape(batch_size, channels, h, w)
                #add identity of layer_output
                layer_output = layer_output + attnOutput
        
        layer_output = self.downsample_conv(layer_output)
        return layer_output


class MidBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers_repetition,
                 resnet_settings: ResnetSettings, 
                 attention_block_settings: AttentionBlockSettings,
                 time_embedding_settings: TimeEmbeddingSettings = None):
        """Mid block for VAE / VQVAE, Input should be in shape of (B, C, H, W)
            Retains same sequence dimension (w,h), transform channel on the first resnet
            Params: 
                in_channels: int --> Number of input channels
                out_channels: int --> Number of output channels
                layers_repetition: int --> Number of layers in the block 
                    Resnet --> (Repetition of:  Resnet --> Attention)
                resnet_settings: ResnetSettings --> Settings for the resnet block
                attention_block_settings: AttentionBlockSettings --> Settings for the attention block
                time_embedding_settings: TimeEmbeddingSettings --> Settings for the time embedding block
        """
        super(MidBlock, self).__init__()

        #assertion
        assert resnet_settings.group_norm_channel_dim <= in_channels #GroupNorm channel dimension should be less than input channel dimension
        assert attention_block_settings.attn_num_heads <= attention_block_settings.attn_embed_dim #Make the same to avoid adding another linear layer
        assert attention_block_settings.attn_embed_dim == out_channels #Make the same to avoid adding another linear layer
        assert attention_block_settings.batch_first == True #Shape consistency between components
        
        self.layers_repetition = layers_repetition

        self.resnet_layers_before_timeembedding = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.GroupNorm(resnet_settings.group_norm_channel_dim, in_channels if idx == 0 else out_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(in_channels if idx ==0 else out_channels, out_channels, 
                                    kernel_size=3, 
                                    padding=1,
                                    stride=1)
                )
                for idx in range(layers_repetition+1) #+1 due to we have extra resnet in the beginning
            ]
        )

        self.time_embedding_layers = None
        if time_embedding_settings != None:
            self.time_embedding_layers = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Linear(time_embedding_settings.time_embedding_dim, out_channels)
                    )
                    for _ in range(layers_repetition) #+1 due to we have extra resnet in the beginning
                ]
            )
        
        self.resnet_layers_after_timeembedding = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.GroupNorm(resnet_settings.group_norm_channel_dim, out_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=3, 
                                    padding=1,
                                    stride=1)
                )
                for _ in range(layers_repetition+1) #+1 due to we have extra resnet in the beginning
            ]
        )

        #For projecting possible mismatch of channel for residual input (especially first layer)
        self.residual_input_channel_transform = self.residual_input_conv = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, kernel_size=1)
                for idx in range(layers_repetition+1) #+1 due to we have extra resnet in the beginning
            ]
        )

        
        #attention norms and multi head attention needs to be separate as their input shape is different
        #attention_norms input should be in (B, C, H*W)
        #attention multi head input should be in (B, L (sequence length), F(feature vector)) when batch_first
        #   whereas in case of image, L = H*W and F = Channels (feature vector of each pixel)
        self.attention_norms = torch.nn.ModuleList(
            [
                torch.nn.GroupNorm(attention_block_settings.group_norm_channel_dim, out_channels)
                for _ in range (layers_repetition)
            ]
        )
        self.attention_multiheads = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(embed_dim=attention_block_settings.attn_embed_dim, 
                                                num_heads=attention_block_settings.attn_num_heads, 
                                                batch_first=attention_block_settings.batch_first)
                for _ in range(layers_repetition)
            ]
        )
        
    def forward(self, x, time_embedding=None):
        layer_output = x

        #first resnet
        residual_input = layer_output
        layer_output = self.resnet_layers_before_timeembedding[0](layer_output)

        #add time embedding
        if self.time_embedding_layers and not time_embedding:
            time_embed_output = self.time_embedding_layers[0](time_embedding)
            layer_output = layer_output + time_embed_output[:,:,None, None]
        layer_output = self.resnet_layers_after_timeembedding[0](layer_output)

        #Add identity
        layer_output = layer_output + self.residual_input_channel_transform[0](residual_input)

        for layerIdx in range(self.layers_repetition):
            #attention block is 3D, so h and w is usually flattened (h*w)
            batch_size, channels, h, w = layer_output.shape
            attnInput = layer_output.reshape(batch_size, channels, h*w)
            attnInput = self.attention_norms[layerIdx](attnInput)
            attnInput = attnInput.transpose(1,2)
            attnOutput, _ = self.attention_multiheads[layerIdx](attnInput, attnInput, attnInput)
            attnOutput = attnOutput.transpose(1,2).reshape(batch_size, channels, h, w)
            #add identity of layer_output
            layer_output = layer_output + attnOutput

            
            residual_input = layer_output
            layer_output = self.resnet_layers_before_timeembedding[layerIdx+1](layer_output)
            #add time embedding
            if self.time_embedding_layers and not time_embedding:
                time_embed_output = self.time_embedding_layers[layerIdx+1](time_embedding)
                layer_output = layer_output + time_embed_output[:,:,None, None]
            layer_output = self.resnet_layers_after_timeembedding[layerIdx+1](layer_output)

            #Add identity
            layer_output = layer_output + self.residual_input_channel_transform[layerIdx + 1](residual_input)
        
        return layer_output
    
class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, layers_repetition,
                 do_upsample: bool, 
                 resnet_settings: ResnetSettings, 
                 time_embedding_settings: TimeEmbeddingSettings = None,
                 attention_block_settings: AttentionBlockSettings = None,):
        """Up sample block for VAE / VQVAE, Input should be in shape of (B, C, H, W)
        Upsampling only happens at the beginning of the block (change h,w), 
        and channel transformation happens at the first resnet
        the rest retains the same shape
        Params: 
            in_channels: int --> Number of input channels
            out_channels: int --> Number of output channels
            layers_repetition: int --> Number of layers in the block 
                Upsample --> Concat with down block output -->(Repetition of: --> [Attention])
                Note that the block enclosed by [] is optional
            do_upsample: bool --> whether upsample is done or not
            resnet_settings: ResnetSettings --> Settings for the resnet block
            time_embedding_settings: TimeEmbeddingSettings --> Settings for the time embedding block
            attention_block_settings: AttentionBlockSettings --> Settings for the attention block
        """
        super(UpsampleBlock, self).__init__()

        #assertion
        assert resnet_settings.group_norm_channel_dim <= in_channels #GroupNorm channel dimension should be less than input channel dimension
        if attention_block_settings:
            assert attention_block_settings.attn_num_heads <= attention_block_settings.attn_embed_dim #Make the same to avoid adding another linear layer
            assert attention_block_settings.attn_embed_dim == out_channels #Make the same to avoid adding another linear layer
            assert attention_block_settings.batch_first == True #Shape consistency between components
        
        self.layers_repetition = layers_repetition

        firstResnetChannel = in_channels*2 #concatenated channel from down block
        self.resnet_layers_before_timeembedding = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.GroupNorm(resnet_settings.group_norm_channel_dim, firstResnetChannel if idx == 0 else out_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(firstResnetChannel if idx ==0 else out_channels, out_channels, 
                                    kernel_size=3, 
                                    padding=1,
                                    stride=1)
                )
                for idx in range(layers_repetition)
            ]
        )

        self.time_embedding_layers = None
        if time_embedding_settings != None:
            self.time_embedding_layers = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Linear(time_embedding_settings.time_embedding_dim, out_channels)
                    )
                    for _ in range(layers_repetition)
                ]
            )
        
        self.resnet_layers_after_timeembedding = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.GroupNorm(resnet_settings.group_norm_channel_dim, out_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=3, 
                                    padding=1,
                                    stride=1)
                )
                for _ in range(layers_repetition)
            ]
        )

        #For projecting possible mismatch of channel for residual input (especially first layer)
        self.residual_input_channel_transform = self.residual_input_conv = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(firstResnetChannel if idx == 0 else out_channels, out_channels, kernel_size=1)
                for idx in range(layers_repetition)
            ]
        )

        self.attention_multiheads = None
        self.attention_norms = None
        if attention_block_settings != None:
            #attention norms and multi head attention needs to be separate as their input shape is different
            #attention_norms input should be in (B, C, H*W)
            #attention multi head input should be in (B, L (sequence length), F(feature vector)) when batch_first
            #   whereas in case of image, L = H*W and F = Channels (feature vector of each pixel)
            self.attention_norms = torch.nn.ModuleList(
                [
                    torch.nn.GroupNorm(attention_block_settings.group_norm_channel_dim, out_channels)
                    for _ in range (layers_repetition)
                ]
            )
            self.attention_multiheads = torch.nn.ModuleList(
                [
                    torch.nn.MultiheadAttention(embed_dim=attention_block_settings.attn_embed_dim, 
                                                    num_heads=attention_block_settings.attn_num_heads, 
                                                    batch_first=attention_block_settings.batch_first)
                    for _ in range(layers_repetition)
                ]
            )

        #upsample the hw dimension
        self.upsample_conv = torch.nn.Identity()
        if do_upsample:
            upsample_original_channel = out_channels
            # in channels will be original input.shape[1] + down_block_shape.shape[1], 
            #   hence this conv dim should be in_channels//2
            self.upsample_conv = torch.nn.ConvTranspose2d(upsample_original_channel, upsample_original_channel, kernel_size=4, stride=2, padding = 1)

    def forward(self, x, down_block_output = None, time_embedding=None):
        layer_output = x
        #layer_output = self.upsample_conv(layer_output)

        if down_block_output != None:
            layer_output = torch.cat([layer_output, down_block_output], dim = 1) #append on channel dim
        

        for layerIdx in range(self.layers_repetition):
            residual_input = layer_output
            layer_output = self.resnet_layers_before_timeembedding[layerIdx](layer_output)

            #add time embedding
            if self.time_embedding_layers and not time_embedding:
                time_embed_output = self.time_embedding_layers[layerIdx](time_embedding)
                layer_output = layer_output + time_embed_output[:,:,None, None]
            layer_output = self.resnet_layers_after_timeembedding[layerIdx](layer_output)

            #Add identity
            layer_output = layer_output + self.residual_input_channel_transform[layerIdx](residual_input)

            if self.attention_multiheads and self.attention_norms:
                #attention block is 3D, so h and w is usually flattened (h*w)
                batch_size, channels, h, w = layer_output.shape
                attnInput = layer_output.reshape(batch_size, channels, h*w)
                attnInput = self.attention_norms[layerIdx](attnInput)
                attnInput = attnInput.transpose(1,2)
                attnOutput, _ = self.attention_multiheads[layerIdx](attnInput, attnInput, attnInput)
                attnOutput = attnOutput.transpose(1,2).reshape(batch_size, channels, h, w)
                #add identity of layer_output
                layer_output = layer_output + attnOutput
        
        layer_output = self.upsample_conv(layer_output)
        return layer_output
    