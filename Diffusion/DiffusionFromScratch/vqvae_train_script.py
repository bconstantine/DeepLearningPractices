import torch
import torchvision
from datasets_tools import torchvision_datasets
from models.discriminator import PatchGANDiscriminator
from models.LPIPS import LPIPS_5Layer
from models.VQVAE import VQVAE
from models import settings
from tqdm import tqdm
import os
from datetime import datetime
import argparse
import json

def format_needed_gpu(multi_device_gpu_count: int, cuda_device_count: int) -> str:
    if torch.cuda.is_available():
        if multi_device_gpu_count < cuda_device_count:
            return f"cuda:{multi_device_gpu_count}"
        else:
            return "cuda"
    else:
        return "cpu"

def train_vqvae(torchvision_dataset_name: str = "LFWPeople", 
                batch_size: int = 128, #decrease batch size
                image_size: int = 256,
                data_root_path = "./data", 
                num_epochs: int = 20,
                partial_train_image_save_path: str = "./partial_train_images",
                show_partial_train_image_epoch: int = 0, #set to 0 if you don't want to show partial images
                codebook_loss_scale: float = 1,
                commitment_loss_scale: float = 0.25,
                discriminator_loss_scale_for_generator: float = 0.5,
                discriminator_loss_scale_for_discriminator: float = 0.5,
                vqvae_lr:float = 1e-5,
                discriminator_lr: float = 5e-6,
                model_save_path: str = "./model_output",
                save_partial_checkpoint_epoch: int = 0, #set to 0 if you don't want to save partial checkpoints
                custom_gpu_mapping = {"LPIPS": 1, "VQVAE": 0, "Discriminator": 0}
                ):
    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count > 1:
        print(f"Using {cuda_device_count} GPUs")
    elif torch.cuda.is_available():
        print("Using 1 GPU")
    else:
        print("Using CPU")

    """GPU Strategy: (In case there is 2 GPUs)
    1"""

    combine_dataset, dataloader = torchvision_datasets.prepare_torch_default_image(torchvision_dataset_name, 
                                                              batch_size, 
                                                              image_size,
                                                              data_root_path)
    print("Dataset loaded")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(partial_train_image_save_path, exist_ok=True)

    down_block_config =  settings.DownsampleBlockConfig(
        down_channels_list = [64, 128, 256, 256],
        layers_in_each_block = [2, 2, 2],
        resnet_settings = settings.ResnetSettings(group_norm_channel_dim = 32),
        time_embedding_settings = None, # No time embedding
        attention_block_settings = None, # No attention block
    )
    mid_block_config = settings.MidBlockConfig(
        mid_channels_list = [256, 256],
        layers_in_each_block = [2],
        resnet_settings = settings.ResnetSettings(group_norm_channel_dim = 32),
        time_embedding_settings = None, # No time embedding
        attention_block_settings = settings.AttentionBlockSettings(group_norm_channel_dim=32,
                                                                   attn_num_heads=4,
                                                                   attn_embed_dim=256,
                                                                   attn_batch_first=True),
    )
    up_block_config = settings.UpsampleBlockConfig(
        up_channels_list = [256, 256, 128, 64],
        layers_in_each_block = [2, 2, 2],
        resnet_settings = settings.ResnetSettings(group_norm_channel_dim = 32),
        time_embedding_settings = None, # No time embedding
        attention_block_settings = None, # No attention block
    )
    latent_config = settings.LatentConfig(
        latent_codebook_size=8192, 
        latent_channel_dim=3,
    )
    enc_dec_final_groupnorm_channel_dim = 32
    VQVAE_model = VQVAE(down_block_config, 
                        mid_block_config, 
                        up_block_config, 
                        latent_config, 
                        enc_dec_final_groupnorm_channel_dim).to(format_needed_gpu(custom_gpu_mapping["VQVAE"], cuda_device_count))
    generator_l2_criterion = torch.nn.MSELoss()
    generator_optimizer = torch.optim.Adam(VQVAE_model.parameters(), lr = vqvae_lr, 
                                              betas=(0.5, 0.999))
    
    print("VQVAE model loaded")

    lpips_model = LPIPS_5Layer().eval().to(format_needed_gpu(custom_gpu_mapping["LPIPS"], cuda_device_count))


    discriminator_config = settings.DiscriminatorConfig(
        conv_channels_list = [64, 128, 256],
        kernel_size_list = [4, 4, 4, 4],
        stride_list = [2, 2, 2, 1],
        padding_list = [1, 1, 1, 1]
    )
    discriminator_model = PatchGANDiscriminator(discriminator_config, im_channels = 3).to(format_needed_gpu(custom_gpu_mapping["Discriminator"], cuda_device_count))
    discriminator_criterion = torch.nn.BCEWithLogitsLoss() #each patch to indicate real of fake
    discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr = discriminator_lr, 
                                               betas=(0.5, 0.999))
    discriminator_start_update = 4000
    iterations = 0
    print("Loading Loss components completed")

    for epoch_idx in range(num_epochs):
        last_image_batch = None
        last_generated_result = None
        for image_batch, _ in tqdm(dataloader):
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            
            iterations += 1

            image_batch = image_batch.to(format_needed_gpu(custom_gpu_mapping["VQVAE"], cuda_device_count))
            # Train the generator
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()
            
            generated_result, cookbook_losses, commitment_losses = VQVAE_model(image_batch)
            generator_loss = generator_l2_criterion(generated_result, image_batch) + \
                                codebook_loss_scale * cookbook_losses + \
                                commitment_loss_scale * commitment_losses
            
            #activate the discriminator after some iterations for generator update
            if iterations > discriminator_start_update:
                #used for adding to generator loss
                generated_result = generated_result.to(format_needed_gpu(custom_gpu_mapping["Discriminator"], cuda_device_count))
                discriminator_result_generated_img= discriminator_model(generated_result)
                #we want the discriminator to think that the generated image is real (1)
                discriminator_target_for_generator_loss = torch.ones_like(discriminator_result_generated_img).to(
                    format_needed_gpu(custom_gpu_mapping["Discriminator"], cuda_device_count))
                
                discriminator_loss = discriminator_criterion(discriminator_result_generated_img, discriminator_target_for_generator_loss)
                discriminator_loss = discriminator_loss.to(format_needed_gpu(custom_gpu_mapping["VQVAE"], cuda_device_count))
                generator_loss += discriminator_loss * discriminator_loss_scale_for_generator
            
            #lpips loss for generator perceptual loss
            generated_result = generated_result.to(format_needed_gpu(custom_gpu_mapping["LPIPS"], cuda_device_count))
            image_batch = image_batch.to(format_needed_gpu(custom_gpu_mapping["LPIPS"], cuda_device_count))
            lpips_loss = lpips_model(generated_result, image_batch)
            generator_loss += lpips_loss

            #generator learning
            generator_loss.backward(retain_graph = True)
            generator_optimizer.step()

            #discriminator learning
            if iterations > discriminator_start_update: 
                #since we have batch norm in, it is better that we feed the same image type to the discriminator
                discriminator_result_generated_img = discriminator_model(
                    generated_result.detach().to(format_needed_gpu(custom_gpu_mapping["Discriminator", cuda_device_count]))) #why detach? just to speed up training (we don't need to track the loss for generator)
                discriminator_result_real_img = discriminator_model(image_batch)
                discriminator_target_for_real_img = torch.ones_like(discriminator_result_real_img).to(
                    format_needed_gpu(custom_gpu_mapping["Discriminator", cuda_device_count]))
                discriminator_target_for_generated_img = torch.zeros_like(discriminator_result_generated_img).to(
                    format_needed_gpu(custom_gpu_mapping["Discriminator", cuda_device_count]))

                discriminator_loss = discriminator_loss_scale_for_discriminator* (discriminator_criterion(discriminator_result_real_img, discriminator_target_for_real_img) + \
                                    discriminator_criterion(discriminator_result_generated_img, discriminator_target_for_generated_img))
                
                discriminator_loss.backward()
                discriminator_optimizer.step()
            last_image_batch = image_batch
            last_generated_result = generated_result

        if show_partial_train_image_epoch and epoch_idx % show_partial_train_image_epoch == 0:
            random_idx = [0, 5, 7, 8, 9]
            selected_generated_img = last_generated_result[random_idx, :, :, :]
            selected_real_img = last_image_batch[random_idx, :, :, :]
            for idx, image_tensor in enumerate(selected_generated_img):
                torchvision.utils.save_image(image_tensor, os.path.join(partial_train_image_save_path, f"generated_image_epoch_{epoch_idx}_idx_{idx}.png"))
            for idx, image_tensor in enumerate(selected_real_img):
                torchvision.utils.save_image(image_tensor, os.path.join(partial_train_image_save_path, f"real_image_epoch_{epoch_idx}_idx_{idx}.png"))
        if save_partial_checkpoint_epoch and epoch_idx % save_partial_checkpoint_epoch == 0:
            cur_time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
            torch.save(VQVAE_model.state_dict(), os.path.join(model_save_path, f"vqvae_model_{cur_time}_epoch_{epoch_idx}.pth"))
            torch.save(discriminator_model.state_dict(), os.path.join(model_save_path, f"discriminator_model_{cur_time}_epoch_{epoch_idx}.pth"))
    
    torch.save(VQVAE_model.state_dict(), os.path.join(model_save_path, f"vqvae_model_finished_{cur_time}.pth"))
    torch.save(discriminator_model.state_dict(), os.path.join(model_save_path, f"discriminator_model_finished_{cur_time}.pth"))
        

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--batch_size", type=int, default=128)
    argParser.add_argument("--custom_gpu_mapping", type=json.loads, default={"LPIPS": 1, "VQVAE": 0, "Discriminator": 0})
    argVal = argParser.parse_args()
    print("Batch size: ", argVal.batch_size)
    print("Custom GPU Mapping: ", argVal.custom_gpu_mapping)
    train_vqvae(batch_size=argVal.batch_size,
                custom_gpu_mapping=argVal.custom_gpu_mapping)