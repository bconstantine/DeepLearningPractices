import torch
import torchvision

def prepare_torch_default_image(dataset_name, 
                                batch_size = 128, 
                                size = 256, 
                                root_path = "./data"):
    if dataset_name == "LFWPeople":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.ToTensor()
        ])
        train_dataset = torchvision.datasets.LFWPeople(root=root_path, download = True, 
                                                       split = 'train', transform = transform)
        test_dataset = torchvision.datasets.LFWPeople(root=root_path, download = True, 
                                                      split = 'test', transform=transform)
        combine_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        dataloader = torch.utils.data.DataLoader(combine_dataset, batch_size, shuffle=True, drop_last = True)
    else:
        raise ValueError("Dataset not supported")
    return combine_dataset, dataloader