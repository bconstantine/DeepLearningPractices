# Federated Image Captioning with NVFlare
This project demonstrates how to apply federated learning to the task of image captioning using NVFlare and PyTorch. By leveraging a federated learning approach, the project aims to train an image captioning model in a privacy-preserving manner, with the model training distributed across multiple clients without sharing their local data.

##  Project Overview
The project uses ResNet-152 as an encoder to extract feature vectors from images, and an LSTM-based RNN as a decoder to generate captions. The federated learning framework is managed by NVFlare, allowing for decentralized model training and aggregation using the FedAvg algorithm.

## Federated Learning Configuration
This project utilizes client-side model initialization, ensuring that all participating clients start with the same model architecture but allow for customization in the initialization phase. The federated learning setup is configured to support pre-training, training, and cross-site validation workflows, enabling a comprehensive federated learning cycle.

## Getting Started
1. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```
2. Dataset Preparation
    Download the COCO Caption 2014 dataset for image captioning:
    ```bash
    ./download.sh #make sure data is downloaded in ./data folder
    ```
3. Reformat the data (image size)
    For vocab, we will built word2idx for all each word appear during training more than a certain threshold (default 4)
    ```bash
    #make sure train2014/, val2014/, annotations/ exist inside ./data folder from previous step
    python build_vocab.py   #build vocab for training image
    python resize_train_image.py #resize train image to 256x256 to ./data/resizedtrain2014 folder
    python resize_train_image.py --image_dir './data/val2014' --output_dir './data/resizedval2014' #do the same to val images
    ```

4. Build Vocabulary word2idx for training
    For vocab, we will built word2idx for all each word appear during training more than a certain threshold (default 4)
     ```bash
    #make sure train2014/, val2014/, annotations/ exist inside ./data folder from previous step
    cd ./imagecaptioningfederatedlearning/app/custom/
    python build_vocab.py   #build vocab for training image
    ```
5. Run the Federated Experiment
    Utilize the NVFlare simulator to run the federated learning experiment with predefined configurations:
    ```bash
    cd ../../../ #root folder in /ImageCaptioning/
    nvflare simulator -w /tmp/nvflare/ -n [number_of_clients] -t [number_of_training_rounds] ./imagecaptioningfederatedlearning/ 
    # for example, fill number_of_clients as 2 and number_of_training_rounds as 2
    ```

6. Access Logs and Results
    Check the simulation results and logs at the specified workspace directory:
    ```
    ls /tmp/nvflare/simulate_job/
    ```

## Federated Learning Workflow Configuration
There are two important configs for this project, each in imagecaptioningfederatedlearning/app/config for client side and server side configuration. 

The project defines three key workflows in the config_fed_server.json:
- Pre-training for model initialization using the InitializeGlobalWeights controller.
- Training via the ScatterAndGather controller.
- Cross-site validation using the CrossSiteModelEval controller.

These workflows ensure that the model is initialized, trained, and evaluated in a federated manner, with all steps configured to preserve data privacy and security.

## Model Initialization Approach
This project opts for client-side model initialization, where each client prepares its model instance before training begins. This strategy enhances security by avoiding server-side custom code execution and simplifies the initial setup process.

## Running the Experiment
To start the federated learning experiment, ensure that all configurations are set up correctly, including the server and clients' NVFlare configurations. Then, initiate the experiment using NVFlare commands, monitoring the process through the logs generated in the workspace.

## Other notes
In /imagecaptioningfederatedlearning/app/custom/, please refer to this folder for all the program components used for training pipeline. Refer to appConstants.py also to modify training parameters, file naming and locations, task names for NVFlare, etc. 

## Conclusion
By following the steps outlined, you can deploy a federated learning experiment that respects user privacy while leveraging the collective power of decentralized data for training sophisticated models.