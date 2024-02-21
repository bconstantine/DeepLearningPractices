import os.path
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

import appConstants
from nvflare.apis.executor import Executor
from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager

class ImageCaptionTrainer(Executor):
    def __init__(self):
        super().__init__()
        self._lr = appConstants.EXECUTABLE_ARGS['learning_rate']
        self._epochs = appConstants.EXECUTABLE_ARGS['num_epochs']
        self._train_task_name = appConstants.TRAIN_TASK_NAME
        self._pre_train_task_name = appConstants.GET_WEIGHTS_TASK_NAME
        self._submit_model_task_name = appConstants.SUBMIT_MODEL_TASK_NAME
        self._exclude_vars = None

        # Load vocabulary wrapper
        with open(appConstants.EXECUTABLE_ARGS['vocab_path'], 'rb') as f:
            vocab = pickle.load(f)

        # Training setup
        self.training_setup = {
            'encoder':EncoderCNN(appConstants.EXECUTABLE_ARGS['embed_size']),
            'decoder':DecoderRNN(appConstants.EXECUTABLE_ARGS['embed_size'], appConstants.EXECUTABLE_ARGS['hidden_size'], len(vocab), appConstants.EXECUTABLE_ARGS['num_layers']),
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'loss': nn.CrossEntropyLoss(),
            'transform': transforms.Compose([ 
                transforms.RandomCrop(appConstants.EXECUTABLE_ARGS['crop_size']),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), 
                                    (0.229, 0.224, 0.225))])
        }
        self.training_setup['encoder'].to(self.training_setup['device'])
        self.training_setup['decoder'].to(self.training_setup['device'])
        params = list(self.training_setup['decoder'].parameters()) + list(self.training_setup['encoder'].parameters()) + list(self.training_setup['encoder'].bn.parameters())
        self.training_setup['optimizer'] = torch.optim.Adam(params, lr=appConstants.EXECUTABLE_ARGS['learning_rate'])
        self.training_setup['train_loader'] = get_loader(appConstants.EXECUTABLE_ARGS['train_image_dir'], appConstants.EXECUTABLE_ARGS['caption_path'], vocab, 
                             self.training_setup['transform'], appConstants.EXECUTABLE_ARGS['batch_size'],
                             shuffle=True, num_workers=appConstants.EXECUTABLE_ARGS['num_workers']) 
    def _local_train(self, fl_ctx, weights, abort_signal):
        # Load weights into both models
        encoder_weights = {k: v for k, v in weights.items() if k.startswith('encoder')}
        decoder_weights = {k: v for k, v in weights.items() if k.startswith('decoder')}
        self.training_setup['encoder'].load_state_dict(encoder_weights)
        self.training_setup['decoder'].load_state_dict(decoder_weights)

        # Training loop modification to use both encoder and decoder
        # Basic training
        self.training_setup['encoder'].train()
        self.training_setup['decoder'].train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            for i, batch in enumerate(self.training_setup['train_loader']):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i % 3000 == 0:
                    self.log_info(
                        fl_ctx, f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {running_loss/3000}"
                    )
                    running_loss = 0.0

    def _save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def _load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))