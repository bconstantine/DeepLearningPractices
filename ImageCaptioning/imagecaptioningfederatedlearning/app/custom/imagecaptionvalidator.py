import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, ImageCaptioningModel
from PIL import Image
import appConstants
import torch.nn as nn
from data_loader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence

from nvflare.apis.flcontext import FLContext, Signal, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply #nvflare communicate weights in Shareable format
from nvflare.apis.dxo import DXO, DataKind, from_shareable #dxo communicate the shareable
from nvflare.apis.executor import Executor #task to be executed
import appConstants

class ImageCaptionValidator(Executor):
    def __init__(self):
        super().__init__()

        self._validate_task_name = appConstants.TASK_VALIDATION

        # Setup the model# Load vocabulary wrapper
        with open(appConstants.EXECUTABLE_ARGS['vocab_path'], 'rb') as f:
            vocab = pickle.load(f)

        self.validating_setup = {
            'model': ImageCaptioningModel(appConstants.EXECUTABLE_ARGS['embed_size'], appConstants.EXECUTABLE_ARGS['hidden_size'], len(vocab), appConstants.EXECUTABLE_ARGS['num_layers']),
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'loss': nn.CrossEntropyLoss(),
            'transform': transforms.Compose([ 
                transforms.RandomCrop(appConstants.EXECUTABLE_ARGS['crop_size']),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), 
                                    (0.229, 0.224, 0.225))])
        }
        self.validating_setup['model'].to(self.validating_setup['device'])

        self.validating_setup['val_loader'] = get_loader(appConstants.EXECUTABLE_ARGS['validate_image_dir'], appConstants.EXECUTABLE_ARGS['validate_caption_path'], vocab, 
                             self.training_setup['transform'], appConstants.EXECUTABLE_ARGS['batch_size'],
                             shuffle=True, num_workers=appConstants.EXECUTABLE_ARGS['num_workers']) 

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """NVFlare executor entry point.
        
        Args are taken directly as recommended from NVFlare documentation.
        
        DXO (Data Exchange Object): A standardized container for data exchange in NVFlare, 
                                    facilitating consistent data handling and manipulation 
                                    across federated learning systems.
        Shareable: A communication medium in NVFlare for exchanging data 
                    and commands between the server and clients, ensuring data integrity 
                    and standardization.
        FLContext: Provides contextual information within NVFlare, 
                    supporting the coordination and execution of federated learning tasks 
                    by passing essential state and configuration details.
        """
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(appConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.validating_setup['device']) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self._validate(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _validate(self, weights, abort_signal):
        """Validate pipeline for the model
        
        Args are taken directly as recommended from NVFlare documentation."""
        self.validating_setup['model'].load_state_dict(weights)
        self.validating_setup['model'].eval()
        total_loss = 0.0
        total_step = len(self.validating_setup['val_loader'])
        with torch.no_grad():
            running_loss = 0.0
            running_perplexity = 0.0
            for i, (images, captions, lengths) in enumerate(self.validating_setup['val_loader']):
                if abort_signal.triggered:
                    return 0
                
                # Set mini-batch dataset
                images = images.to(self.validating_setup['device'])
                captions = captions.to(self.validating_setup['device'])
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0] #captions padded with same sequence length
                
                # Forward, backward and optimize
                outputs = self.validating_setup['model'](images, captions, lengths)
                loss = self.validating_setup['criterion'](outputs, targets)
                self.validating_setup['optimizer'].zero_grad()
                loss.backward()
                self.validating_setup['optimizer'].step()
                
                lossAmount = loss.item()
                perplexity = np.exp(lossAmount)
                total_loss += lossAmount
                running_loss += lossAmount / appConstants.EXECUTABLE_ARGS['log_step']
                running_perplexity += perplexity / appConstants.EXECUTABLE_ARGS['log_step']
                # Print log info
                if i % appConstants.EXECUTABLE_ARGS['validating_step'] == 0:
                    print('Validating step Step [{}/{}], Loss: {:.4f}, Running Loss: {:.4f}, Perplexity: {:5.4f}, Running Perplexity: {:5.4f}'
                        .format(i, total_step, lossAmount, running_loss, perplexity, running_perplexity)) 
                    running_loss = 0.0
                    running_perplexity = 0.0

        return total_loss/total_step
