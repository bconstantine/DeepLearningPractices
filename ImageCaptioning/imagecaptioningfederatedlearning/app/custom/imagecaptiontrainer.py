#adjust import path for system
import sys
# sys.path.append('/home/mvc/BryanExperiment/DeepLearningPractices/ImageCaptioning/')

import os.path
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from model import EncoderCNN, DecoderRNN, ImageCaptioningModel
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
# from build_vocab import Vocabulary, build_vocab_main, load_vocab_from_another_file
import build_vocab

import appConstants
from nvflare.apis.shareable import Shareable, make_reply #nvflare communicate weights in Shareable format
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable #dxo communicate the shareable
from nvflare.apis.executor import Executor #task to be executed
from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
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
        #self.vocab = build_vocab_main(appConstants.EXECUTABLE_ARGS['caption_path'], appConstants.EXECUTABLE_ARGS['word_threshold'])
        self.vocab = build_vocab.load_vocab_from_another_file(appConstants.EXECUTABLE_ARGS['new_vocab_path'])
        # with open(appConstants.EXECUTABLE_ARGS['vocab_path'], 'rb') as f:
        #     vocab = pickle.load(f)

        # Training setup
        self.training_setup = {
            'model': ImageCaptioningModel(appConstants.EXECUTABLE_ARGS['embed_size'], appConstants.EXECUTABLE_ARGS['hidden_size'], len(self.vocab), appConstants.EXECUTABLE_ARGS['num_layers']),
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'loss': nn.CrossEntropyLoss(),
            'transform': transforms.Compose([ 
                transforms.RandomCrop(appConstants.EXECUTABLE_ARGS['crop_size']),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), 
                                    (0.229, 0.224, 0.225))]), 
            'criterion': nn.CrossEntropyLoss(),
        }
        self.training_setup['model'].to(self.training_setup['device'])
        self.training_setup['optimizer'] = torch.optim.Adam(self.training_setup['model'].parameters(), lr=appConstants.EXECUTABLE_ARGS['learning_rate'])
        self.training_setup['train_loader'] = get_loader(appConstants.EXECUTABLE_ARGS['train_image_dir'], appConstants.EXECUTABLE_ARGS['caption_path'], self.vocab, 
                             self.training_setup['transform'], appConstants.EXECUTABLE_ARGS['batch_size'],
                             shuffle=True, num_workers=appConstants.EXECUTABLE_ARGS['num_workers']) 
        
        # to save the PT model from NVFlare api
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._n_iterations = len(self.training_setup['train_loader'])
        self._default_train_conf = {"train": {"model": type(self.training_setup['model']).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.training_setup['model'].state_dict(), default_train_conf=self._default_train_conf
        )
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
        try:
            if task_name == self._pre_train_task_name:
                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._train_task_name:
                # Execute training task on local, and return their weights after training loop
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self._local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self._save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self._load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        
    def _get_model_weights(self) -> Shareable:
        # Get the new state dict and send as weights
        weights = {k: v.cpu().numpy() for k, v in self.training_setup['model'].state_dict().items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()
    
    def _local_train(self, fl_ctx, weights, abort_signal):
        """Local training loop for the model.

        Args are the same as execute method, and is expected to be called inside it
        """
        # Load weights into both models
        self.training_setup['model'].load_state_dict(state_dict=weights)

        # Training loop modification to use both encoder and decoder
        # Basic training
        self.training_setup['model'].train()
        total_step = self._n_iterations
        for epoch in range(appConstants.EXECUTABLE_ARGS['num_epochs']):
            running_loss = 0.0
            running_perplexity = 0.0
            for i, (images, captions, lengths) in enumerate(self.training_setup['train_loader']):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return
                
                # Set mini-batch dataset
                images = images.to(self.training_setup['device'])
                captions = captions.to(self.training_setup['device'])
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0] #captions padded with same sequence length
                
                # Forward, backward and optimize
                outputs = self.training_setup['model'](images, captions, lengths)
                loss = self.training_setup['criterion'](outputs, targets)
                self.training_setup['optimizer'].zero_grad()
                loss.backward()
                self.training_setup['optimizer'].step()
                
                lossAmount = loss.item()
                perplexity = np.exp(lossAmount)
                running_loss += lossAmount / appConstants.EXECUTABLE_ARGS['log_step']
                running_perplexity += perplexity / appConstants.EXECUTABLE_ARGS['log_step']
                # Print log info
                if i % appConstants.EXECUTABLE_ARGS['log_step'] == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Running Loss: {:.4f}, Perplexity: {:5.4f}, Running Perplexity: {:5.4f}'
                        .format(epoch, appConstants.EXECUTABLE_ARGS['num_epochs'], i, total_step, lossAmount, running_loss, perplexity, running_perplexity)) 
                    running_loss = 0.0
                    running_perplexity = 0.0

    def _save_local_model(self, fl_ctx: FLContext):
        """Saving local model

        Args are from execute method, and function is expected to be called inside it
        NVFLare will dictate when to save the model
        """
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir,appConstants.EXECUTABLE_ARGS['PTModelsDir']) #all models folder
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, appConstants.EXECUTABLE_ARGS['PTLocalModelName']) #local model path

        ml = make_model_learnable(self.training_setup["model"].state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def _load_local_model(self, fl_ctx: FLContext):
        """Loading local model

        Args are from execute method, and function is expected to be called inside it
        NVFLare will dictate when to save the model
        """
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, appConstants.EXECUTABLE_ARGS['PTModelsDir'])
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, appConstants.EXECUTABLE_ARGS['PTLocalModelName'])

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml