import os
from typing import List, Union

import torch.cuda
from model import EncoderCNN, DecoderRNN, ImageCaptioningModel
import appConstants
import pickle

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class PTModelLocator(ModelLocator):
    def __init__(self):
        super().__init__()
        with open(appConstants.EXECUTABLE_ARGS['vocab_path'], 'rb') as f:
            vocab = pickle.load(f)
        self.model = ImageCaptioningModel(appConstants.EXECUTABLE_ARGS['embed_size'], appConstants.EXECUTABLE_ARGS['hidden_size'], len(vocab), appConstants.EXECUTABLE_ARGS['num_layers'])

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        return [appConstants.EXECUTABLE_ARGS['PTServerName']]

    def locate_model(self, model_name, fl_ctx: FLContext) -> Union[DXO, None]:
        if model_name == appConstants.EXECUTABLE_ARGS['PTServerName']:
            try:
                server_run_dir = fl_ctx.get_engine().get_workspace().get_app_dir(fl_ctx.get_job_id())
                model_path = os.path.join(server_run_dir, appConstants.EXECUTABLE_ARGS['PTFileModelName'])
                if not os.path.exists(model_path):
                    return None

                # Load the torch model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                data = torch.load(model_path, map_location=device)

                # Set up the persistence manager.
                if self.model:
                    default_train_conf = {"train": {"model": type(self.model).__name__}}
                else:
                    default_train_conf = None

                # Use persistence manager to get learnable
                persistence_manager = PTModelPersistenceFormatManager(data, default_train_conf=default_train_conf)
                ml = persistence_manager.to_model_learnable(exclude_vars=None)

                # Create dxo and return
                return model_learnable_to_dxo(ml)
            except Exception as e:
                self.log_error(fl_ctx, f"Error in retrieving {model_name}: {e}.", fire_event=False)
                return None
        else:
            self.log_error(fl_ctx, f"PTModelLocator doesn't recognize name: {model_name}", fire_event=False)
            return None