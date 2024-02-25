#adjust import path for system
import sys
#sys.path.append('/home/mvc/BryanExperiment/DeepLearningPractices/ImageCaptioning/')

import os
from typing import List, Union

import torch.cuda
from model import EncoderCNN, DecoderRNN, ImageCaptioningModel
import appConstants
import pickle
import build_vocab
# from build_vocab import Vocabulary, build_vocab_main, load_vocab_from_another_file

from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager

try:
    from build_vocab import Vocabulary
    print("Successfully imported MyClass from mymodule.")
except ImportError as e:
    print(f"Failed to import MyClass: {e}")

class PTModelLocator(ModelLocator):
    def __init__(self):
        super().__init__()
        print(f"Current pwd: {os.getcwd()}")
        #self.vocab = build_vocab_main(appConstants.EXECUTABLE_ARGS['caption_path'], appConstants.EXECUTABLE_ARGS['word_threshold'])
        self.vocab = build_vocab.load_vocab_from_another_file(appConstants.EXECUTABLE_ARGS['new_vocab_path'])
        self.model = ImageCaptioningModel(appConstants.EXECUTABLE_ARGS['embed_size'], appConstants.EXECUTABLE_ARGS['hidden_size'], len(self.vocab), appConstants.EXECUTABLE_ARGS['num_layers'])

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