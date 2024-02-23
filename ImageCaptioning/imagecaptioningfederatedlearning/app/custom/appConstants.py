TRAIN_TASK_NAME="train"
SUBMIT_MODEL_TASK_NAME="submit_model"
GET_WEIGHTS_TASK_NAME="get_weights"
TASK_VALIDATION="validate"
MODEL_OWNER="_model_owner_"
EXECUTABLE_ARGS = {

    'PTServerName':"server",
    'PTFileModelName': "FL_global_model.pt",
    'PTLocalModelName':"local_model.pt",
    'PTModelsDir':"models",#path for saving trained models (all models)

    #Filing and Input
    'crop_size': 224, #size for randomly cropping images
    'vocab_path':'../../../data/vocab.pkl', #path for vocabulary wrapper
    'train_image_dir':'../../../data/resizedtrain2014', #directory for resized training images
    'caption_path':'../../../data/annotations/captions_train2014.json', #path for train annotation json file
    'validate_image_dir':'../../../data/resizedval2014', #directory for resized validation images
    'validate_caption_path':'../../../data/annotations/captions_val2014.json', #path for validation annotation json file
    'log_step':10,#step size for prining log info
    'validating_step':10, #step size for prining validating info
    'save_step':1000, #step size for saving trained models
    
    # Model parameters
    'embed_size':256, #dimension of word embedding vectors
    'hidden_size':512, #dimension of lstm hidden states
    'num_layers':1, #number of layers in lstm
    
    #training par
    'num_epochs':5,
    'batch_size':128,
    'num_workers':2,
    'learning_rate':0.001,
}