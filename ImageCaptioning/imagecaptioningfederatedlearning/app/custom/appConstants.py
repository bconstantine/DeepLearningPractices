TRAIN_TASK_NAME="train"
SUBMIT_MODEL_TASK_NAME="submit_model"
GET_WEIGHTS_TASK_NAME="get_weights"
TASK_VALIDATION="validate"
EXECUTABLE_ARGS = {
    #Filing and Input
    'model_path':'models/', #path for saving trained models
    'crop_size': 224, #size for randomly cropping images
    'vocab_path':'data/vocab.pkl', #path for vocabulary wrapper
    'train_image_dir':'data/resizedtrain2014', #directory for resized training images
    'caption_path':'data/annotations/captions_train2014.json', #path for train annotation json file
    'log_step':10,#step size for prining log info
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