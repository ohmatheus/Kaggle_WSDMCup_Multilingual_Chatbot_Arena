# This is simply config text file that I saved as .py to have easy syntax color in editor

#--------------------------------------------------------------------------
[defaults]
config_name = 'default_gemma2_2b_fp16'
output_path = 'output/'
train_data = '../Data/Preprocessed/train_preprocessed_FULL_original.csv'
checkpoints_path = '../Checkpoints'
max_length = 1024
sample_size = 1.0
train_batch = 2
eval_batch = 8
gradient_accumulation_steps = 2
n_epochs = 1
freeze_layers = 16 # there're 42 layers in total, we don't add adapters to the first 16 layers
start_lr = 0.0004
warmup_steps = 20
random_seed = 707 # should be the same for every config
optim_type = "adamw_8bit"
n_splits = 5
fold_idx = 0
#--- model part ---
transformers_basemodel_path = 'unsloth/gemma-2-2b'
basemodel_path = '../BaseModel/gemma2_2b_unsloth' # loading from file and not from Transformers (faster)
quantize = '4bit'
fp16 = True
feature_dims = 4
num_classes = 2
#--- lora part ----
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_bias = 'none'
spread_max_length = False

#--------------------------------------------------------------------------
[save_load_gemma2_2b_fp16]
config_name = 'save_load_gemma2_2b_fp16'
transformers_basemodel_path = 'unsloth/gemma-2-2b'
basemodel_path = '../BaseModel/gemma2_2b_unsloth_fp16'
quantize = None
fp16 = True

#--------------------------------------------------------------------------
# Just to make sure everything run smoothly - ultra speed test config
[micro]
config_name = 'micro_gemma2_2b_fp16'
transformers_basemodel_path = 'unsloth/gemma-2-2b'
basemodel_path = '../BaseModel/gemma2_2b_unsloth_fp16'
quantize = '4bit'
fp16 = True
train_batch = 2
eval_batch = 2
n_epochs = 2
sample_size = 0.005
max_length = 2048
spread_max_length = True

#--------------------------------------------------------------------------
# Testing robustness, memory, etc
[mini] 
sample_size = 0.01
max_length = 2048

#--------------------------------------------------------------------------
# Half the size for hyperparameters testing
[medium]
sample_size = 0.5

#--------------------------------------------------------------------------
# Full trainning
[full]
n_epoch = 8