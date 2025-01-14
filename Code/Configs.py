# This is simply config text file that I saved as .py to have easy syntax color in editor

# Every configs "Inherits" from default (hard coded)

#--------------------------------------------------------------------------
[defaults]
config_name = 'default_gemma2_2b_fp16'
output_path = 'output/'
train_data = '../Data/Preprocessed/train_preprocessed_FULL_original.csv'
swapab_data = '../Data/Preprocessed/train_preprocessed_swapped_FULL_custom.csv'
checkpoints_path = '../Checkpoints'
max_length = 1024
sample_size = 1.0
train_batch = 2
eval_batch = 8
test_batch=1
gradient_accumulation_steps = 2
n_epochs = 1
freeze_layers = 16 # there're 42 layers in total, we don't add adapters to the first 16 layers
max_layers = 42
base_model_lr = 2e-6
feature_fc_lr = 1e-4
classifier_lr = 1e-3
warmup_steps = 20
random_seed = 707 # should be the same for every config
optim_type = "adamw_8bit"
n_splits = 5
fold_idx = 0
validation_size=0.1
#--- model part ---
transformers_basemodel_path = 'unsloth/gemma-2-2b'
basemodel_path = '../BaseModel/gemma2_2b_unsloth' # loading from file and not from Transformers (faster)
quantize = '4bit'
fp16 = True
feature_dims = 63
num_classes = 1
hidden_dim=128
#--- lora part ----
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_bias = 'none'
spread_max_length = False # tokenizer will equally truncate and pad all 3 texts inputs (prompt, A, B)

#--------------------------------------------------------------------------
# Just to make sure everything run smoothly - ultra speed test config
[micro]
train_data = '../Data/Preprocessed/train_preprocessed_FULL_custom.csv'
#train_data = '../Data/Preprocessed/train_preprocessed_FULL_EN.csv'
#train_data = '../Data/Preprocessed/train_preprocessed_FULL_original.csv'
config_name = 'micro_gemma2_2b_fp16_4bit'
transformers_basemodel_path = 'unsloth/gemma-2-2b'
basemodel_path = '../BaseModel/gemma2_2b_unsloth_fp16_4bit'
max_layers = 26
quantize = '4bit'
fp16 = True
train_batch = 2
eval_batch = 2
n_epochs = 5
sample_size = 0.002
base_model_lr = 1e-5
feature_fc_lr = 5e-4
classifier_lr = 5e-4
max_length=256
spread_max_length = False
hidden_dim=10

#--------------------------------------------------------------------------
[runpod_1]
train_data = '../Data/Preprocessed/train_preprocessed_FULL_EN.csv'
config_name = 'runpod_1'
transformers_basemodel_path = 'unsloth/gemma-2-2b'
basemodel_path = '../BaseModel/gemma2_2b_unsloth_fp16'
quantize='4bit'
fp16=True
train_batch=4
eval_batch=4
n_epochs = 3
sample_size = 0.5
max_length = 2048
spread_max_length = False
hidden_dim=4096

#--------------------------------------------------------------------------
[runpod_2]
train_data = '../Data/Preprocessed/train_preprocessed_FULL_EN.csv'
config_name = 'runpod_2'
transformers_basemodel_path = 'unsloth/gemma-2-9b-bnb-4bit'
basemodel_path = '../BaseModel/gemma2_9b_unsloth_fp16_4bit'
quantize='4bit'
fp16=True
train_batch=4
eval_batch=4
n_epochs=3
sample_size=0.25
max_length=2048
spread_max_length=False
hidden_dim=4096

#--------------------------------------------------------------------------
[gemma2_9b_fp16_4bit_h1536]
config_name = 'gemma2_9b_fp16_4bit_h1536'
train_data = '../Data/Preprocessed/train_preprocessed_FULL_custom.csv'
transformers_basemodel_path = 'google/gemma-2-9b-it'
basemodel_path='../BaseModel/gemma2_9b_fp16_4bit'
quantize='4bit'
max_layers=42
train_batch=4
eval_batch=4
fp16=True
sample_size=0.01
n_epochs=3
max_length=2048
spread_max_length=False
hidden_dim=1536
base_model_lr=1e-5
feature_fc_lr=5e-4
classifier_lr=5e-4
validation_size=0.09


