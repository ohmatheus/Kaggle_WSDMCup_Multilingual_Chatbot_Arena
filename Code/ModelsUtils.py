import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from tqdm import tqdm

from peft import PeftModel, prepare_model_for_kbit_training

import Configurations as Configs

from datetime import datetime
import matplotlib.pyplot as plt
import pickle

from textblob import TextBlob #for sentiment analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#-------------------------------------------------------------------
label2name = {0: 'model_a', 1: 'model_b'}
name2label = {v:k for k, v in label2name.items()}
class_labels = list(label2name.keys())
class_names = list(label2name.values())

feature_list_bycol = [
    '_len',
    '_spaces',
    '_punct',
    '_question_mark',
    '_quot',
    '_formatting_chars',
    '_math_chars',
    '_curly_open',
    '_curly_close',
    '_round_open',
    '_round_close',
    '_special_chars',
    '_digits',
    '_lower',
    '_upper',
    '_chinese',
    '_round_balance',
    '_curly_balance',
    '_json',
    '_sentiment',
]

feature_added_manually = [
    'cosine_similarity_a',
    'cosine_similarity_b',
    'cosine_similarity_diff',
]

# 63 features in total

#-------------------------------------------------------------------
# Define a function to create options based on the prompt and choices
def reencode(row):
    row["encode_fail"] = False
    try:
        row["prompt"] = row.prompt.encode("utf-8").decode("utf-8")
    except:
        row["prompt"] = ""
        row["encode_fail"] = True

    try:
        row["response_a"] = row.response_a.encode("utf-8").decode("utf-8")
    except:
        row["response_a"] = ""
        row["encode_fail"] = True

    try:
        row["response_b"] = row.response_b.encode("utf-8").decode("utf-8")
    except:
        row["response_b"] = ""
        row["encode_fail"] = True
        
    return row


#-------------------------------------------------------------------
# Tokenization function
def tokenize(tokenizer, prompt, response_a, response_b, max_length=256, spread_max_length=False):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]

    if spread_max_length:
        prompt = tokenizer(prompt, max_length=max_length//3, return_tensors="pt", truncation=True,  padding="max_length").input_ids
        response_a = tokenizer(response_a, max_length=max_length//3, return_tensors="pt", truncation=True, padding="max_length").input_ids
        response_b = tokenizer(response_b, max_length=max_length//3, return_tensors="pt", truncation=True, padding="max_length").input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        attention_mask = [[1]* len(i) for i in input_ids]
    else:
        text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = tokenizer(text, max_length=max_length, return_tensors="pt", padding="max_length", truncation=True) # padding=False
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        #'token_len': [len(item) for item in input_ids],
    }



#-------------------------------------------------------------------
def save_history(history, config, checkpointName):
    savePath = config.checkpoints_path + '/' + config.config_name + '/' + checkpointName + '/'
    history['config_name'] = config.config_name
    with open(savePath+'history.pkl', 'wb') as fp:
        pickle.dump(history, fp)


#-------------------------------------------------------------------
def load_history(config, checkpointName):
    savePath = config.checkpoints_path + '/' + config.config_name + '/' + checkpointName
    with open(savePath+'history.pkl', 'rb') as fp:
        history = pickle.load(fp)
        return history


#-------------------------------------------------------------------
def plot_model_history(history, title):
    loss = history['train_accum_loss']
    val_loss = history['valid_loss']
    acc = history['train_accum_accuracy']
    val_acc = history['valid_accuracy']
    config_name = history['config_name']
    
    # show best epoch with a red line ?

    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    
    fig.suptitle(title + ' ' + config_name)
    
    ax[0].set(title='Loss')
    ax[0].plot(epochs, loss, label='Training')
    ax[0].plot(epochs, val_loss, label='Validation')
    ax[0].legend(loc="upper right")
    
    ax[1].set(title='accuracy')
    ax[1].plot(epochs, acc, label='Training')
    ax[1].plot(epochs, val_acc, label='Validation')
    ax[1].legend(loc="lower right")
    
    plt.close(fig)
    return fig


#-------------------------------------------------------------------
# Evaluation (used for trainning)
def evaluate_model(model, dataloader, device="cuda"):
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    loss_fn = nn.BCELoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), unit='row'):

            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                features=batch['features'].to(device)
            )

            labels = batch['label'].to(device)  # One-hot encoded labels

            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Compute predictions and accuracy
            normLogits = (logits>0.5).float()
            predictions = normLogits    #torch.argmax(logits, dim=1)  # Class with highest score
            true_labels = labels        #torch.argmax(labels, dim=1)  # Convert one-hot to class indices
            
            correct += (predictions == true_labels).sum().item()
            total_samples += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }



#-------------------------------------------------------------------
def train_model(model, dataloader, valid_dataloader, optimizer, config, scheduler=None,  device="cuda"):
    model = model.to(device)
    model.train()
    min_val_loss = float('inf')
    min_acc = 0
    history = {"train_accum_loss" : [], "train_accum_accuracy" : [], "valid_loss" : [], 
                "valid_accuracy" : []}
    history["best_epoch"]=0
    history["best_loss"]=0
    history["best_acc"]=0

    loss_fn = nn.BCELoss()

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M")
    checkpoint_prefix = date_time+'_'+str(config.max_length)+'_'

    for epoch in range(config.n_epochs):
        total_loss = 0
        model.train()
        correct = 0
        total_samples = 0
        
        for batch in tqdm(dataloader, total=len(dataloader), unit='row'):
            optimizer.zero_grad()
            
            logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                features=batch['features'].to(device)
            )
            
            # One-hot labels
            labels = batch['label'].to(device)
        
            loss = loss_fn(logits, labels)
        
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
            # Compute predictions and accuracy
            normLogits = (logits>0.5).float()
            predictions = normLogits    #torch.argmax(logits, dim=1)  # Class with highest score
            true_labels = labels        #torch.argmax(labels, dim=1)  # Convert one-hot to class indices
            
            correct += (predictions == true_labels).sum().item()
            total_samples += labels.size(0)
        
        # add date and hour + epochs in checkpoint_name
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples

        metrics = evaluate_model(model, valid_dataloader, device=device)
        
        print(f"Epoch {epoch + 1} Finished")
        print(f"Accumulated Train Loss: {avg_loss}")
        print(f"Accumulated Train Accuracy: {accuracy}")
        print(f"Valid Loss: {metrics['loss']}, Valid Accuracy : {metrics['accuracy']}")
        
        history['train_accum_loss'].append(avg_loss)
        history['train_accum_accuracy'].append(accuracy)
        history['valid_loss'].append(metrics['loss'])
        history['valid_accuracy'].append(metrics['accuracy'])
        
        chkptName = checkpoint_prefix + 'train'
        
        if min_val_loss > metrics['loss']:
            print(f"{metrics['loss']} val loss is better than previous {min_val_loss}, saving checkpoint_lossBest, epoch: ", epoch + 1)
            custom_save_model_chkpt(model, config, checkpointName=chkptName+"_lossBest", epoch=epoch+1)
            history["best_epoch"] = epoch + 1
            history["best_loss"] = metrics['loss']
            min_val_loss = metrics['loss']
            
        if min_acc < metrics['accuracy']:
            print(f"{metrics['accuracy']} val accuracy is better than previous {min_acc}, saving checkpoint_accBest, epoch: ", epoch + 1)
            custom_save_model_chkpt(model, config, checkpointName=chkptName+"_accBest", epoch=epoch+1)
            history["best_epoch"] = epoch + 1
            history["best_acc"] = metrics['accuracy']
            min_acc = metrics['accuracy']

        save_history(history, config, chkptName+"_lossBest")
        print(f"-----------------------------------------------------------------")
        #for param_group in optimizer.param_groups:
        #    print(f"Current learning rate: {param_group['lr']}")
    return history


#-------------------------------------------------------------------
def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# #-------------------------------------------------------------------
# def last_token_pool(hidden_states, attention_mask):
#     # Use the mean of non-masked tokens for pooling (faster on GPU)
#     mask = attention_mask.unsqueeze(-1).float()
#     pooled = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
#     return pooled

#-------------------------------------------------------------------
class PreferencePredictionModel(nn.Module):
    def __init__(self, gemma_model, feature_dim, hidden_dim=128, num_classes=2):
        super(PreferencePredictionModel, self).__init__()
        
        # Load transformer model
        self.gemma_model = gemma_model #AutoModel.from_pretrained(transformer_name)
        #transformer_hidden_size = gemma_model.model.model.config.hidden_size
        transformer_hidden_size = gemma_model.config.hidden_size
        
        # Fully connected layers for features
        self.feature_fc = nn.Linear(feature_dim, 128)
        # Xavier initialization for feature_fc weights
        init.xavier_uniform_(self.feature_fc.weight)
        if self.feature_fc.bias is not None:
            init.zeros_(self.feature_fc.bias)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(transformer_hidden_size + 128, hidden_dim), #embedding + features
            #nn.Linear(transformer_hidden_size, hidden_dim), #embedding + features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, features):
        #outputs = self.gemma_model.model.model(input_ids=input_ids, attention_mask=attention_mask) # dont take head from causalLM, just model
        outputs = self.gemma_model(input_ids=input_ids, attention_mask=attention_mask) #, output_hidden_states=True
        
        #embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        #.hidden_states[-1][0, -1, :]
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Feature processing
        feature_output = self.feature_fc(features)
        feature_output = F.normalize(feature_output, p=2, dim=1)
        
        # Concatenate and classify
        combined = torch.cat((embeddings, feature_output), dim=1)
        #combined = embeddings
        logits = self.classifier(combined)
        
        return logits

#-------------------------------------------------------------------
def custom_save_model_chkpt(model, config, checkpointName, epoch=0, optimizer=None):
    # peft model
    
    savePath = config.checkpoints_path + '/' + config.config_name + '/' + checkpointName

    model.gemma_model.save_pretrained(f'{savePath}/PEFT', save_adapters=True, save_embedding_layers=True)
    
    # features and classifier
    torch.save({
        'epoch': epoch,
        #'optimizer_state_dict': optimizer.state_dict(),
        'feature_fc_state_dict': model.feature_fc.state_dict(),
        'classifier_state_dict': model.classifier.state_dict(),
        }, f'{savePath}/PreferencePredictionModel.pt')

#-------------------------------------------------------------------
def custom_load_model_chkpt(config, checkpointName, loadFrom=None, device="cpu", is_trainable=True, optimizer=None):
    # load base
    quantization_config = None
    if config.quantize=='4bit': #should not be use as this is choosed when preparing model see 'Prepare_BaseModel' notebook
        quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
                )
    
    baseModel = AutoModel.from_pretrained(
            config.basemodel_path,
            torch_dtype=torch.float16,
            device_map=device,
            #quantization_config=quantization_config
            )

    #baseModel = prepare_model_for_kbit_training(baseModel)

    peftModelPath = ""
    if loadFrom:
        peftModelPath=f"{loadFrom.checkpoints_path}/{loadFrom.config_name}/"
    else:
        peftModelPath=f"{config.checkpoints_path}/{config.config_name}/"
    
    loadPath = peftModelPath + checkpointName
    
    # load peft from base
    loraModel_load = PeftModel.from_pretrained(
            baseModel,
            f'{loadPath}/PEFT',
            is_trainable=is_trainable)
    
    predictionModelLoaded = PreferencePredictionModel(
            loraModel_load,
            feature_dim=config.feature_dims,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim
            )
    
    checkpoint = torch.load(f'{loadPath}/PreferencePredictionModel.pt', weights_only=True)
    
    predictionModelLoaded.feature_fc.load_state_dict(checkpoint['feature_fc_state_dict'])
    predictionModelLoaded.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    return predictionModelLoaded


#-------------------------------------------------------------------
class ChatbotArenaDataset(Dataset):
    def __init__(self, dataframe, tokenizer, test=False, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = 2
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize the text
        #tokens = tokenize(self.tokenizer, [row['prompt']], [row['response_a']], [row['response_b']], max_length=self.max_length)
        tokens = row['tokens']
        
        # Extract features
        features = torch.tensor([], dtype=torch.float)

        for feat in feature_list_bycol:
            feat = torch.tensor([
                row[f'prompt{feat}'],
                row[f'response_a{feat}'],
                row[f'response_b{feat}']
                ], dtype=torch.float)
            features = torch.cat((features, feat))

        similarity_feat = torch.tensor([
                row[f'cosine_similarity_a'],
                row[f'cosine_similarity_b'],
                row[f'cosine_similarity_diff']
                ], dtype=torch.float)

        features = torch.cat((features, similarity_feat))

        if not self.test:
            # Label
            #label = torch.nn.functional.one_hot(torch.tensor(row['class_label']), num_classes=self.num_classes).float()
            #label = torch.nn.functional.one_hot(torch.tensor(row['class_label']), num_classes=1).float()
            label = torch.tensor([row['class_label']]).float()

            return {
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'features': features,
                'label': label
            }
        else:
            return {
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0),
                'features': features
            }



#-------------------------------------------------------------------
#----------------------- FEATURE ENGINEERING -----------------------
#-------------------------------------------------------------------
def compute_feats(df):
    for col in ["response_a","response_b","prompt"]:
        #df[f"{col}_len"]=df[f"{col}"].str.len()

        # Calculating Features:
        df[f"{col}_spaces"]=df[f"{col}"].str.count("\s")
        df[f"{col}_punct"]=df[f"{col}"].str.count(",|\.|!")
        df[f"{col}_question_mark"]=df[f"{col}"].str.count("\?")
        df[f"{col}_quot"]=df[f"{col}"].str.count("'|\"")
        df[f"{col}_formatting_chars"]=df[f"{col}"].str.count("\*|\_")
        df[f"{col}_math_chars"]=df[f"{col}"].str.count("\-|\+|\=")
        df[f"{col}_curly_open"]=df[f"{col}"].str.count("\{")
        df[f"{col}_curly_close"]=df[f"{col}"].str.count("}")
        df[f"{col}_round_open"]=df[f"{col}"].str.count("\(")
        df[f"{col}_round_close"]=df[f"{col}"].str.count("\)")
        df[f"{col}_special_chars"]=df[f"{col}"].str.count("\W")
        df[f"{col}_digits"]=df[f"{col}"].str.count("\d") #>0.astype('int32')
        df[f"{col}_lower"]=df[f"{col}"].str.count("[a-z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_upper"]=df[f"{col}"].str.count("[A-Z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_chinese"]=df[f"{col}"].str.count(r'[\u4e00-\u9fff]+').astype("float32")/df[f"{col}_len"]

        # Bracket Balance Features:
        df[f"{col}_round_balance"]=df[f"{col}_round_open"]-df[f"{col}_round_close"]
        df[f"{col}_curly_balance"]=df[f"{col}_curly_open"]-df[f"{col}_curly_close"]

        # JSON Feature:
        df[f"{col}_json"]=df[f"{col}"].str.lower().str.count("json")
        # 19*3 = 57 features == all columns - 6
    return df


#-------------------------------------------------------------------
def add_computed_feats(df):
    df = compute_feats(df)
    return df


#-------------------------------------------------------------------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


#-------------------------------------------------------------------
def add_sentiment_polarity(df):
    df['prompt_sentiment'] = df['prompt'].apply(get_sentiment)
    df['response_a_sentiment'] = df['response_a'].apply(get_sentiment)
    df['response_b_sentiment'] = df['response_b'].apply(get_sentiment)
    return df


#-------------------------------------------------------------------
def add_cosine_similarity(df):
    vectorizer = TfidfVectorizer(max_features=200)
    cosine_similarities_a=[]
    cosine_similarities_b=[]

    for i in range(len(df)):
        prompt = df['prompt'].iloc[i].strip()
        response_a = df['response_a'].iloc[i].strip()
        response_b = df['response_b'].iloc[i].strip()

        if not prompt or not response_a or not response_b:
            cosine_similarities_a.append(-1)
            cosine_similarities_b.append(-1)
            continue
        try:
            tfidf_matrix= vectorizer.fit_transform([prompt])
            tfidf_matrix_a = vectorizer.transform([response_a])
            tfidf_matrix_b = vectorizer.transform([response_b])
            cosine_similarities_a.append(cosine_similarity(tfidf_matrix, tfidf_matrix_a)[0][0])
            cosine_similarities_b.append(cosine_similarity(tfidf_matrix, tfidf_matrix_b)[0][0])
        except Exception as e:
            print(f"Error processing document {i}: {e}")
            cosine_similarities_a.append(-1)
            cosine_similarities_b.append(-1)

    df['cosine_similarity_a'] = cosine_similarities_a
    df['cosine_similarity_b'] = cosine_similarities_b
    df['cosine_similarity_diff']=df['cosine_similarity_a'] - df['cosine_similarity_b'] 
    return df


#-------------------------------------------------------------------
def extract_all_features(df):
    df = add_computed_feats(df)
    df = add_sentiment_polarity(df)
    df = add_cosine_similarity(df)
    return df