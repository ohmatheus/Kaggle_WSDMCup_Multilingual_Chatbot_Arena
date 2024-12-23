import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

#-------------------------------------------------------------------
label2name = {0: 'model_a', 1: 'model_b'}
name2label = {v:k for k, v in label2name.items()}
class_labels = list(label2name.keys())
class_names = list(label2name.values())


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
def tokenize_data(tokenizer, prompt, response1, response2, max_length=256):
    tokens_resp1 = tokenizer(
        prompt,
        response1,  # Pair of responses
        #[response1, response2],  # Pair of responses
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    tokens_resp2 = tokenizer(
        prompt,
        response2,  # Pair of responses
        #[response1, response2],  # Pair of responses
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        'input_ids_resp1': tokens_resp1['input_ids'],
        'attention_mask_resp1': tokens_resp1['attention_mask'],
        'input_ids_resp2': tokens_resp2['input_ids'],
        'attention_mask_resp2': tokens_resp2['attention_mask']
    }



#-------------------------------------------------------------------
def predict(model, dataloader, device="cuda"):
    """
    Predict outcomes using a DataLoader for the test dataset.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the test dataset.
        device: Device to perform inference ('cpu' or 'cuda').

    Returns:
        A list of predicted class labels for the entire test dataset.
    """
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            input_ids_resp1 = batch['input_ids_resp1'].to(device)
            attention_mask_resp1 = batch['attention_mask_resp1'].to(device)
            input_ids_resp2 = batch['input_ids_resp2'].to(device)
            attention_mask_resp2 = batch['attention_mask_resp2'].to(device)
            features = batch['features'].to(device)

            # Forward pass through the model
            logits = model(
                input_ids_resp1=input_ids_resp1,
                attention_mask_resp1=attention_mask_resp1,
                input_ids_resp2=input_ids_resp2,
                attention_mask_resp2=attention_mask_resp2,
                features=features
            )

            # Convert logits to predicted class
            #batch_predictions = torch.argmax(logits, dim=1).cpu().tolist()
            #batch_predictions = logits.cpu().tolist()
            batch_probs = torch.softmax(logits, dim=1).cpu().tolist()
            predictions.extend(batch_probs)

    return predictions



#-------------------------------------------------------------------
# Evaluation (use for trainning)
def evaluate_model(model, dataloader, device="cuda"):
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    # Use BCEWithLogitsLoss for one-hot encoded labels
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.BCEWithLogitsLoss()
    #loss_fn = nn.BCELoss()

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            input_ids_resp1 = batch['input_ids_resp1'].to(device)
            attention_mask_resp1 = batch['attention_mask_resp1'].to(device)
            input_ids_resp2 = batch['input_ids_resp2'].to(device)
            attention_mask_resp2 = batch['attention_mask_resp2'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)  # One-hot encoded labels

            # Forward pass
            logits = model(
                input_ids_resp1=input_ids_resp1,
                attention_mask_resp1=attention_mask_resp1,
                input_ids_resp2=input_ids_resp2,
                attention_mask_resp2=attention_mask_resp2,
                features=features
            )

            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Compute predictions and accuracy
            predictions = torch.argmax(logits, dim=1)  # Class with highest score
            true_labels = torch.argmax(labels, dim=1)  # Convert one-hot to class indices
            
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
# Training loop
def train_model(model, dataloader, valid_dataloader, optimizer, scheduler = None, num_epochs=5, device="cuda"):
    model = model.to(device)
    model.train()
    min_val_loss = float('inf') #checkpoint

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        
        for batch in tqdm(dataloader, total=len(dataloader), unit='row'):
            optimizer.zero_grad()
            
            logits = model(
                input_ids_resp1=batch['input_ids_resp1'].to(device),
                attention_mask_resp1=batch['attention_mask_resp1'].to(device),
                input_ids_resp2=batch['input_ids_resp2'].to(device),
                attention_mask_resp2=batch['attention_mask_resp2'].to(device),
                features=batch['features'].to(device)
            )
            
            # One-hot labels
            labels = batch['label'].to(device)
        
            #loss = nn.BCEWithLogitsLoss()(logits, labels)
            loss = nn.CrossEntropyLoss()(logits, labels)
        
            # Use BCELoss for one-hot encoded labels
            #loss = nn.BCELoss()(logits, labels) #more stable, It combines a sigmoid activation and binary cross-entropy loss.
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
        
        metrics = evaluate_model(model, valid_dataloader, device=device)
        
        if min_val_loss > metrics['loss']:
            torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, f'PreferencePredictionModel.pt')
            print(f"{metrics['loss']} val loss is better than previous {min_val_loss}, saving checkpoint epoch: ", epoch + 1)
            min_val_loss = metrics['loss']
            

        print(f"Trainning Epoch {epoch + 1}, Accumulated Train Loss: {total_loss / len(dataloader)}")
        print(f"Eval : Valid Loss: {metrics['loss']}, Valid Accuracy : {metrics['accuracy']}")
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")



#-------------------------------------------------------------------
class PreferencePredictionModel(nn.Module):
    def __init__(self, transformer_name, feature_dim, num_classes=3):
        super(PreferencePredictionModel, self).__init__()
        
        # Load transformer model
        self.transformer = AutoModel.from_pretrained(transformer_name)
        transformer_hidden_size = self.transformer.config.hidden_size  # e.g., 768 for XLM-RoBERTa
        
        # Fully connected layers for features
        self.feature_fc = nn.Linear(feature_dim, 64)
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(2 * transformer_hidden_size + 64, 128),  # Combine response1, response2, and features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, input_ids_resp1, attention_mask_resp1, input_ids_resp2, attention_mask_resp2, features):
        # Process response1
        output_resp1 = self.transformer(input_ids=input_ids_resp1, attention_mask=attention_mask_resp1)
        cls_embedding_resp1 = output_resp1.last_hidden_state[:, 0, :]  # CLS token
        
        # Process response2
        output_resp2 = self.transformer(input_ids=input_ids_resp2, attention_mask=attention_mask_resp2)
        cls_embedding_resp2 = output_resp2.last_hidden_state[:, 0, :]  # CLS token
        
        # Feature processing
        feature_output = self.feature_fc(features)
        
        # Concatenate and classify
        combined = torch.cat((cls_embedding_resp1, cls_embedding_resp2, feature_output), dim=1)
        logits = self.classifier(combined)
        
        return logits



#-------------------------------------------------------------------
class ChatbotArenaDataset(Dataset):
    def __init__(self, dataframe, tokenizer, test=False, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = 3
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize the text
        tokens = tokenize_data(self.tokenizer, row['prompt'], row['response_a'], row['response_b'], self.max_length)
        
        # Extract engineered features
        features = torch.tensor([
            #row['resp1_length'],
            #row['resp2_length'],
            row['length_diff'],
            #row['resp1_lexical_div'],
            #row['resp2_lexical_div'],
            row['lexical_div_diff'],
            #row['resp1_similarity'],
            #row['resp2_similarity'],
            row['similarity_diff'],
            #row['resp1_keyword_overlap'],
            #row['resp2_keyword_overlap'],
            row['keyword_overlap_diff'],
        ], dtype=torch.float)

        if not self.test:
            # Label
            label = torch.nn.functional.one_hot(torch.tensor(row['class_label']), num_classes=self.num_classes).float()

            return {
                'input_ids_resp1': tokens['input_ids_resp1'].squeeze(0),
                'attention_mask_resp1': tokens['attention_mask_resp1'].squeeze(0),
                'input_ids_resp2': tokens['input_ids_resp2'].squeeze(0),
                'attention_mask_resp2': tokens['attention_mask_resp2'].squeeze(0),
                'features': features,
                'label': label
            }
        else:
            return {
                'input_ids_resp1': tokens['input_ids_resp1'].squeeze(0),
                'attention_mask_resp1': tokens['attention_mask_resp1'].squeeze(0),
                'input_ids_resp2': tokens['input_ids_resp2'].squeeze(0),
                'attention_mask_resp2': tokens['attention_mask_resp2'].squeeze(0),
                'features': features
            }



#-------------------------------------------------------------------
#----------------------- FEATURE ENGINEERING -----------------------
#-------------------------------------------------------------------

#-------------------------------------------------------------------
def add_length_features(df):
    df['resp1_length'] = df['response_a'].apply(len)
    df['resp2_length'] = df['response_b'].apply(len)
    df['length_diff'] = df['resp1_length'] - df['resp2_length']  # Difference in lengths
    return df

#-------------------------------------------------------------------
def lexical_diversity(text):
    tokens = text.split()  # Tokenize by whitespace
    return len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

def add_lexical_features(df):
    df['resp1_lexical_div'] = df['response_a'].apply(lexical_diversity)
    df['resp2_lexical_div'] = df['response_b'].apply(lexical_diversity)
    df['lexical_div_diff'] = df['resp1_lexical_div'] - df['resp2_lexical_div']
    return df

#-------------------------------------------------------------------
#from transformers import pipeline

## Load sentiment analysis pipeline (ensure it's multilingual)
#sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0)
#
#def get_sentiment(text):
#    result = sentiment_analyzer(text[:512])  # Truncate to 512 tokens for BERT-based models
#    return result[0]['label']
#
#def add_sentiment_features(df):
#    df['resp1_sentiment'] = df['response_a'].apply(get_sentiment)
#    df['resp2_sentiment'] = df['response_b'].apply(get_sentiment)
#    # Convert sentiments to numeric scale (e.g., positive=1, neutral=0, negative=-1)
#    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
#    df['resp1_sentiment_num'] = df['resp1_sentiment'].map(sentiment_map)
#    df['resp2_sentiment_num'] = df['resp2_sentiment'].map(sentiment_map)
#    df['sentiment_diff'] = df['resp1_sentiment_num'] - df['resp2_sentiment_num']
#    return df

#-------------------------------------------------------------------
def calculate_similarity(prompt, response, embedder):
    embeddings = embedder.encode([prompt, response])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def add_similarity_features(df, embedder):
    df['resp1_similarity'] = df.apply(lambda x: calculate_similarity(x['prompt'], x['response_a'], embedder), axis=1)
    df['resp2_similarity'] = df.apply(lambda x: calculate_similarity(x['prompt'], x['response_b'], embedder), axis=1)
    df['similarity_diff'] = df['resp1_similarity'] - df['resp2_similarity']
    return df

from keybert import KeyBERT

# Use KeyBERT for keyword extraction
#kw_model = KeyBERT()


#-------------------------------------------------------------------
def get_keyword_overlap(prompt, response, kw_model):
    prompt_keywords = set([kw[0] for kw in kw_model.extract_keywords(prompt)])
    response_keywords = set([kw[0] for kw in kw_model.extract_keywords(response)])
    overlap = len(prompt_keywords & response_keywords)
    return overlap / len(prompt_keywords) if len(prompt_keywords) > 0 else 0

def add_keyword_overlap_features(df, kw_model):
    df['resp1_keyword_overlap'] = df.apply(lambda x: get_keyword_overlap(x['prompt'], x['response_a'], kw_model), axis=1)
    df['resp2_keyword_overlap'] = df.apply(lambda x: get_keyword_overlap(x['prompt'], x['response_b'], kw_model), axis=1)
    df['keyword_overlap_diff'] = df['resp1_keyword_overlap'] - df['resp2_keyword_overlap']
    return df


#-------------------------------------------------------------------
def extract_all_features(df, embedder, kw_model):
    df = add_length_features(df)
    df = add_lexical_features(df)
    #df = add_sentiment_features(df)
    df = add_similarity_features(df, embedder)
    df = add_keyword_overlap_features(df, kw_model)
    #df = add_formality_features(df)
    #df = add_ner_features(df)
    return df