# Install the hugging face transformer and pytorch lighting
#!pip install  transformers
#!pip install  pytorch-lightning

#importing the library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import torch
import re
from string import punctuation
from pathlib import Path
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap
from transformers import AdamW,T5ForConditionalGeneration,T5TokenizerFast
import tqdm.auto as tqdm
from pylab import rcParams
#set the seed value
pl.seed_everything(42)

#load the data
with Path("QA/BioASQ/BioASQ-train-factoid-4b.json").open() as json_file:
    data=json.load(json_file)
     
# Build the question and answering dataframe
def extract_question_answer(filepath:Path):
    with filepath.open() as json_file:
        data=json.load(json_file)
    
    questions=data["data"][0]["paragraphs"]
    
    data_rows= []
    for question in questions:
        context=question["context"]
        qa=question["qas"][0]
        answer=qa["answers"]
        question=qa["question"]
        
        ans=answer[0]
        ans_text=ans["text"]
        ans_start=ans["answer_start"]
        ans_end=ans["answer_start"] + len(ans_text)
        
        data_rows.append({
                        "question": question,
                        "context" : context,
                        "answer" : ans_text,
                        "ans_start" : ans_start,
                        "ans_end" : ans_end })
                
    return pd.DataFrame(data_rows)       

#grab all the train json file
file_paths=sorted(list(Path("QA/BioASQ/").glob("BioASQ-train-*")))

#call the extract_question_answer function
dfs=[]

for file_path in file_paths:
    dfs.append(extract_question_answer(file_path))

df=pd.concat(dfs) 

print("shape:",df.shape)
print("unique_question:",len(df.question.unique()))
print("unique_context:",len(df.context.unique()))
print("unique_answer:",len(df.answer.unique()))

#dropping the duplicate rows .
df= df.drop_duplicates(subset=["context"]).reset_insex(drop=True)


#initialize the tokenizer
model_name="t5-base"
tokenizer=T5TokenizerFast.from_pretrained(model_name) 

#train test split the data
train_df,test_df=train_test_split(df,test_size=0.05)
print(train_df.shape,test_df.shape)  

# create the model inputs (tokenized the data)
class BiaQADataset(Dataset):
    def __init__(self,
                data:pd.DataFrame,
                tokenizer: T5TokenizerFast,
                source_max_token_len: int=396,
                target_max_token_len: int=32):
        
        self.tokenizer=tokenizer
        self.data=data
        self.source_max_token_len=source_max_token_len
        self.target_max_token_len=target_max_token_len
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self,index: int):
        data_row=self.data.iloc[index]
        
        question=data_row["question"]
        context=data_row["context"]
        ans=data_row["answer_text"]
        
        source_encoding=tokenizer(
        question,
        context,
        max_length=self.source_max_token_len,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")
        
        target_encoding=tokenizer(
        ans,
        max_length=self.target_max_token_len, #create the encodings vector of fixed length
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")
        
        labels=target_encoding["input_ids"] # input_ids=unique id of each token
        labels[labels==0]=-100 # avoid the computations of padding loss.  
        
        return dict(
        question=question,
        context=context,
        ans=ans,
        source_input_ids=source_encoding["input_ids"].flatten(),
        source_attention_mask=source_encoding["attention_mask"].flatten(),
        labels=labels.flatten(),
        labels_attention_mask=target_encoding["attention_mask"].flatten())
    

# create the train + test tokenized dataset and train and test dataloader
class BiaASQDataModule(pl.LightningDataModule):
    def __init__(self,
                train_df:pd.DataFrame,
                test_df:pd.DataFrame,
                tokenizer:T5TokenizerFast,
                batch_size: int=8,
                source_max_token_len: int=512,
                target_max_token_len: int=128):
        super().__init__()
        
        self.train_df=train_df
        self.test_df=test_df
        self.batch_size=batch_size
        self.tokenizer=tokenizer
        self.source_max_token_len=source_max_token_len
        self.target_max_token_len=target_max_token_len   
      
    #create training and test dataset
    def setup(self):
        self.train_dataset=BiaQADataset(
        self.train_df,
        self.tokenizer,
        self.source_max_token_len,
        self.target_max_token_len)
        
        self.test_dataset=BiaQADataset(
        self.test_df,
        self.tokenizer,
        self.source_max_token_len,
        self.target_max_token_len)    
                   
    #create the dataloader
    def train_dataloader(self):
        return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size=1,
        shuffle=False
        )
    def test_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size=1,
        shuffle=False
        )   
            
    
NB_EPOCHS=6 # try more epochs to get good result
BATCH_SIZE=8    

#initialize the data module .
data_module=BiaASQDataModule(train_df,test_df,tokenizer,batch_size=BATCH_SIZE)      
data_module.setup()       
        
#----------------Building the model-------------------------------------

class BioASQModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model=T5ForConditionalGeneration.from_pretrained(model_name,return_dict=True)
        
    def forward(self,input_ids,attention_mask,decoder_attention_mask,labels=None):
        output=self.model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels, # already calculated .
        decoder_attention_mask=decoder_attention_mask    
        )
        
        return output.loss,output.logits       
           
    # To complete training loop(batch wise)
    def training_step(self,batch,batch_idx): 
        input_ids=batch["source_input_ids"]
        attention_mask=batch["source_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]
        
        # call the forward function
        loss,outputs=self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels)
               
        self.log("train_loss",loss,prog_bar=True,logger=True)
        return loss

    # To complete validation loop(batch wise)
    def validation_step(self,batch,batch_idx):
        
        input_ids=batch["source_input_ids"]
        attention_mask=batch["source_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]
         
        # call the forward function
        loss,outputs=self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels)
         
        self.log("val_loss",loss,prog_bar=True,logger=True)
        return loss
   
    # To complete test loop(batch wise)
    def test_step(self,batch,batch_idx):
        
        input_ids=batch["source_input_ids"]
        attention_mask=batch["source_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]
         
        # call the forward function
        loss,outputs=self(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=labels_attention_mask,
        labels=labels)
         
        self.log("test_loss",loss,prog_bar=True,logger=True)
        return loss

    # define optimizers and LR schedulers
    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=0.0001)

# Initialize the T5 model .   
model=BioASQModel()

# build the model chaeckpoints 
check_point_callback=ModelCheckpoint(
                        dirpath="checkpoints",
                        filename="best_checkpoint",
                        save_top_k=1,
                        verbose=True,
                        monitor="val_loss",
                        mode="min")

#Trainer handles the training loop details
trainer=pl.Trainer(checkpoint_callback=check_point_callback,
                  max_epochs=NB_EPOCHS)

# start the training process
trainer.fit(model,data_module)

# show the test loss(call the test_step())
trainer.test()

#load our trained model
trained_model=BioASQModel.load_from_checkpoint(
                       trainer.checkpoint_callback.best_model_path)

trained_model.freeeze()

#---------------------Generate and show the predictions----------------------

def generate_ans(question):
    source_encoding=tokenizer(
    question["question"],
    question["context"],
    max_length=396,
    padding="max_length",
    truncation="only_second",
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt")
    
    # Use of the trained model for generate the questions answer.
    generated_ids=trained_model.model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask=source_encoding["attention_mask"],
        max_length=80,
        num_beams=1,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True) 
    
    preds = [
        tokenizer.decode(gen_id,skip_special_tokens=True,clean_up_tokenization_spaces=True) for gen_id in generated_ids
    ]
    
    return "".join(preds)

#------------------- show the predictions---------------------------------

sample_row=test_df.iloc[0]
question=sample_row["question"]
model_output=generate_ans(sample_row)
print("question :",question)
print("Original_ans:",sample_row["ans_text"])  
print("Predicted_ans:",model_output) 
