import evaluate
import random
import wandb
import torch
import pandas as pd
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from dotenv import load_dotenv


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


load_dotenv() # huggingface and wandb credentials

wandb.login()

run = wandb.init(
    project='MBert-Factual',
    job_type="training",
    anonymous="allow"
)