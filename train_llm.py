# -*- coding: utf-8 -*-
import os
import numpy as np 
import torch
import transformers
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import re
from transformers import  Trainer # noqa: F402

from tqdm import tqdm
from modules.tglllm import TKGLLMEVO
from modules.utils_llm import Datasets

def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
eps = 1e-9

class Trainer2(Trainer):
    def _save(self, output_dir= None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.model.llama_model.save_pretrained(output_dir)
        torch.save(self.model.projector.state_dict(), os.path.join(output_dir, "projector.bin"))
        if self.model.conf["soft_prompt"]:
            torch.save(self.model.projector_evo.state_dict(), os.path.join(output_dir, "projector_evo.bin"))
            torch.save(self.model.prompt_token.state_dict(), os.path.join(output_dir, "prompt_token.bin"))


        # remove the main model to save the space
        for i in [
            os.path.join(output_dir, "pytorch_model.bin"),
        ]:
            if os.path.exists(i):
                os.remove(i)


class DataCollator:
    def __init__(self):
        pass

    def __call__(self, samples):
        return {
            "event_id": np.stack([sample[0]for sample in samples]),
            "history": np.stack([sample[1]for sample in samples]),
            "candidates_id": np.stack([sample[2]for sample in samples]),
        }

class DataCollator_align:
    def __init__(self):
        pass

    def __call__(self, samples):
        return {
            "event_id": np.stack([sample for sample in samples]),
        }



def train(conf, dataset):
    
    device = conf["device"]
    print('done')

    model = TKGLLMEVO(
        conf, is_training=True
    ).to(device)
    train_data = dataset.train_dataset

    os.environ["WANDB_DISABLED"] = "true"
    eval_step = 50

    GRADIENT_ACCUMULATION_STEPS = conf["batch_size"] // conf["batch_size_train"]
    # model.print_trainable_params()
    # history_length/
    output_dir = f"./checkpoints/LLM/Test_{conf['num_candidate']}_Hit_len_{conf['hist_len']}_{conf['dataset']}_{len(train_data)}"
    trainer = Trainer2(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            warmup_steps=20,
            num_train_epochs=conf['train_epoch'],
            learning_rate=3e-4,
            per_device_train_batch_size=conf["batch_size_train"],
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            run_name=f"TKGLLM_{conf['dataset']}",
            metric_for_best_model="eval_hitrate",
            report_to=None,
            save_safetensors=False,
        ),
        data_collator=DataCollator(),
    )

    trainer.train()
    trainer.save_model(output_dir + "/model_final")

def test(conf, dataset):
    
    model = TKGLLMEVO(
        conf, is_training=False
    ).to(device)
    # .bfloat16() to solve RuntimeError: probability tensor contains either inf, nan or element < 0

    # import pdb;pdb.set_trace()
    generation_config = transformers.GenerationConfig(
        num_beams=1,
        bos_token_id=model.tokenizer.bos_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
        max_new_tokens=5,
        do_sample=False,
    )
    model.eval()
    
    # log_path = conf["log_path"]
    with torch.no_grad():
        count_hit = 0
        # count_valid = 0
        count_all = 0
        output_results = []
        for event_id,history,candidates_id in tqdm(dataset.test_loader):
            outputs,label,candidates_list = model.evaluate(event_id, history,candidates_id, generation_config)
            index = event_id[:,4].tolist()
            output_seqs = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            count_all += len(output_seqs)

            for i in range(len(output_seqs)):
                pattern = re.compile(r'[A-Z]')
                match = pattern.search(output_seqs[i])
                
                if match:
                    if match.group()==label[i]:
                        count_hit+=1   

                output_results.append({
                    "response": output_seqs[i],
                    "groundtruth": label[i],
                    "candidates_list":candidates_list[i],
                    "id":index[i],
                })

            if count_all%(conf['batch_size_test']*10)==0:
                print(f"NOW hitrate: {count_hit/count_all:.4f}")

        hitrate = count_hit / count_all
        print('='*20)
        print(f'hitrate: {hitrate:.4f}')  
        
        return hitrate

import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", help="option: train or test", default="test", required=False)
    parser.add_argument("-d", "--dataset", default="IR",type=str, help="which dataset to use")
    parser.add_argument("-i", "--info", default="",type=str, help="")
    parser.add_argument("-p", "--pretrain_model_path", default="",type=str, help="")
    parser.add_argument("-a", "--align", action="store_true",help="stage 2")
    parser.add_argument("-k",'--num_candidate', default=9, type=int,help="3 or 5 or 9")
    parser.add_argument("--soft_prompt", action='store_false',help="")
    parser.add_argument("--hist_len", default=3, type=int, help="")
    parser.add_argument("--train_lora", default="", type=str,help="")
    parser.add_argument("--wandb", default="False", type=str, help="")
    parser.add_argument("-rs",'--train_sample', action="store_true",help="")

    paras = parser.parse_args().__dict__
    conf = yaml.safe_load(open("./config.yaml"))[paras["dataset"]]
    for p in paras:
        conf[p] = paras[p]


    print("load config file done!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    dataset = Datasets(conf)
    conf["num_ent"] = dataset.num_ent
    conf["num_rel"] = dataset.num_rel
    conf["dataset"] = paras["dataset"]
    conf["from_pretrain"] = True if conf["pretrain_model_path"] != "" else False

    setup_seeds()

    if paras["option"] == "train":
        if conf["train_lora"] == "":
            conf["train_lora"] = True
        else:
            conf["train_lora"] = eval(conf["train_lora"])
        print(conf)
        os.environ["WANDB_DISABLED"] = "true" if conf["wandb"] == "False" else "false"
        train(conf, dataset)
    elif paras["option"] == "test":

        conf["train_lora"] = False
        
        metrics = {
            "valid_ratio": [],
            "hitrate": []
        }
        
        metric = test(conf, dataset)
        with open("results.txt", "a") as f:
            f.write(f"hitrate: {metric:.4f}\n")