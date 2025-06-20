import os
import torch
import torch.nn as nn
from collections import OrderedDict
import random
from peft import (  
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers import AutoModelForCausalLM,AutoTokenizer 
from sklearn import metrics as sk_metrics
import pickle
import json
import pandas as pd
from modules.regcn import REGCN


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)


class TKGLLMEVO(nn.Module):
    def __init__(self, conf, is_training=False):
        super(TKGLLMEVO, self).__init__()
        self.conf=conf
        self.device = device = conf["device"]
        self.llama_model_path = conf["base_model"]
        self.is_training = is_training
        self.is_align = conf["align"]
        print(f"is_align:{self.is_align}")
        self.hist_len = conf["hist_len"]
        self.k = conf["num_candidate"]


        self.dataset_name = conf["dataset"]
        self.dir = os.path.join(conf['data_path'], conf['dataset'])

        with open(os.path.join(self.dir, 'entity2id.json'), 'r') as f:
            self.ent2id = json.load(f)
        with open(os.path.join(self.dir, 'relation2id.json'), 'r') as f:
            self.rel2id = json.load(f)
        with open(os.path.join(self.dir, 'ts2id.json'), 'r') as f:
            self.ts2id = json.load(f)

        self.id2ent = {v:k for k,v in self.ent2id.items()}
        self.id2ts = {v:k for k,v in self.ts2id.items()}
        self.id2rel = {v:k for k,v in self.rel2id.items()}
        if self.is_training:
            self.data_textual=pd.read_csv(os.path.join(self.dir, 'train.csv'))
            self.data_textual=self.data_textual[["Subject","Relation","Object","Date"]]
            self.data_textual_candidates=pd.read_csv(os.path.join(self.dir, 'candidates','K_'+str(self.k),'train_'+str(self.k)+'_candidates.csv'))
        else:
            self.data_textual=pd.read_csv(os.path.join(self.dir, 'test.csv'))
            self.data_textual=self.data_textual[["Subject","Relation","Object","Date"]]
            self.data_textual_candidates=pd.read_csv(os.path.join(self.dir, 'candidates','K_'+str(self.k),'test_'+str(self.k)+'_candidates.csv'))               

        self.num_ent = conf["num_ent"]
        self.num_rel = conf["num_rel"]
        self.feat_dim = conf["h_dim"]
        with open(os.path.join(self.dir, 'graph_dict.pkl'), 'rb') as fp:
            self.graph_dict = pickle.load(fp)

        # TKG_model
        self.tkg_model = REGCN(conf)
        self.tkg_model.to(self.device)
        checkpoint = torch.load(conf['tkg_path'], map_location=conf["device"])
        self.tkg_model.load_state_dict(checkpoint['state_dict'])
        for _, param in self.tkg_model.named_parameters():
            param.requires_grad = False


        # LLM
        self.padding_side = "right"
        self.tokenizer = AutoTokenizer.from_pretrained(self.llama_model_path, padding_side = self.padding_side)
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_prefix_space = False         
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            self.llama_model_path,
            load_in_8bit=True,
            device_map=self.device,
        )

        self.llama_model.resize_token_embeddings(len(self.tokenizer))  
        
        # Projector
        conf["llama_size"] = self.llama_model.config.hidden_size
        if self.conf["soft_prompt"]:
            self.num_pt = 1
            self.prompt_token = nn.Embedding(self.num_pt, self.llama_model.config.hidden_size).to(device)
            init(self.prompt_token)
        else:
            self.prompt_token = None

        self.projector = nn.Sequential(OrderedDict([
                ('dense1', nn.Linear( self.feat_dim,  self.feat_dim * 2)),
                ('act1', nn.GELU()),
                ('output1', nn.Linear( self.feat_dim * 2, self.llama_model.config.hidden_size)),
            ])).to(device)
        
        self.projector_evo = nn.Sequential(OrderedDict([
                ('dense2', nn.Linear( self.feat_dim,  self.feat_dim * 2)),
                ('act2', nn.GELU()),
                ('output2', nn.Linear( self.feat_dim * 2, self.llama_model.config.hidden_size)),
            ])).to(device)
    
        if self.is_training:
            self.llama_model = prepare_model_for_kbit_training(self.llama_model)

            if conf['align']:
                self.lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                if os.path.exists(f"{conf['pretrain_model_path']}/adapter_model.safetensors"):
                    self.llama_model = PeftModel.from_pretrained(self.llama_model, conf["pretrain_model_path"], config=self.lora_config, is_trainable=conf["train_lora"])
                    print(">>>> pretrained llama lora loaded!")

                if self.conf["soft_prompt"] and os.path.exists(f"{conf['pretrain_model_path']}/prompt_token.bin"):
                    self.prompt_token.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/prompt_token.bin"))
                    print(">>>> prompt_token.bin loaded!")

                if os.path.exists(f"{conf['pretrain_model_path']}/projector_evo.bin"):
                    self.projector_evo.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/projector_evo.bin"))
                    print(">>>> projector_evo.bin loaded!")

                if os.path.exists(f"{conf['pretrain_model_path']}/projector.bin"):
                    self.projector.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/projector.bin"))
                    print(">>>> projector.bin loaded!")

            else:
                self.lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llama_model = get_peft_model(self.llama_model, self.lora_config)

        else:
            self.lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if os.path.exists(f"{conf['pretrain_model_path']}/adapter_model.safetensors"):
                self.llama_model = PeftModel.from_pretrained(self.llama_model, conf["pretrain_model_path"], config=self.lora_config, is_trainable=conf["train_lora"])
                print(">>>> pretrained llama lora loaded!")
            else:
                self.llama_model = get_peft_model(self.llama_model, self.lora_config)
                print(">>>> initial llama lora loaded!")

            if self.conf["soft_prompt"] and os.path.exists(f"{conf['pretrain_model_path']}/prompt_token.bin"):
                self.prompt_token.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/prompt_token.bin"))
                print(">>>> prompt_token.bin loaded!")

            if os.path.exists(f"{conf['pretrain_model_path']}/projector_evo.bin"):
                self.projector_evo.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/projector_evo.bin"))
                print(">>>> projector_evo.bin loaded!")

            if os.path.exists(f"{conf['pretrain_model_path']}/projector.bin"):
                self.projector.load_state_dict(torch.load(f"{conf['pretrain_model_path']}/projector.bin"))
                print(">>>> projector.bin loaded!")


        self.llama_model.model.config.use_cache = False
        self.llama_model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.llama_model.config.bos_token_id = 1
        self.llama_model.config.eos_token_id = 2
        self.pad_embeds = self._embed_tokens(torch.tensor([self.tokenizer.pad_token_id]).to(device))
        self.bos_embeds = self._embed_tokens(torch.tensor([self.tokenizer.bos_token_id]).to(device))
        self.eos_embeds = self._embed_tokens(torch.tensor([self.tokenizer.eos_token_id]).to(device))
        self.cutoff_len = 4096

    def _embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## PeftModelForCausalLM
            embeds = self.llama_model.base_model.base_model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds
    
    def generate_prompt(self,event,options):
        chat = [
                {"role": "system", "content": f"""You are an assistant for event forecasting and expected to correctly predict the missing object from the query in the form of "(Subject, Relation, Date)".You should directly answer the question by choosing the letter of the correct option."""},
                {"role": "user", "content": f"""[Query]: ({event[0]}, {event[1]}, {event[3]})\n[Subject Feature]: <f>\n[Relation Feature]: <f>\n[Option Feature]: <f>\n[Option]:[{', '.join(options)}]"""},
                {"role": "assistant", "content":"The answer is "}
                ]
        return self.tokenizer.apply_chat_template(chat, tokenize=False).replace("<s>", "").replace("</s>", "")


    def print_trainable_params(self):
        print('Trainable parameters:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print('\t'+name)
    
    def _gen_label_ids(self, gt_label):
        label_ids = self._tokenize(gt_label, add_eos_token=False).input_ids

        return torch.LongTensor(label_ids).to(self.device)

    def _tokenize(self, prompt, cutoff_len=None, add_eos_token=True):
        if cutoff_len is None:
            cutoff_len = self.cutoff_len
        result = self.tokenizer(
            prompt,
            padding=False,
            return_tensors=None,
        )
        
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
        return result

    def forward(self, event_id, history, candidates_id):
        query_objects = event_id[:,2].tolist()
        labels = [self.id2ent[object] for object in query_objects]
        index = event_id[:,4].tolist()
        events = self.data_textual.iloc[index,:4].values.tolist()
        candidates = self.data_textual_candidates.iloc[index,:]['Candidates'].apply(eval).values.tolist()

        feature_list = []
        candidates_embeddings = []
        label_embeddings = []

        for k,tim_list in enumerate(history.tolist()):
            g_list = [self.graph_dict[tim].to(self.device) for tim in tim_list]
            self.tkg_model.eval()
            query = torch.LongTensor(event_id[k,:3]).to(self.device).unsqueeze(0)
            _, history_embedding, _, rel_embed, _  = self.tkg_model.predict(g_list,query)

            candidate_embedding_map = []
            feature = []

            for i in list(candidates_id[k,1:]):
                candidate_emb = history_embedding[i].unsqueeze(0)
                candidate_embedding_map.append(self.projector_evo(candidate_emb))

            candidates_embeddings.append(candidate_embedding_map)
            
            label_emb = history_embedding[event_id[k,2]].unsqueeze(0)
            label_emb = self.projector_evo(label_emb)
            label_embeddings.append(label_emb)

            evo_embedding = history_embedding[event_id[k,0]].unsqueeze(0)
            evo_embedding = self.projector_evo(evo_embedding)
            feature.append(evo_embedding)

            rel_embedding = rel_embed[event_id[k,1]].reshape(1,1,-1)
            rel_embedding = self.projector(rel_embedding)

            feature.append(rel_embedding)
            feature_list.append(feature)

        inputs_embeds_list=[]
        label_embedding_list = []
        label_ids = []
        for idx,event in enumerate(events):
            prompt_embedding = [self.bos_embeds.unsqueeze(0)]
            label_embedding=[]
            options=[]

            index = random.randint(0, len(candidates[idx]))
            candidates[idx].insert(index, labels[idx])
            candidates_embeddings[idx].insert(index, label_embeddings[idx])

            for i,candidate in enumerate(candidates[idx]):
                options.append(chr(ord('A')+i) + '. '+ candidate)
                if i == index:
                    label = chr(ord('A')+i)

            for i, j in enumerate(self.generate_prompt(event,options).split("<f>")):
                prompt_id = torch.tensor((self._tokenize(j, add_eos_token=False).input_ids)).to(self.device).long()
                prompt_embedding.append(self._embed_tokens(prompt_id).unsqueeze(0))
                if i==0:
                    prompt_embedding.append(self.prompt_token.weight.view(1,self.num_pt,-1))
                    prompt_embedding.append(feature_list[idx][0])
                elif i==1:
                    prompt_embedding.append(self.prompt_token.weight.view(1,self.num_pt,-1))
                    prompt_embedding.append(feature_list[idx][1])
                elif i==2:
                    for candi_embedding in candidates_embeddings[idx]:
                        prompt_embedding.append(self.prompt_token.weight.view(1,self.num_pt,-1))
                        prompt_embedding.append(candi_embedding)

            inputs_embeds_list.append(torch.cat(prompt_embedding, dim=1))
            label_id = self._gen_label_ids(label)
            label_embedding.append(self._embed_tokens(label_id).unsqueeze(0))
            label_embedding.append(self.eos_embeds.unsqueeze(0))
            label_embedding_list.append(torch.cat(label_embedding, dim=1))
            label_ids.append(torch.cat([label_id,torch.LongTensor([self.tokenizer.eos_token_id]).to(self.device)], dim=0))


        emb_lens = [inputs_embeds.shape[1] for inputs_embeds in inputs_embeds_list]
        emb_max_len = max(emb_lens)
        emb_max_len_label = 5
        wrapped_embs = self.pad_embeds.expand(len(inputs_embeds_list), emb_max_len , -1).clone()
        wrapped_atts = torch.zeros(len(inputs_embeds_list), emb_max_len, dtype=torch.long).to(self.device)
        wrapped_embs_label = self.pad_embeds.expand(len(label_embedding_list), emb_max_len_label , -1).clone()
        wrapped_atts_label = torch.zeros(len(label_embedding_list), emb_max_len_label, dtype=torch.long).to(self.device)

        for i, inputs_embeds in enumerate(inputs_embeds_list):
            wrapped_embs[i, - emb_lens[i]:] = inputs_embeds
            wrapped_atts[i, - emb_lens[i]:] = 1
        for i, label_embeds in enumerate(label_embedding_list):
            wrapped_embs_label[i, :label_embeds.shape[1]] = label_embeds
            wrapped_atts_label[i, :label_embeds.shape[1]] = 1
        wrapped_embs = torch.cat([wrapped_embs,wrapped_embs_label], dim=1)
        wrapped_atts = torch.cat([wrapped_atts,wrapped_atts_label], dim=1)

        label_pad_ids = torch.full([len(wrapped_embs), emb_max_len+emb_max_len_label], -100, dtype=torch.long).to(self.device)

        for i, label_id in enumerate(label_ids):
            label_pad_ids[i, emb_max_len:emb_max_len+label_id.shape[0]] = label_id
        
        label_pad_ids = label_pad_ids.to(self.device)
        

        outputs = self.llama_model(
                    inputs_embeds=wrapped_embs, # [4, 324, 4096]
                    attention_mask=wrapped_atts, # [4, 324]
                    return_dict=True,
                    labels=label_pad_ids, # [4, 324]
                )
        
        return {
            "loss": outputs.loss, #+ option_level_loss,
            "logits": outputs.logits,
        }        


    def generate(self, inputs_embeds_list, config):
        emb_lens = [inputs_embeds.shape[1] for inputs_embeds in inputs_embeds_list]
        emb_max_len = max(emb_lens)
        wrapped_embs = self.pad_embeds.expand(len(inputs_embeds_list), emb_max_len , -1).clone()
        wrapped_atts = torch.zeros(len(inputs_embeds_list), emb_max_len, dtype=torch.long).to(self.device)
        for i, inputs_embeds in enumerate(inputs_embeds_list):
            wrapped_embs[i, - emb_lens[i]:] = inputs_embeds
            wrapped_atts[i, - emb_lens[i]:] = 1

        self.token_lens = emb_lens

        return self.llama_model.generate(
            inputs_embeds=wrapped_embs, # [2, 195, 4096]
            attention_mask=wrapped_atts, # [2, 195]
            generation_config=config,
        )
    
    def evaluate(self, event_id, history, candidates_id, config):
        index = event_id[:,4].tolist()
        events = self.data_textual.iloc[index,:4].values.tolist()
        query_objects = event_id[:,2].tolist()
        labels = [self.id2ent[object] for object in query_objects]
        candidates = self.data_textual_candidates.iloc[index,:]['Candidates'].apply(eval).values.tolist()
        
        feature_list = []
        evo_embedding_list = []
        candidates_embeddings = []
        label_embeddings = []
        for k,tim_list in enumerate(history.tolist()):
            g_list = [self.graph_dict[tim].to(self.device) for tim in tim_list]
            query = torch.LongTensor(event_id[k,:3]).to(self.device).unsqueeze(0)
            _, history_embedding, _, rel_embs, _  = self.tkg_model.predict(g_list,query)
            candidate_embedding_map = []
            feature = []

            for i in list(candidates_id[k,1:]):
                candidate_emb = history_embedding[i].unsqueeze(0)
                candidate_embedding_map.append(self.projector_evo(candidate_emb))
            candidates_embeddings.append(candidate_embedding_map)
            
            label_emb = history_embedding[event_id[k,2]].unsqueeze(0)
            label_emb = self.projector_evo(label_emb)
            label_embeddings.append(label_emb)

            evo_embedding = history_embedding[event_id[k,0]].unsqueeze(0)
            evo_embedding = self.projector_evo(evo_embedding)
            feature.append(evo_embedding)

            rel_embedding = rel_embs[event_id[k,1]].reshape(1,1,-1)
            rel_embedding = self.projector(rel_embedding)

            feature.append(rel_embedding)
            feature_list.append(feature)

        inputs_embeds_list=[]
        label_ids = []
        candidates_list=[]
        for idx,event in enumerate(events):
            prompt_embedding = [self.bos_embeds.unsqueeze(0)]
            options=[]

            index = random.randint(0, len(candidates[idx]))
            candidates[idx].insert(index, labels[idx])
            candidates_embeddings[idx].insert(index, label_embeddings[idx])
            candidates_list.append(candidates[idx])
            for i,candidate in enumerate(candidates[idx]):
                options.append(chr(ord('A')+i) + '. '+ candidate)
                if i == index:
                    label = chr(ord('A')+i)

            for i, j in enumerate(self.generate_prompt(event,options).split("<f>")):
                prompt_id = torch.tensor((self._tokenize(j, add_eos_token=False).input_ids)).to(self.device).long()
                prompt_embedding.append(self._embed_tokens(prompt_id).unsqueeze(0))
                if i==0:
                    prompt_embedding.append(self.prompt_token.weight.view(1,self.num_pt,-1))
                    prompt_embedding.append(feature_list[idx][0])
                elif i==1:
                    prompt_embedding.append(self.prompt_token.weight.view(1,self.num_pt,-1))
                    prompt_embedding.append(feature_list[idx][1])
                elif i==2:
                    for candi_embedding in candidates_embeddings[idx]:
                        prompt_embedding.append(self.prompt_token.weight.view(1,self.num_pt,-1))
                        prompt_embedding.append(candi_embedding)

            inputs_embeds_list.append(torch.cat(prompt_embedding, dim=1))
            label_ids.append(label)

        return self.generate(inputs_embeds_list, config),label_ids,candidates_list
