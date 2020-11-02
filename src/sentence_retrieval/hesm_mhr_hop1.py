import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from tqdm import tqdm
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

import config
import utils
from flint import torch_util
from log_util import save_tool
from sample_for_nli.tf_idf_sample_v1_0 import (convert_evidence2scoring_format,
                                               sample_v1_0,
                                               select_sent_for_eval)
from sentence_retrieval.sampler_for_nmodel import (get_additional_list,
                                                   get_first_evidence_list,
                                                   get_full_list,
                                                   get_full_list_from_list_d,
                                                   post_filter)
from utils import c_scorer, common


class HESMDataset(Dataset):
    def __init__(self, train_dict):
        super().__init__()
        self.texts = train_dict['text']
        self.labels = train_dict['labels']
        self.pid = train_dict['pid']

    def __len__(self):
        return len(self.texts)    

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'labels': self.labels[idx], 'pid': self.pid[idx], 'idx': idx}


class HESMUtil():
    def __init__(self, model, model_name='albert-base-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = model

    def text_to_instance(self,
                         premise, # evidence
                         hypothesis, # claim
                         pid = None,
                         label = None):

        labels_2_id = {'true': 0, 'false': 1, 'hidden': -2}

        premise_tokens = premise
        hypothesis_tokens = hypothesis
    
        label_id = labels_2_id[label]
        return hypothesis_tokens, premise_tokens, label_id, pid

    def read(self, data_list):
        sent_list = []
        label_list = []
        pid_list = []
        for example in data_list:
            label = example["selection_label"]

            premise = example["text"]
            hypothesis = example["query"]

            if premise == "":
                premise = "@@@EMPTY@@@"

            pid = str(example['selection_id'])

            claim, evid, label_id, pid = self.text_to_instance(premise, hypothesis, pid, label)
            sent_list.append((claim, evid))
            label_list.append(label_id)
            pid_list.append(pid)

        return sent_list, label_list, pid_list
    
    def step_max_len(self, sent_list, tot_max_len):
        max_len = 0
        for sent in sent_list:

            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(*sent, max_length=tot_max_len, add_special_tokens=True)

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))

        return max_len

    def step(self, batch, is_eval=False, tot_max_len = 100):
        sent_list = list(zip(batch['text'][0], batch['text'][1]))
        labels = batch['labels']

        # TODO: Move to HESMDataset to compute max_len only once  
        max_len = self.step_max_len(sent_list, tot_max_len)

        # TODO: Validate whether max_length needs to be dynamic based on the batch, else move to HESMDataset
        encoded_batch = self.tokenizer.batch_encode_plus(sent_list, max_length=max_len, pad_to_max_length=True)
        
        b_input_ids = torch.tensor(encoded_batch['input_ids'])
        b_input_mask = torch.tensor(encoded_batch['attention_mask'])
        b_token_type_ids = torch.tensor(encoded_batch['token_type_ids'])
        b_labels = labels

        b_input_ids = b_input_ids.cuda()
        b_input_mask = b_input_mask.cuda()
        b_token_type_ids = b_token_type_ids.cuda()
        b_labels = b_labels.cuda() if is_eval else None
        
        retval = self.model(b_input_ids, 
                            token_type_ids=b_token_type_ids, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
        
        return retval


def hidden_eval_hesm(hesm_model, model, dataloader, dev_data_list):
    with torch.no_grad():
        print("Evaluating ...")
        model.eval()
        total_size = 0

        y_pred_logits_list = []
        y_pred_prob_list = []
        y_id_list = []
        
        for batch in tqdm(dataloader):
            out = hesm_model.step(batch, is_eval=True)[0]
            prob = F.softmax(out, dim=1)

            y = batch['labels']
            y_id_list.extend(list(batch['pid']))

            y_pred_logits_list.extend(out[:, 0].tolist())
            y_pred_prob_list.extend(prob[:, 0].tolist())

            total_size += y.size(0)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_pred_logits_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['selection_id'])

            dev_data_list[i]['score'] = y_pred_logits_list[i]
            dev_data_list[i]['prob'] = y_pred_prob_list[i]

        print('total_size:', total_size)
    
    return dev_data_list


def score_converter(org_data_file, full_sent_list, top_k=5, prob_thr=0.5):
    """
        Combines sentences of same claim 
        :param org_data_file:
        :param full_sent_list: append full_sent_score list to evidence of original data file
        :param top_k: top k sentences to be retrieved
        :param prob_thr: probability threshold for retrieved sentences
        :return:
        """
    d_list = common.load_jsonl(org_data_file)
    augmented_dict = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']
        org_id = int(selection_id.split('<##>')[0])
        if org_id in augmented_dict:
            augmented_dict[org_id].append(sent_item)
        else:
            augmented_dict[org_id] = [sent_item]

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []
            sents = augmented_dict[int(item['id'])]

            for sent_i in sents:
                if sent_i['prob'] >= prob_thr:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score']))

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids
        item['predicted_sentids'] = [sid for sid, _ in item['scored_sentids']][:top_k]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        item['predicted_label'] = item['label']  # give ground truth label (for OFEVER calculation)

    # Removing all score and prob
    for sent_item in full_sent_list:
        if 'score' in sent_item.keys():
            del sent_item['score']
            del sent_item['prob']

    return d_list


def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    opt_dict = dict()
    model_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_dict)

    if optimizer is not None:
        opt_dict = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(opt_dict)
        
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']


def save_model(path, model, optimizer):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'epoch': epoch,
            #'loss': loss
            }, path)


def display(model, exclude=None):
    total_p_size = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())

            exclude_this = False
            for exclude_name in exclude:
                if exclude_name in str(name):
                    exclude_this = True

            if exclude_this:
                continue

            nn = 1
            for s in list(param.size()):
                nn = nn * s
            total_p_size += nn

    print('Total Size:', total_p_size)


def train_fever_hesm(model_name = "albert-base-v2"):
    seed = 12
    torch.manual_seed(seed)
    
    num_epoch = 4
    batch_size = 64
    
    # parameters for annealed sampling
    keep_neg_sample_prob = 0.06
    sample_prob_decay = 0.01
    min_keep_neg_sample_prob = 0.02

    experiment_name = "simple_nn_startkp_{}_de_{}".format(keep_neg_sample_prob, sample_prob_decay)
    resume_model = None

    dev_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc_exec/2019_10_07_10:14:16_r/nn_doc_retr_1_shared_task_dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc/2019_10_27_16:48:33_r/nn_doc_retr_1_train.jsonl"

    complete_upstream_dev_data = get_first_evidence_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True)
    print("Dev size:", len(complete_upstream_dev_data))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = 2,  
        output_attentions = False,
        output_hidden_states = False,
    )

    if torch.cuda.device_count() > 1:
        print("More than 1 gpu device found...")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    start_lr = 2e-5
    optimizer = AdamW(model.parameters(),
                  lr = start_lr,
                  eps = 1e-8
                )

    if resume_model is not None:
        print("Resume From:", resume_model)
        load_model(resume_model, model, optimizer)

    # Create Log File
    file_path_prefix, _ = save_tool.gen_file_prefix(f"{experiment_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # Save source code end.

    best_dev = -1
    iteration = 0
    
    criterion = nn.CrossEntropyLoss()
    hesm_model = HESMUtil(model, model_name=model_name)
    display(model)

    for i_epoch in range(num_epoch):
        print("Get first evidence for training...")
        complete_upstream_train_data = get_first_evidence_list(config.T_FEVER_TRAIN_JSONL, train_upstream_file, pred=False)
        
        print("Resampling...")
        print("Sample Prob.:", keep_neg_sample_prob)
        filtered_train_data = post_filter(complete_upstream_train_data, keep_prob=keep_neg_sample_prob,
                                          seed=12 + i_epoch)
        
        keep_neg_sample_prob -= sample_prob_decay
        if keep_neg_sample_prob <= min_keep_neg_sample_prob:
            keep_neg_sample_prob = min_keep_neg_sample_prob
        print("Sampled length:", len(filtered_train_data))
        
        sent_list, label_list, pid_list = hesm_model.read(filtered_train_data)

        train_dataset = HESMDataset({'text': sent_list, 'labels': label_list, 'pid': pid_list})    
        train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )
        
        if i_epoch == 0:
            accumulation_steps = 2 # accumulate gradients for increasing `batch_size` by a factor of `accumulation_steps`
            steps_per_epoch = len(train_dataloader)
            total_steps = steps_per_epoch * num_epoch
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)        
            save_epoch = 0.5 # evaluate and save every `save_epoch` epochs

        optimizer.zero_grad()
        for i, batch in tqdm(enumerate(train_dataloader)):
            model.train()
            out = hesm_model.step(batch)
            y = batch['labels'].cuda()
            
            loss = criterion(out, y)
            loss = loss / accumulation_steps
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                scheduler.step()
                optimizer.zero_grad()
            iteration += 1

            mod = steps_per_epoch * save_epoch
            if iteration % mod == 0:
                
                sent_list, label_list, pid_list = hesm_model.read(complete_upstream_dev_data)

                eval_dataset = HESMDataset({'text': sent_list, 'labels': label_list, 'pid': pid_list})    
                eval_dataloader = DataLoader(
                    eval_dataset,
                    sampler = SequentialSampler(eval_dataset),
                    batch_size = batch_size
                )
                
                complete_upstream_dev_data = hidden_eval_hesm(hesm_model, model, eval_dataloader, complete_upstream_dev_data)

                dev_results_list = score_converter(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data)
                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list, common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode, verbose=False)
                total = len(dev_results_list)
                hit = eval_mode['check_sent_id_correct_hits']
                tracking_score = hit / total

                print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
                print("Strict score:", strict_score)
                print(f"Eval Tracking score:", f"{tracking_score}")

                need_save = False
                if tracking_score > best_dev:
                    best_dev = tracking_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_'
                        f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})'
                    )

                    save_model(save_path, model, optimizer)


if __name__ == "__main__":
    train_fever_hesm()
