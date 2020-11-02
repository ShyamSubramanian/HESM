from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel, AlbertModel, AlbertPreTrainedModel
from transformers.modeling_albert import *
from transformers.modeling_bert import BertEncoder
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)
from torch import nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
import copy

import config
from utils import c_scorer, common
from log_util import save_tool
from simi_sampler_nli_v0.simi_sampler import select_sent_with_prob_for_eval_esets, adv_simi_sample_with_prob_esets


# TODO: AutoModel for HESM


def norm_weight(din, dout, std=0.15):
    if dout is None:
        return nn.init.normal_(torch.empty(din, ),std=std)
    else:
        return nn.init.normal_(torch.empty(din, dout),std=std)


class AlbertTransformerEx(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])
        self.num_tune_layer = config.num_tune_layer

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        
        for i in range(self.config.num_hidden_layers):
            if i < self.num_tune_layer:
                with torch.no_grad():
                    # Number of layers in a hidden group
                    layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

                    # Index of the hidden group
                    group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

                    layer_group_output = self.albert_layer_groups[group_idx](
                        hidden_states,
                        attention_mask,
                        head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                    )
                    hidden_states = layer_group_output[0]

                    if self.output_attentions:
                        all_attentions = all_attentions + layer_group_output[-1]

                    if self.output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden_states,)

            else:
                # Number of layers in a hidden group
                layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

                # Index of the hidden group
                group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

                layer_group_output = self.albert_layer_groups[group_idx](
                    hidden_states,
                    attention_mask,
                    head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                )
                hidden_states = layer_group_output[0]

                if self.output_attentions:
                    all_attentions = all_attentions + layer_group_output[-1]

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)


        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class AlbertModelEx(AlbertPreTrainedModel):

    config_class = AlbertConfig
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformerEx(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs


class AlbertForHESM(AlbertPreTrainedModel):
    def __init__(self, config, num_tune_layer=-1):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.num_tune_layer = num_tune_layer # Layer number to start fine-tuning from

        self.albert = AlbertModelEx(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        self.word_attn_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.word_attn_vector = nn.Parameter(norm_weight(config.hidden_size, None))
        
        self.sent_attn_linear_nocontext = nn.Linear(config.hidden_size, config.hidden_size)
        self.sent_attn_vector_nocontext = nn.Parameter(norm_weight(config.hidden_size, None))
        
        self.sent_attn_linear_context = nn.Linear(config.hidden_size, config.hidden_size)
        self.sent_attn_vector_context = nn.Parameter(norm_weight(config.hidden_size, None))

        self.classifier_esm = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier_agg = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.config_custom = copy.deepcopy(config)
        self.config_custom.num_hidden_layers = 2
        self.bert_encoder_custom = BertEncoder(self.config_custom)
        
        self.beta1 = nn.Parameter(torch.Tensor([0.5]))
        self.beta2 = nn.Parameter(torch.Tensor([0.5]))

        self.init_weights()

    def forward(
        self,
        multi_input_ids=None,
        multi_attention_mask=None,
        multi_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        multi_labels=None,
        final_label=None,
        sent_attn_mask=None
    ):
        multi_summarized = [] # contains summarized vectors for each evidence set
        multi_pooled = [] # contains pooled albert output from each evidence set
        multi_logits = [] # contains classification logits from each evidence set

        # Evidence Set Modeling Block (ESM)
        total_loss = 0
        for input_ids, attention_mask, labels, token_type_ids in zip(multi_input_ids, multi_attention_mask, multi_labels, multi_token_type_ids):
            outputs = self.albert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            sequence_output = outputs[0]
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)

            # ESM Classifier & logits
            logits = self.classifier_esm(pooled_output)
            
            outputs = (logits,) + outputs[2:]

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
            
            total_loss += outputs[0]
            multi_pooled.append(pooled_output)
            multi_logits.append(logits)
            
            ## Word-level attention
            word_attn_linear_out = self.word_attn_linear(sequence_output)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) 
            attention_scores = torch.matmul(word_attn_linear_out, self.word_attn_vector)
            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            sequence_out_attn = attention_probs.unsqueeze(-1) * sequence_output 
            sentence_output = torch.sum(sequence_out_attn, 1)
            multi_summarized.append(sentence_output)
            
        # Hierarchical Aggregation Modeling (HAM)

        ## Non-contextual aggregation
        sentences = torch.stack(multi_pooled, 1)
        
        ### Evidence set-level aggregation
        sent_attn_linear_out = self.sent_attn_linear(sentences)
        ex_sent_attn_mask = (1.0 - sent_attn_mask) * -10000.0
        ex_sent_attn_mask = ex_sent_attn_mask.to(dtype=next(self.parameters()).dtype) 
        attention_scores = torch.matmul(sent_attn_linear_out, self.sent_attn_vector)
        attention_scores = attention_scores + ex_sent_attn_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        ### Aggregated logits
        nocontext_logits = torch.stack(multi_logits, 1)
        nocontext_logits = attention_probs.unsqueeze(-1) * nocontext_logits
        nocontext_logits = torch.sum(nocontext_logits, 1)

        ## Contextual aggregation
        multi_stacked_cls = torch.stack(multi_summarized, 1)
        multi_stacked_cls = multi_stacked_cls.to(next(self.parameters()).device)
        head_mask = [None] * self.config_custom.num_hidden_layers
        sent_attn_mask_ex = sent_attn_mask[:,None,None,:]
        extended_attention_mask = sent_attn_mask_ex.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        ### Transformer encoder
        final_output = self.bert_encoder_custom(multi_stacked_cls, extended_attention_mask, head_mask)[0]
        
        ### Evidence set-level aggregation
        final_output_linear_out = self.sent_attn_linear_final(final_output)
        attention_scores = torch.matmul(final_output_linear_out, self.sent_attn_vector_final)
        attention_scores = attention_scores + ex_sent_attn_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        final_output = torch.sum(attention_probs.unsqueeze(-1) * final_output, 1) 
        final_output = self.dropout(final_output)

        ### Aggregated evidence classifier
        context_logits = self.classifier_agg(final_output)

        ### HAM Logits
        ens_logits = self.beta1*nocontext_logits + self.beta2*context_logits
        
        # Training loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(ens_logits.view(-1, self.num_labels), final_label.view(-1))
        total_loss = total_loss.mean() + loss
        
        # Outputs
        outputs = (ens_logits, multi_logits)
        outputs = (total_loss,) + outputs

        return outputs


class BertForHESM(AlbertPreTrainedModel):
    def __init__(self, config, num_tune_layer):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.num_tune_layer = num_tune_layer # Layer number to start fine-tuning from

        self.bert = BertModel(config)
        self.freeze_bert_layers(self.bert, num_tune_layer)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)

        self.word_attn_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.word_attn_vector = nn.Parameter(norm_weight(config.hidden_size, None))
        
        self.sent_attn_linear_nocontext = nn.Linear(config.hidden_size, config.hidden_size)
        self.sent_attn_vector_nocontext = nn.Parameter(norm_weight(config.hidden_size, None))
        
        self.sent_attn_linear_context = nn.Linear(config.hidden_size, config.hidden_size)
        self.sent_attn_vector_context = nn.Parameter(norm_weight(config.hidden_size, None))

        self.classifier_esm = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier_agg = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.config_custom = copy.deepcopy(config)
        self.config_custom.num_hidden_layers = 2
        self.bert_encoder_custom = BertEncoder(self.config_custom)
        
        self.beta1 = nn.Parameter(torch.Tensor([0.5]))
        self.beta2 = nn.Parameter(torch.Tensor([0.5]))

        self.init_weights()

    @staticmethod
    def freeze_bert_layers(model, fine_tune_layer=6):
        for name, param in model.named_parameters():
            if 'module.bert.encoder.layer' in name:
                layer_num = int(name.split('.')[4])
                if layer_num < fine_tune_layer:
                    param.requires_grad = False
                else:
                    break
            elif 'bert.encoder.layer' in name:
                layer_num = int(name.split('.')[3])
                if layer_num < fine_tune_layer:
                    param.requires_grad = False
                else:
                    break
            else:
                param.requires_grad = True

    def forward(
        self,
        multi_input_ids=None,
        multi_attention_mask=None,
        multi_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        multi_labels=None,
        final_label=None,
        sent_attn_mask=None
    ):
        multi_summarized = [] # contains summarized vectors for each evidence set
        multi_pooled = [] # contains pooled albert output from each evidence set
        multi_logits = [] # contains classification logits from each evidence set

        # Evidence Set Modeling Block (ESM)
        total_loss = 0
        for input_ids, attention_mask, labels, token_type_ids in zip(multi_input_ids, multi_attention_mask, multi_labels, multi_token_type_ids):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

            sequence_output = outputs[0]
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)

            # ESM Classifier & logits
            logits = self.classifier_esm(pooled_output)
            
            outputs = (logits,) + outputs[2:]

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
            
            total_loss += outputs[0]
            multi_pooled.append(pooled_output)
            multi_logits.append(logits)
            
            ## Word-level attention
            word_attn_linear_out = self.word_attn_linear(sequence_output)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) 
            attention_scores = torch.matmul(word_attn_linear_out, self.word_attn_vector)
            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            sequence_out_attn = attention_probs.unsqueeze(-1) * sequence_output 
            sentence_output = torch.sum(sequence_out_attn, 1)
            multi_summarized.append(sentence_output)
            
        # Hierarchical Aggregation Modeling (HAM)

        ## Non-contextual aggregation
        sentences = torch.stack(multi_pooled, 1)
        
        ### Evidence set-level aggregation
        sent_attn_linear_out = self.sent_attn_linear(sentences)
        ex_sent_attn_mask = (1.0 - sent_attn_mask) * -10000.0
        ex_sent_attn_mask = ex_sent_attn_mask.to(dtype=next(self.parameters()).dtype) 
        attention_scores = torch.matmul(sent_attn_linear_out, self.sent_attn_vector)
        attention_scores = attention_scores + ex_sent_attn_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        ### Aggregated logits
        nocontext_logits = torch.stack(multi_logits, 1)
        nocontext_logits = attention_probs.unsqueeze(-1) * nocontext_logits
        nocontext_logits = torch.sum(nocontext_logits, 1)

        ## Contextual aggregation
        multi_stacked_cls = torch.stack(multi_summarized, 1)
        multi_stacked_cls = multi_stacked_cls.to(next(self.parameters()).device)
        head_mask = [None] * self.config_custom.num_hidden_layers
        sent_attn_mask_ex = sent_attn_mask[:,None,None,:]
        extended_attention_mask = sent_attn_mask_ex.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        ### Transformer encoder
        final_output = self.bert_encoder_custom(multi_stacked_cls, extended_attention_mask, head_mask)[0]
        
        ### Evidence set-level aggregation
        final_output_linear_out = self.sent_attn_linear_final(final_output)
        attention_scores = torch.matmul(final_output_linear_out, self.sent_attn_vector_final)
        attention_scores = attention_scores + ex_sent_attn_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        final_output = torch.sum(attention_probs.unsqueeze(-1) * final_output, 1) 
        final_output = self.dropout(final_output)

        ### Aggregated evidence classifier
        context_logits = self.classifier_agg(final_output)

        ### HAM Logits
        ens_logits = self.beta1*nocontext_logits + self.beta2*context_logits
        
        # Training loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(ens_logits.view(-1, self.num_labels), final_label.view(-1))
        total_loss = total_loss.mean() + loss
        
        # Outputs
        outputs = (ens_logits, multi_logits)
        outputs = (total_loss,) + outputs

        return outputs


class HESMDataset(Dataset):
    def __init__(self, train_dict):
        super().__init__()
        self.texts = train_dict['text']
        self.labels = train_dict['labels']
        self.multi_labels = train_dict['multi_labels']
        self.sent_attn_mask = train_dict['sent_attn_mask']
        self.pid = train_dict['pid']

    def __len__(self):
        return len(self.texts)    

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'labels': self.labels[idx], 'multi_labels': self.multi_labels[idx], 'sent_attn_mask': self.sent_attn_mask[idx], 'pid': self.pid[idx], 'idx': idx}


class HESMUtil():
    def __init__(self, model, model_name='albert-base-v2', max_evid_sets=3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = model
        self.max_evid_sets = max_evid_sets

    def text_to_instance(self,  # type: ignore
                         multi_premise,  # Important type information
                         hypothesis: str,
                         pid: str = None,
                         label: str = None):

        labels_2_id = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
        
        multi_premise_tokens = []
        multi_labels = []
        sent_attn_mask = []
        for idx, premise in enumerate(multi_premise):
            if idx >= self.max_evid_sets:
                break
            i = 0
            premise_tokens = ''
            for premise_sent, _ in premise[0]:
                if i != 0:
                    premise_tokens += " "

                premise_tokens += premise_sent
                i += 1
            multi_premise_tokens.append(premise_tokens)
            multi_labels.append(labels_2_id[premise[1]])
            sent_attn_mask.append(1)

        rem_evid = self.max_evid_sets - len(multi_premise_tokens)
        
        for i in range(rem_evid):
            multi_premise_tokens.append('@@@EMPTY@@@')
            multi_labels.append(2)
            sent_attn_mask.append(0)

        hypothesis_tokens = hypothesis
    
        label_id = labels_2_id[label]
        return hypothesis_tokens, multi_premise_tokens, label_id, multi_labels, sent_attn_mask, pid

    def read(self, data_list):
        sent_list = []
        label_list = []
        multi_label_list = []
        pid_list = []
        sent_attn_mask_list = []
        for example in data_list:
            label = example["label"]

            premise = example["evid"]
            hypothesis = example["claim"]

            if len(premise) == 0:
                premise = [[[("@@@EMPTY@@@", 0.0)], label]]

            pid = str(example['id'])

            claim, evid, label_id, labels, sent_attn_mask, pid = self.text_to_instance(premise, hypothesis, pid, label)
            sent_list.append((claim, *evid))
            label_list.append(label_id)
            multi_label_list.append(labels)
            sent_attn_mask_list.append(sent_attn_mask)
            pid_list.append(pid)

        return sent_list, label_list, multi_label_list, sent_attn_mask_list, pid_list
    
    def step_max_len(self, sent_list, tot_max_len):
        max_len = 0
        # For every sentence...
        for sent in sent_list:

            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(*sent, max_length=tot_max_len, add_special_tokens=True)

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))
        return max_len

    def step(self, batch, tot_max_len = 260):
        b_input_ids, b_input_mask, b_labels, b_token_type_ids = [], [], [], []
        
        sent_attn_mask = torch.stack(batch['sent_attn_mask']).permute(1,0)
        max_evid = sent_attn_mask.sum(-1).max().item()
        n_evid = len(batch['text'])-1
        assert max_evid <= n_evid
        sent_attn_mask = sent_attn_mask[:, :self.max_evid_sets]

        for i in range(1, self.max_evid_sets+1):
            sent_list = list(zip(batch['text'][0], batch['text'][i]))
            labels = batch['multi_labels'][i-1]

            # TODO: Move to HESMDataset to compute max_len only once
            max_len = self.step_max_len(sent_list, tot_max_len)
            
            # TODO: Validate whether max_length needs to be dynamic based on the batch, else move to HESMDataset
            encoded_batch = self.tokenizer.batch_encode_plus(sent_list, max_length=max_len, pad_to_max_length=True)
            
            input_ids = torch.tensor(encoded_batch['input_ids'])
            input_mask = torch.tensor(encoded_batch['attention_mask'])
            token_type_ids = torch.tensor(encoded_batch['token_type_ids'])
            
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            labels = labels.cuda()

            b_input_ids.append(input_ids)
            b_input_mask.append(input_mask)
            b_labels.append(labels)
            b_token_type_ids.append(token_type_ids)

        sent_attn_mask = sent_attn_mask.cuda()
        retval  = self.model(b_input_ids, 
                             multi_token_type_ids=b_token_type_ids, 
                             attention_mask=b_input_mask, 
                             multi_labels=b_labels,
                             final_label=batch['labels'],
                             sent_attn_mask=sent_attn_mask)

        return retval


def full_eval_model_hesm(hesm_model, model, dataloader, criterion, dev_data_list):
    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }

    print("Evaluating ...")
    model.eval()
    n_correct = 0
    total_size = 0
    loss = 0

    y_pred_list = []
    y_pred_mult_list = []
    y_true_list = []
    y_id_list = []

    with torch.no_grad():  # Important fixing.

        for batch in dataloader:
            curloss, out, multiout = hesm_model.step(batch)
            
            y = batch['labels'].cuda()
            y_id_list.extend(list(batch['pid']))

            max_index = torch.max(out, 1)[1]
            
            n_correct += (max_index.view(y.size()) == y).sum().item()
            total_size += y.size(0)
            
            y_pred_list.extend(max_index.view(y.size()).tolist())
            y_true_list.extend(y.tolist())

            loss += curloss.mean()
            
            if multiout is not None:
                cur_s_label = []
                for sout in multiout:
                    cur_s_label.append(torch.max(sout, 1)[1].tolist())
                cur_s_label = list(zip(*cur_s_label))
                y_pred_mult_list.extend(cur_s_label)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_pred_list) == len(dev_data_list)
        assert len(y_true_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['id'])

            dev_data_list[i]['predicted_label'] = id2label[y_pred_list[i]]
            
            if len(y_pred_mult_list) > 0: 
                dev_data_list[i]['multi_predicted_label'] = [id2label[x] for x in y_pred_mult_list[i]]

            if len(dev_data_list[i]['predicted_sentids']) == 0:
                dev_data_list[i]['predicted_label'] = "NOT ENOUGH INFO"

        print('n_correct:', n_correct)
        print('total_size:', total_size)

        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_data_list, dev_data_list, mode=eval_mode,
                                                                    verbose=False)
        print("Fever Score(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

        avg_loss = loss / total_size

    return strict_score, avg_loss


def construct_evidence_sets(upstream_file, d_list, a_list):    
    d_list = common.load_jsonl(d_list)
    a_list = common.load_jsonl(a_list)
    ans_list = common.load_jsonl(upstream_file)
    
    augmented_dict = dict()
    for sent_item in tqdm(a_list):
        selection_id = sent_item[0]['selection_id'][0]
        org_id = int(selection_id.split('<##>')[0])
        if org_id in augmented_dict:
            augmented_dict[org_id].append(sent_item)
        else:
            augmented_dict[org_id] = sent_item

    assert len(d_list) == len(ans_list)
    for item, ans in zip(d_list, ans_list):
        if int(item['id']) not in augmented_dict:
            predicted_sentids = []
        else:
            cur_predicted_sentids_dict = dict() 
            cur_predicted_sentids = []
            sents = augmented_dict[int(item['id'])]
            for sent_i in sents:
                if type(sent_i['fsid']) == list:
                    fsidt = sent_i['fsid'][0]+'<SENT_LINE>'
                    for s in sent_i['fsid'][1:]:
                        fsidt += str(s)
                    sent_i['fsid'] = fsidt
                
                issid = False
                for fsidt, _, _ in cur_predicted_sentids_dict:
                    if sent_i['fsid'] == fsidt:
                        issid = True
                        break

                if not issid:
                    cur_pred_list = []
                else:
                    continue

                assert len(sent_i['selection_id']) == len(sent_i['score'])
                assert len(sent_i['selection_id']) == len(sent_i['prob'])

                for sid, score, prob in zip(sent_i['selection_id'],sent_i['score'],sent_i['prob']):
                    if prob > 0.8:
                        cur_pred_list.append((sid.split('<##>')[1], score, prob))

                cur_predicted_sentids_dict[(sent_i['fsid'], sent_i['fscore'], sent_i['fprob'])] = cur_pred_list

            for sid1, score1, prob1 in ans['scored_sentids']:
                isthere = False
                for sid2, _, _ in cur_predicted_sentids_dict:
                    if sid1 == sid2:
                        isthere = True
                        break
                if not isthere:
                    cur_predicted_sentids_dict[(sid1, score1, prob1)] = []

            sorted_keys = sorted(cur_predicted_sentids_dict, key=lambda x: (-x[1]))        
            
            predicted_sentids = []
            
            for idx, k in enumerate(sorted_keys):  
                cur_predicted_sentids = []
                if idx < 5:
                    cps_tmp = cur_predicted_sentids_dict[k]
                    cps_tmp = sorted(cps_tmp, key=lambda x: (-x[1]))
                    
                    pres=False
                    for z in cur_predicted_sentids:
                        if k[0] == z[0]:
                            pres = True
                            break
                
                    if not pres:
                        cur_predicted_sentids.append(k)
 
                    combevi = cps_tmp[:2]
                    for cur_sent in combevi:
                        pres = False
                        for z in cur_predicted_sentids:
                            if cur_sent[0] == z[0]:
                                pres = True
                                break

                        if not pres:
                            cur_predicted_sentids.append(cur_sent)

                    predicted_sentids.append(cur_predicted_sentids)

        item['predicted_sentids'] = predicted_sentids

        if len(item['predicted_sentids']) == 0:
            for idx2, (sid1, score1, prob1) in enumerate(ans['scored_sentids']):
                if idx2 < 5:
                    isthere = False

                    for sentid in item['predicted_sentids']:
                        for sid2, _, _ in sentid:
                            if sid1 == sid2:
                                isthere = True
                                break
                    if not isthere:
                        item['predicted_sentids'].append([(sid1, score1, prob1)])
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


def train_fever_hesm(model_name="albert-base-v2"):
    seed = 12
    num_epoch = 4
    batch_size = 16

    dev_prob_threshold = 0.5
    train_prob_threshold = 0.35
    train_sample_top_k = 12
    
    experiment_name = f"hesm|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"
    resume_model = None 

    dev_subset_file = None # str(config.RESULT_PATH /
            # "pipeline_r_aaai_doc_exec/2019_10_07_10:14:16_r/dev_selected_subset1_shared_task_dev.jsonl")
    train_subset_file = None # str(config.RESULT_PATH /
            # "pipeline_r_aaai_doc/2019_10_27_16:48:33_r/dev_selected_subset1_train.jsonl")

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    # Get dev data
    dev_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc_exec/2019_10_07_10:14:16_r/doc_retr_2_shared_task_dev.jsonl" #19998
    dev_d_list = config.T_FEVER_DEV_JSONL #19998
    dev_a_list = config.RESULT_PATH / "../verifyevaldictalbert_6.jsonl" #19083
    upstream_dev_list = construct_evidence_sets(dev_upstream_file, dev_d_list, dev_a_list)
    
    complete_upstream_dev_data = select_sent_with_prob_for_eval_esets(dev_subset_file, config.T_FEVER_DEV_JSONL, upstream_dev_list, tokenized=True)
    print("Sample dev data length:", len(complete_upstream_dev_data))

    # Get train data
    train_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc/2019_10_27_16:48:33_r/doc_retr_2_train.jsonl" #145449
    train_d_list = config.T_FEVER_TRAIN_JSONL #145449
    train_a_list = config.RESULT_PATH / "../verifytraindictsubset.jsonl" #38883
    upstream_train_list = construct_evidence_sets(train_upstream_file, train_d_list, train_a_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

    model = AlbertForHESM.from_pretrained(
        model_name, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 3, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    if torch.cuda.device_count() > 1:
        print("More than 1 gpu device found...")
        model = nn.DataParallel(model)
    
    model.to(device)

    start_lr = 2e-5
    optimizer = AdamW(model.parameters(),
                  lr = start_lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

    if resume_model is not None:
        print("Resume From:", resume_model)
        load_model(resume_model, model.module.albert, optimizer)


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
        print("Resampling...")
        # Resampling

        complete_upstream_train_data = adv_simi_sample_with_prob_esets(train_subset_file, config.T_FEVER_TRAIN_JSONL,
                                                                      upstream_train_list,
                                                                      tokenized=True)

        print("Sample train data length:", len(complete_upstream_train_data))

        
        sent_list, label_list, multi_labels_list, sent_attn_mask_list, pid_list = hesm_model.read(complete_upstream_train_data)

        del complete_upstream_train_data
        
        train_dataset = HESMDataset({'text': sent_list, 'labels': label_list, 'pid': pid_list, 'multi_labels': multi_labels_list, 'sent_attn_mask': sent_attn_mask_list})    
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
        
        if i_epoch == 0:
            accumulation_steps = 2 # accumulate gradients for increasing `batch_size` by a factor of `accumulation_steps`
            steps_per_epoch = len(train_dataloader)
            total_steps = steps_per_epoch * num_epoch # / accumulation_steps
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
            save_epoch = 0.5 # evaluate and save every `save_epoch` epochs

        print("steps per epoch : ", len(train_dataloader))
        optimizer.zero_grad()
        for i, batch in tqdm(enumerate(train_dataloader)):
            model.train()
            loss, _, _ = hesm_model.step(batch)
            loss = loss.mean()
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                scheduler.step()
                optimizer.zero_grad()
            iteration += 1

            mod = steps_per_epoch * save_epoch
            if iteration % mod == 0:
                sent_list, label_list, multi_labels_list, sent_attn_mask_list, pid_list = hesm_model.read(complete_upstream_dev_data)

                eval_dataset = HESMDataset({'text': sent_list, 'labels': label_list, 'pid': pid_list, 'multi_labels': multi_labels_list, 'sent_attn_mask': sent_attn_mask_list})    
                eval_dataloader = DataLoader(
                    eval_dataset,  # The training samples.
                    sampler = SequentialSampler(eval_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
                
                dev_score, dev_loss = full_eval_model_hesm(hesm_model, model, eval_dataloader, criterion, complete_upstream_dev_data)

                print(f"Dev:{dev_score}/{dev_loss}")

                need_save = True
                if dev_score > best_dev:
                    best_dev = dev_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({dev_score})_loss({dev_loss})_seed({seed})'
                    )

                    save_model(save_path, model, optimizer)


if __name__ == "__main__":
    train_fever_hesm()