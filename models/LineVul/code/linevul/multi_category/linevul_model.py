import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # if the model is two classification, here should keep the below line. Since when it is train, I keep the below line.
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def forward_feature_list(self, features, **kwargs):
        features_list = []
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        # print(x.shape)
        features_list.append(x)
        x = torch.tanh(x)
        # print(x.shape)
        features_list.append(x)
        x = self.dropout(x)
        # print(x.shape)
        features_list.append(x)
        x = self.out_proj(x)
        # print(x.shape)
        features_list.append(x)
        return x, features_list


class Model(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None, output_feature_list=False):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1),
                                               output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            feature_list = []
            if input_ids is not None:
                outputs = \
                self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            feature_list.append(outputs)
            logits = self.classifier(outputs)
            feature_list.append(logits)
            prob = torch.softmax(logits, dim=-1)
            feature_list.append(prob)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                if not output_feature_list:
                    return loss, prob
                else:
                    return loss, prob, feature_list
            else:
                if not output_feature_list:
                    return  prob
                else:
                    return  prob, feature_list

    def feature_list(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1),
                                               output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            output_attentions = True
            output_hidden_states=True
            # para = []
            if input_ids is not None:
                outputs_original = \
                self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions, output_hidden_states=output_hidden_states)
                outputs = outputs_original[0]
                hidden_states = outputs_original.hidden_states
                attentions = outputs_original.attentions
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            # para.append(outputs)
            # logits = self.classifier(outputs)
            # para.append(logits)
            # logits = self.classifier(outputs)
            logits, features_list = self.classifier.forward_feature_list(outputs)

            hidden_states = [item.mean(dim=1) for item in hidden_states]
            features_list = hidden_states + features_list
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, features_list
            else:
                return logits, prob, features_list
