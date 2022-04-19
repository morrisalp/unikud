import torch
from torch import nn
from transformers import CanineModel, CaninePreTrainedModel, CanineTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

NIKUD_START_ORD = 1456
NIKUD_END_ORD = 1474
SPECIAL_ORDS = {1466, 1469, 1470, 1471, 1472}
EXTENDED_NIKUD = {chr(i) for i in range(NIKUD_START_ORD, NIKUD_END_ORD + 1)}
NIKUD = {c for c in EXTENDED_NIKUD if ord(c) not in SPECIAL_ORDS}
N_VOWELS = len(NIKUD) - 3 # not including dagesh, shin dot, sin dot

idx2chr = dict()
j = 0
for i in range(NIKUD_START_ORD, NIKUD_END_ORD + 1):
    if i not in SPECIAL_ORDS:
        idx2chr[j] = chr(i)
        j += 1

ABG = set('אבגדהוזחטיכךלמםנןסעפףצץקרשת')

class UnikudModel(CaninePreTrainedModel):
    # based on CaninePreTrainedModel
    # slightly modified for multilabel classification
    
    def __init__(self, config, num_labels=(len(NIKUD) + 1)):
        # Note: one label for each nikud type, plus one for the deletion flag
        super().__init__(config)
        config.num_labels = num_labels
        self.num_labels = config.num_labels
        
        self.canine = CanineModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights and apply final processing
        self.post_init()
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.canine(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Unikud:

    def __init__(self, hub_name='malper/unikud', device='cpu', max_len=2046):
        self.tokenizer = CanineTokenizer.from_pretrained(hub_name)
        self.model = UnikudModel.from_pretrained(hub_name)
        self.model.to(device)
        self.device = device
        self.max_len = max_len
        # Note: max_len is 2 less than model's input length 2048,
        # to account for BOS and EOS tokens
        
    def _decode_nikud_probs(self, probs, d_thresh=0.5, v_thresh=0.5, o_thresh=0.5):
        # probs: N_TARGET_LABELS probabilities for nikkud for a single character, or deletion (last prob)

        # Note: first N_VOWELS are mutually exclusive vowels
        # next are dagesh, shin dot, and sin dot
        # finally the deletion flag
        
        vowel_probs = probs[:N_VOWELS]
        other_probs = probs[N_VOWELS:-1]
        del_prob = probs[-1]
        
        maxvow = vowel_probs.max().item()
        argmaxvow = vowel_probs.argmax().item()
        
        if del_prob > d_thresh:
            return None # special symbol for deletion
    
        out = ''
    
        if maxvow > v_thresh:
            out += idx2chr[argmaxvow]

        for i, p in enumerate(other_probs):
            if p > o_thresh:
                out += idx2chr[N_VOWELS + i]

        return out
        
    
    def __call__(self, text, **kwargs):
        
        assert len(text) <= self.max_len, f'Input text cannot be longer than {self.max_len} characters.'
        
        X = self.tokenizer([text], return_tensors='pt').to(self.device)
        logits = self.model(**X).logits.detach()[0]
        probs = torch.sigmoid(logits)
        
        output = ''
        for i, char in enumerate(text):
            output += char
            if char in ABG:
                char_probs = probs[i + 1]

                decoded = self._decode_nikud_probs(char_probs, **kwargs)

                if decoded is None and len(output) > 0:
                    output = output[:-1]
                else:
                    output += decoded

        return output