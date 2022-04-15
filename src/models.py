from torch import nn
from transformers import CanineModel, CanineForTokenClassification, CaninePreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from hebrew_utils import NIKUD

class KtivMaleModel(CanineForTokenClassification):
    
    def __init__(self, config):
        assert hasattr(config, 'num_labels') and config.num_labels == 3
        super().__init__(config)

class UnikudModel(CaninePreTrainedModel):
    # based on CaninePreTrainedModel
    # slightly modified for multilabel classification
    
    def __init__(self, config, num_labels=len(NIKUD)):
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


if __name__ == '__main__':

    km_model = KtivMaleModel.from_pretrained("google/canine-c", num_labels=3)
    u_model = UnikudModel.from_pretrained("google/canine-c")