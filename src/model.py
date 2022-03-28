
from transformers import Wav2Vec2ForCTC, HubertForCTC, WavLMForCTC

def model_factory(processor,  
                  model_type='wav2vec',
                  model_id=None,
                  final_layer=None,
                  attention_dropout=0.1,
                  hidden_dropout=0.1,
                  feat_proj_dropout=0.0,
                  mask_time_prob=0.05,
                  layerdrop=0.1,):

    """
    processor: wav2vec processor with vocab configurations
    model_type: wav2vec, wavlm or hubert
    model_id: model ids, exs:
        "facebook/wav2vec2-base-960h"
        "facebook/wav2vec2-large-xlsr-53"
        "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
        "facebook/hubert-large-ls960-ft"
    final_layer: torch layers. output has to be size of vocab for CTC

    """

    if model_type == 'wav2vec':
        if model_id==None:
            model_id = "facebook/wav2vec2-base"
        model = Wav2Vec2ForCTC.from_pretrained(
        model_id, 
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        feat_proj_dropout=feat_proj_dropout,
        mask_time_prob=mask_time_prob,
        layerdrop=layerdrop,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
        )
    elif model_type == 'hubert':
        if model_id==None:
            model_id = "facebook/hubert-large-ls960-ft"
        model = HubertForCTC.from_pretrained(
        model_id, 
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        feat_proj_dropout=feat_proj_dropout,
        mask_time_prob=mask_time_prob,
        layerdrop=layerdrop,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
        )
    elif model_type == 'wavlm':
        if model_id == None:
            model_id = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
        model = WavLMForCTC.from_pretrained(
        model_id, 
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        feat_proj_dropout=feat_proj_dropout,
        mask_time_prob=mask_time_prob,
        layerdrop=layerdrop,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
        )
    
    if final_layer is not None:
        # For CTC loss the final layer default is nn.Linear(output_hidden_size, config.vocab_size)
        model.lm_head = final_layer
        model.post_init()
        
    # set requires_grad to False for the feature extraction part (suff. trained during pre-training)
    model.freeze_feature_encoder()

    # gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    return model