
from transformers import Wav2Vec2ForCTC, HubertForCTC, WavLMForCTC

def model_factory(model_id,
                 cache_dir,
                 activation_dropout,
                 attention_dropout,
                 hidden_dropout,
                 feat_proj_dropout,
                 mask_time_prob,
                 gradient_checkpointing,
                 layerdrop,
                 ctc_loss_reduction,
                 pad_token_id,
                 vocab_size,
                  model_type='wav2vec',
                  final_layer=None,
                  freeze_encoder=False):

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
                        cache_dir=cache_dir,
                        activation_dropout=activation_dropout,
                        attention_dropout=attention_dropout,
                        hidden_dropout=hidden_dropout,
                        feat_proj_dropout=feat_proj_dropout,
                        mask_time_prob=mask_time_prob,
                        gradient_checkpointing=gradient_checkpointing,
                        layerdrop=layerdrop,
                        ctc_loss_reduction=ctc_loss_reduction,
                        pad_token_id=pad_token_id,
                        vocab_size=vocab_size
        )
    elif model_type == 'hubert':
        if model_id==None:
            model_id = "facebook/hubert-large-ls960-ft"
        model = HubertForCTC.from_pretrained(
                    model_id, 
                    cache_dir=cache_dir,
                    activation_dropout=activation_dropout,
                    attention_dropout=attention_dropout,
                    hidden_dropout=hidden_dropout,
                    feat_proj_dropout=feat_proj_dropout,
                    mask_time_prob=mask_time_prob,
                    gradient_checkpointing=gradient_checkpointing,
                    layerdrop=layerdrop,
                    ctc_loss_reduction=ctc_loss_reduction,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size)
        
    elif model_type == 'wavlm':
        if model_id == None:
            model_id = "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
        model = WavLMForCTC.from_pretrained(
                    model_id, 
                    cache_dir=cache_dir,
                    activation_dropout=activation_dropout,
                    attention_dropout=attention_dropout,
                    hidden_dropout=hidden_dropout,
                    feat_proj_dropout=feat_proj_dropout,
                    mask_time_prob=mask_time_prob,
                    gradient_checkpointing=gradient_checkpointing,
                    layerdrop=layerdrop,
                    ctc_loss_reduction=ctc_loss_reduction,
                    pad_token_id=pad_token_id,
                    vocab_size=vocab_size)
                    
    
    if final_layer is not None:
        # For CTC loss the final layer default is nn.Linear(output_hidden_size, config.vocab_size)
        model.lm_head = final_layer
        model.post_init()
        
    # set requires_grad to False for the feature extraction part (suff. trained during pre-training)
    if freeze_encoder:
        model.freeze_feature_encoder()

    # gradient checkpointing to save memory
    #model.gradient_checkpointing_enable()

    return model