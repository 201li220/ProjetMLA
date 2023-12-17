import json
from modified_transformers import BertConfig,BertForSequenceClassification

def view_config(path):
    
    with open(path, 'r') as file:
        config = json.load(file)

    num_layers = config.get('num_hidden_layers', None)
    num_attention_heads = config.get('num_attention_heads', None)
    hidden_size = config.get('hidden_size', None)
    max_position_embeddings = config.get('max_position_embeddings', None)
    adapter = config.get('adapter', None)
    adapter_size = config.get('adapter_size', None)
    frozen_mode = config.get('frozen_mode', None)
    
    print(f"Number of layers : {num_layers}")
    print(f"Number of attention heads : {num_attention_heads}")
    print(f"Hidden size : {hidden_size}")
    print(f"Adapter : {adapter}")
    print(f"Adapter size : {adapter_size}")

    modelconfig = BertConfig(hidden_size = hidden_size,
                    max_position_embeddings=max_position_embeddings,
                    num_attention_heads=num_attention_heads,
                    num_hidden_layers = num_layers,
                    adapter = adapter,
                    adapter_size = adapter_size,
                    frozen_mode = frozen_mode
                   )
    
    model = BertForSequenceClassification(modelconfig)

    total_parameters = 0
    total_trainable_parameters = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()  # Get the number of elements in the tensor
        total_parameters += num_params
        if param.requires_grad:
            total_trainable_parameters +=num_params


    print(f"Total number of parameters in the model: {total_parameters}")
    print(f"Total number of trainable parameters in the model: {total_trainable_parameters}")



    