from transformers import EsmConfig

CUSTOM_CONFIG = {
    "model": {
        "model_name": "facebook/esm2_t30_150M_UR50D",
        "max_position_embeddings": 4098,  # Maximum sequence length (including bos and eos tokens)
        "max_batch_size": 8196,  # Maximum number of tokens per batch
        "use_fa": True,  # Fast Attention
        "emb_layer_norm_before": True,  # LayerNorm before embedding projection
        "num_layers": 30,  # Number of layers
        "hidden_size": 640,  # Hidden size maybe 960
        "intermediate_size": 2560,  # Intermediate size 4*hidden_size maybe 3840
        "num_attention_heads": 20,  # Number of attention heads maybe 20
    },
    "training": {
        "gradient_accumulation_steps": 64,  # Number of steps to accumulate gradients 32
        # "eval_accumulation_steps": 2,  # Number of steps to accumulate evaluation metrics
        "learning_rate": 4e-4,
        "optimizer": {
            "beta_1": 0.99,
            "beta_2": 0.98,
            "epsilon": 1e-12,
            "weight_decay": 0.01,
        },
        "gradient_clipping": 0.5,
        "lr_scheduler": "linear",
        "evaluation_strategy": "steps",
        "total_steps": 120000,  # Total optimization steps (after accumulation) #125k steps would be ~45 epochs (like ESM-2) with our dataset size
        "warmup_steps": 2000,
        "mlm_probability": 0.25,  # Masking probability (could be 0.2 or 0.15 from "Training Compute-Optimal Protein Language Models" paper) maybe split the difference and call it 0.2
        "logging_steps": 100,
        "eval_steps": 10000,  
        "save_steps": 10000,  # Save every 1000 steps
        "output_dir": "./runs/pLM120K_NL", #########################
        "mixed_precision": "fp16",
        "dataloader_num_workers": 4,
        "dataloader_prefetch_factor": 4, 
        "dataloader_pin_memory": True,
        "seed": 100,
    },
}


def get_model_config():
    """
    Returns a Hugging Face EsmConfig object based on the model parameters in CONFIG.
    """
    return EsmConfig.from_dict(CUSTOM_CONFIG["model"])


def get_training_config():
    """
    Returns the training configuration as a dictionary.
    """
    return CUSTOM_CONFIG["training"]