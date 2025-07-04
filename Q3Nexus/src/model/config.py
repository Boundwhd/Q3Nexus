class Qwen3Config():
    def __init__(
        self,
        bos_token_id=151643,
        eos_token_id=151645,
        hidden_size=2560,
        head_dim=128,
        intermediate_size=9728,
        max_position_embeddings=40960,
        num_attention_heads=32,
        num_hidden_layers=36,
        num_key_value_heads=8,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        vocab_size=151936,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        self.bos_token_id=bos_token_id
        self.eos_token_id=eos_token_id
