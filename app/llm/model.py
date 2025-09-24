import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .tokenizer import SimpleCharTokenizer


class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len=256, **kwargs):
        
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb   = layers.Embedding(max_len, d_model)

    def call(self, x):
       
        T = tf.shape(x)[1]
       
        positions = tf.range(0, T)              
        pos = self.pos_emb(positions)          
        tok = self.token_emb(x)                  
        return tok + pos[tf.newaxis, :, :]     

    def get_config(self):
       
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_len": self.max_len,
        })
        return config


def transformer_block(x, num_heads, d_model, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )
    attn_out = attn(x, x, use_causal_mask=True)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization(epsilon=1e-5)(x)

   
    ffn = keras.Sequential([
        layers.Dense(ff_dim, activation="gelu"),
        layers.Dropout(dropout),
        layers.Dense(d_model),
    ])
    ffn_out = ffn(x)
    x = layers.Add()([x, ffn_out])
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    return x


def build_char_transformer(embed_dim=128, num_heads=8, ff_dim=512, num_layers=4, max_len=256):
    tok = SimpleCharTokenizer()
    vocab_size = tok.vocab_size

    inputs = keras.Input(shape=(None,), dtype="int32")
    x = PositionalEmbedding(vocab_size, embed_dim, max_len=max_len)(inputs)

    for _ in range(num_layers):
        x = transformer_block(x, num_heads=num_heads, d_model=embed_dim, ff_dim=ff_dim)

    logits = layers.Dense(vocab_size)(x)

    model = keras.Model(inputs, logits, name="char_transformer_lm")
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=loss,
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    return model, tok

