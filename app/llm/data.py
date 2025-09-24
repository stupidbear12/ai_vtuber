import numpy as np
import tensorflow as tf
from .tokenizer import SimpleCharTokenizer

def make_xy_from_ids(ids, max_len=256):
    if len(ids) < 3:
        return None, None
    x = ids[:-1]
    y = ids[1:]
    return x[:max_len], y[:max_len]

def build_tf_dataset(texts, tok=None, max_len=128, batch_size=32, shuffle=True):
    tok = tok or SimpleCharTokenizer()
    pairs = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=True)
        x, y = make_xy_from_ids(ids, max_len=max_len)
        if x is None:
            continue
        pairs.append((x, y))

    def gen():
        for x, y in pairs:
            yield x, y

    
    sig = (
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),


    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(pairs)))
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([max_len], [max_len]),
        padding_values=(tf.constant(0, tf.int32), tf.constant(0, tf.int32)),
    ).prefetch(tf.data.AUTOTUNE)

    return ds

