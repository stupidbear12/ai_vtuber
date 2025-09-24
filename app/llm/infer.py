import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from .tokenizer import SimpleCharTokenizer

MODEL_PATH = os.environ.get("VTUBER_MODEL_PATH", "llm/model/model.keras")

def load_model_and_tokenizer():
    tok = SimpleCharTokenizer()
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model, tok

def top_k_logits(logits: tf.Tensor, k: int = 50) -> tf.Tensor:
    """배치별 상위 k개만 남기고 나머지는 -inf로 자르기"""
    if k is None or k <= 0:
        return logits
    values, _ = tf.math.top_k(logits, k=k)           
    min_vals = values[:, -1, None]                   
    neg_inf = tf.constant(-1e9, dtype=logits.dtype)
    return tf.where(logits < min_vals, neg_inf, logits)

def sample_next_token(
    logits: tf.Tensor, temperature: float = 1.0, top_k: int = 50
) -> np.ndarray:
    
    t = max(float(temperature), 1e-6)
    logits = logits / t
    logits = top_k_logits(logits, k=top_k)
    probs = tf.nn.softmax(logits, axis=-1).numpy()   
    V = probs.shape[-1]
    next_ids = [np.random.choice(V, p=p) for p in probs]
    return np.array(next_ids, dtype=np.int32)

def generate(
    model,
    tok: SimpleCharTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_k: int = 50,
) -> str:
    
    ids = tok.encode(prompt, add_special_tokens=True)[None, :]  

    for _ in range(max_new_tokens):
        logits = model(ids)            
        last_logits = logits[:, -1, :] 
        next_id = sample_next_token(last_logits, temperature=temperature, top_k=top_k)
        if int(next_id[0]) == tok.eos_id:
            break
        ids = np.concatenate([ids, next_id[:, None]], axis=1)  

    return tok.decode(ids[0])

if __name__ == "__main__":
    try:
        model, tok = load_model_and_tokenizer()
    except Exception as e:
        print(f"[infer] 모델 로드 실패: {e}")
        print("먼저 'python -m llm.train' 으로 학습/저장을 완료하세요.")
        raise SystemExit(1)

    prompt = "안녕하세요, 오늘 방송 컨셉 추천해주세요"
    out = generate(model, tok, prompt, max_new_tokens=120, temperature=0.9, top_k=50)
    print("=== 프롬프트 ===")
    print(prompt)
    print("=== 생성된 텍스트 ===")
    print(out)


