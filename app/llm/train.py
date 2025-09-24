import os
import tensorflow as tf
from .model import build_char_transformer
from .data import build_tf_dataset
from .tokenizer import SimpleCharTokenizer
import llm.tf_setup

MAX_LEN = 128
BATCH = 32
EPOCHS = 5
MODEL_OUT = os.environ.get("VTUBER_MODEL_PATH", "llm/model/model.keras")

CORPUS = [
    "안녕하세요, 오늘도 즐거운 방송 시작해요!",
    "시청자 여러분 구독과 좋아요는 큰 힘이 됩니다!",
    "오늘은 리듬 게임 함께 해요, 미션 받습니다.",
    "슈퍼챗 감사합니다!",
    "AI 버튜버 프로젝트 화이팅!",
]

def main():
    tok = SimpleCharTokenizer()
   
    train_ds = build_tf_dataset(
            texts=CORPUS,
            tok = tok,
            max_len = MAX_LEN,
            batch_size = BATCH,
            shuffle = True
        )
    

    model, _ = build_char_transformer(
        embed_dim=128, num_heads=8, ff_dim=512, num_layers=4, max_len=MAX_LEN
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor =  "loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="loss"),
    

    ]

    model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)

    model.save(MODEL_OUT)
    print(f" 모델 저장됨 : {MODEL_OUT}")

if __name__ == "__main__":
    main()
