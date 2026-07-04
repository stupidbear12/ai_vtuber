# -*- coding: utf-8 -*-
"""Brain 모듈 모델 테스트"""
import sys
sys.path.insert(0, ".")
import torch
from app.models import HippocampusNet, EMOTIONS

def test_model():
    model = HippocampusNet()
    print(f"Synapse count: {model.count_parameters():,}")

    x = torch.randn(1, 384)
    out = model(x)
    print(f"emotion_logits shape: {out['emotion_logits'].shape}")
    print(f"engagement shape: {out['engagement'].shape}")
    print(f"topic_embedding shape: {out['topic_embedding'].shape}")

    emo, probs = model.predict_emotion(x)
    eng = model.predict_engagement(x)
    topic = model.get_topic_vector(x)
    print(f"Predicted emotion: {emo}")
    print(f"Engagement: {eng:.4f}")
    print(f"Topic vector length: {len(topic)}")
    print("MODEL TEST PASSED")

def test_embedder():
    from app.embedder import SentenceEmbedder
    emb = SentenceEmbedder()
    vec = emb.encode("안녕하세요 시온!")
    print(f"Embedding shape: {vec.shape}")
    print(f"Embedding dim: {emb.dim}")

    pair = emb.encode_pair("시온 노래 해줘!", "노래는 못하지만 DJ는 할 수 있지!")
    print(f"Pair embedding shape: {pair.shape}")
    print("EMBEDDER TEST PASSED")

def test_hippocampus():
    from app.hippocampus import HippocampusEngine, ConversationSample

    engine = HippocampusEngine()

    # Query test
    result = engine.query("시온아 안녕!")
    print(f"Query result: emotion={result['suggested_emotion']}, "
          f"engagement={result['engagement_pred']}")
    print(f"Context hint: {result['context_hint']}")

    # Add samples
    for i in range(5):
        sample = ConversationSample(
            message=f"테스트 메시지 {i}",
            response=f"테스트 응답 {i}",
            emotion="happy",
            engagement=0.7,
        )
        status = engine.add_sample(sample)
    print(f"Buffer: {status['buffer_size']}/{status['experience_threshold']}")

    # Stats
    stats = engine.get_stats()
    print(f"Stats: synapses={stats['synapse_count']:,}, "
          f"experiences={stats['total_experiences']}, "
          f"samples={stats['total_samples_learned']}")
    print("HIPPOCAMPUS TEST PASSED")

if __name__ == "__main__":
    print("=== Model Test ===")
    test_model()
    print()
    print("=== Embedder Test ===")
    test_embedder()
    print()
    print("=== Hippocampus Test ===")
    test_hippocampus()
    print()
    print("ALL TESTS PASSED")
