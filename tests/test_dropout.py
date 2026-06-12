import unittest
import torch
from model.config import DenseTransformerConfig
from model.dense_transformer import DenseTransformer


class TestDropoutRegularization(unittest.TestCase):
    def setUp(self):
        # 결정론적 연산을 위해 난수 시드 설정
        torch.manual_seed(42)

    def test_dropout_layers_present(self):
        """모델 내부 컴포넌트에 드롭아웃 레이어가 정상 정의되어 있는지 검사"""
        config = DenseTransformerConfig(dropout=0.1)
        model = DenseTransformer(config)

        # 1. MultiHeadAttention 드롭아웃 확인 (모든 블록)
        for layer in model.layers:
            self.assertTrue(hasattr(layer.attention, "attn_dropout"))
            self.assertTrue(hasattr(layer.attention, "resid_dropout"))
            self.assertEqual(layer.attention.attn_dropout.p, 0.1)
            self.assertEqual(layer.attention.resid_dropout.p, 0.1)

        # 2. Dense FFN 드롭아웃 확인 (모든 블록)
        for layer in model.layers:
            self.assertTrue(hasattr(layer.ffn.ffn, "dropout"))
            self.assertEqual(layer.ffn.ffn.dropout.p, 0.1)

    def test_dropout_train_vs_eval_behavior(self):
        """학습 모드(train)와 평가 모드(eval)에서 드롭아웃 레이어의 동적 변경 및 출력 검증"""
        # 강렬한 드롭아웃 효과 확인을 위해 0.5(50%)로 지정
        config = DenseTransformerConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=2,
            d_ff=256,
            max_seq_len=64,
            dropout=0.5,
        )
        model = DenseTransformer(config)
        x = torch.randint(0, 1000, (1, 10))

        # [검증 1] 평가 모드(eval): 드롭아웃이 꺼지므로 매번 결과가 100% 동일해야 함
        model.eval()
        with torch.no_grad():
            out_eval1, _, _ = model(x)
            out_eval2, _, _ = model(x)
        self.assertTrue(torch.allclose(out_eval1, out_eval2, atol=1e-7))

        # [검증 2] 학습 모드(train): 드롭아웃이 켜지므로 매 포워드 시 다르게 랜덤 마스킹되어 결과가 달라야 함
        model.train()
        out_train1, _, _ = model(x)
        out_train2, _, _ = model(x)

        # 두 텐서가 완전히 동일하지는 않음을 확인 (드롭아웃 마스킹 동작 증명)
        self.assertFalse(torch.allclose(out_train1, out_train2, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
