import unittest
import torch
from model.config import DenseTransformerConfig
from model.dense_transformer import DenseTransformer

class TestDenseTransformerForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = DenseTransformerConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=2,
            d_ff=256,
            max_seq_len=64,
            dropout=0.0
        )
        self.model = DenseTransformer(self.config)

    def test_forward_return_format(self):
        """forward의 반환 포맷 및 손실 동작 검증"""
        # 1. labels가 없을 때 (logits만 나오고 loss/main_loss는 None)
        x = torch.randint(0, 1000, (2, 8))
        logits, loss, main_loss = self.model(x)
        self.assertEqual(logits.shape, (2, 8, 1000))
        self.assertIsNone(loss)
        self.assertIsNone(main_loss)

        # 2. labels가 있을 때 (3-tuple 반환 및 loss가 계산됨)
        labels = torch.randint(0, 1000, (2, 8))
        logits, loss, main_loss = self.model(x, labels)
        self.assertEqual(logits.shape, (2, 8, 1000))
        self.assertIsNotNone(loss)
        self.assertIsNotNone(main_loss)
        self.assertTrue(loss > 0)
        self.assertEqual(loss.item(), main_loss.item())

    def test_parameter_count_scaling(self):
        """기본 설정(12 layers, d_ff=3072) 모델의 파라미터가 162M ± 5M 범위에 있는지 검증"""
        config = DenseTransformerConfig()
        model = DenseTransformer(config)
        total_params = sum(p.numel() for p in model.parameters())
        # 162.4M 인근 범위 (157M ~ 167M)
        self.assertTrue(157e6 < total_params < 167e6, f"Expected params to be ~162.4M, got {total_params}")

    def test_causal_masking(self):
        """인과적 마스킹(Causal Masking)이 정상 작동하여 미래 토큰이 과거 토큰 예측에 영향을 주지 않는지 검증"""
        self.model.eval()
        
        # 두 시퀀스 생성: 첫 부분은 동일하고 끝부분만 다름
        # x1: [A, B, C]
        # x2: [A, B, D]
        x1 = torch.tensor([[10, 20, 30]])
        x2 = torch.tensor([[10, 20, 40]])
        
        with torch.no_grad():
            logits1, _, _ = self.model(x1)
            logits2, _, _ = self.model(x2)
            
        # 첫 번째 및 두 번째 토큰의 logits은 미래의 세 번째 토큰(30 vs 40)에 영향을 받지 않고 완전히 같아야 함
        # logits shape: (1, seq_len, vocab_size)
        self.assertTrue(torch.allclose(logits1[:, :2, :], logits2[:, :2, :], atol=1e-5))
        
        # 세 번째 토큰의 logits은 서로 달라야 함
        self.assertFalse(torch.allclose(logits1[:, 2, :], logits2[:, 2, :], atol=1e-5))

if __name__ == '__main__':
    unittest.main()
