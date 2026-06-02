import unittest
from model.config import MoETransformerConfig
from model.moe_transformer import MoETransformer

class TestMoETransformerConfig(unittest.TestCase):
    def test_default_config(self):
        """기본 설정값 검증"""
        config = MoETransformerConfig()
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.d_model, 768)
        self.assertEqual(config.n_layers, 8)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.d_ff, 2048)
        self.assertEqual(config.num_experts, 4)
        self.assertEqual(config.k, 2)
        self.assertEqual(config.max_seq_len, 1024)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.eps, 1e-6)

    def test_custom_config(self):
        """사용자 정의 설정값 오버라이드 검증"""
        config = MoETransformerConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            num_experts=2,
            k=1,
            max_seq_len=256,
            dropout=0.2,
            eps=1e-5
        )
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.d_ff, 512)
        self.assertEqual(config.num_experts, 2)
        self.assertEqual(config.k, 1)
        self.assertEqual(config.max_seq_len, 256)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.eps, 1e-5)

    def test_model_initialization_with_config(self):
        """config 객체를 사용한 모델 생성 검증"""
        config = MoETransformerConfig(n_layers=2, d_model=128, n_heads=2)
        model = MoETransformer(config)
        self.assertEqual(model.n_layers, 2)
        self.assertEqual(model.d_model, 128)
        self.assertEqual(model.n_heads, 2)
        self.assertEqual(model.config.n_layers, 2)

    def test_model_initialization_backward_compatibility(self):
        """개별 인자(kwargs)를 직접 전달하는 방식의 하위 호환성 검증"""
        model = MoETransformer(
            vocab_size=20000,
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
            num_experts=6,
            k=3,
            max_seq_len=512
        )
        self.assertEqual(model.vocab_size, 20000)
        self.assertEqual(model.d_model, 256)
        self.assertEqual(model.n_layers, 4)
        self.assertEqual(model.config.num_experts, 6)
        self.assertEqual(model.config.k, 3)

if __name__ == '__main__':
    unittest.main()
