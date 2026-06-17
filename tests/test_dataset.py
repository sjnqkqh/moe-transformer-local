import os
import unittest
import tempfile
import numpy as np
import torch
from train.utils import NumpyDataset


class TestNumpyDataset(unittest.TestCase):
    def setUp(self):
        # 임시 npy 파일 작성을 위한 경로 셋업
        self.test_dir = tempfile.TemporaryDirectory()
        self.npy_path = os.path.join(self.test_dir.name, "test_data.npy")

        # 임의의 토큰 스트림 데이터 생성: 형태 (100, 16) - 100개 샘플, 시퀀스 길이 16
        self.dummy_data = np.random.randint(0, 32000, size=(100, 16), dtype=np.int32)
        np.save(self.npy_path, self.dummy_data)

    def tearDown(self):
        # 임시 디렉토리 및 파일 자원 정리
        self.test_dir.cleanup()

    def test_dataset_length(self):
        """데이터셋 길이 검증"""
        dataset = NumpyDataset(self.npy_path)
        self.assertEqual(len(dataset), 100)

    def test_dataset_item_format(self):
        """데이터셋에서 아이템 인덱싱 시 올바른 텐서 튜플 (inputs, labels) 반환 검증"""
        dataset = NumpyDataset(self.npy_path)
        x, y = dataset[0]

        # 1. 텐서 형식인지 검사
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)

        # 2. 데이터 타입 및 모양 검사 (long 타입)
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(list(x.shape), [16])

        # 3. inputs와 labels의 동치성 확인 (Causal LM용이므로 동일해야 함)
        self.assertTrue(torch.equal(x, y))

        # 4. 실제 생성한 더미 값과 일치하는지 비교
        expected = torch.tensor(self.dummy_data[0], dtype=torch.long)
        self.assertTrue(torch.equal(x, expected))


if __name__ == "__main__":
    unittest.main()
