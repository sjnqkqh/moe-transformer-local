import unittest
import torch
from train.train import RoutingProfiler

class TestVectorizedRoutingCounting(unittest.TestCase):
    def test_bincount_matches_loop_counting(self):
        """bincount() 벡터 연산 결과와 기존 파이썬 for 루프 집계 결과가 완전히 동일한지 비교 검증"""
        # 임의의 전문가 할당 인덱스 생성: (Batch_Size * Seq_Len = 256, k = 2)
        # 전문가 개수는 4개 (인덱스: 0 ~ 3)
        torch.manual_seed(1004)
        all_indices = torch.randint(0, 4, (256, 2))
        
        # 1. 기존 루프 방식 집계
        expert_counts_loop = torch.zeros(4)
        for idx in all_indices.view(-1):
            if idx.item() < 4:
                expert_counts_loop[idx.item()] += 1
                
        # 2. bincount 벡터 연산 방식 집계
        expert_counts_bincount = torch.bincount(all_indices.view(-1), minlength=4).float()
        
        # 두 연산 결과의 값이 완전히 같은지 확인
        self.assertTrue(torch.equal(expert_counts_loop, expert_counts_bincount))
        self.assertEqual(expert_counts_bincount.sum().item(), 512)

    def test_routing_profiler_integration(self):
        """RoutingProfiler가 bincount() 연산을 통해 올바른 형태 및 Shannon Entropy를 반환하는지 테스트"""
        profiler = RoutingProfiler(num_experts=4)
        
        # 가상의 라우팅 결과 추가 (S=10, k=2)
        dummy_top_k_indices1 = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3], [0, 1]])
        dummy_top_k_indices2 = torch.tensor([[2, 3], [0, 3], [1, 2], [1, 0], [2, 3]])
        
        # hook_fn에 넘겨지는 형태로 profiler selections 수집 시뮬레이션
        # output[1] = top_k_indices
        profiler.selections.append(dummy_top_k_indices1)
        profiler.selections.append(dummy_top_k_indices2)
        
        usage, entropy = profiler.get_metrics()
        
        # 1. usage 비율 배열 검증
        self.assertEqual(len(usage), 4)
        self.assertAlmostEqual(sum(usage), 1.0)  # 모든 전문가 선택 빈도의 총합 비율은 1.0 (100%) 이 나옴
        
        # 2. entropy가 양수이며 타당한 범위 내에 있는지 확인
        # Shannon Entropy는 균등 분포일 때 최대값 log(4) = 1.3863
        self.assertTrue(0.0 < entropy <= 1.3863)

if __name__ == '__main__':
    unittest.main()
