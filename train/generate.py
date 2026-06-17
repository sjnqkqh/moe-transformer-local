import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# 프로젝트 루트 경로를 모듈 검색 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dense_transformer import DenseTransformer
from model.config import DenseTransformerConfig


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    device="cpu",
):
    """
    주어진 프롬프트로부터 autoregressive 방식으로 다음 토큰들을 생성합니다.
    """
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    for _ in range(max_new_tokens):
        # 입력 길이가 컨텍스트 윈도우(max_seq_len)를 초과하는 경우 자르기
        if input_ids.shape[1] >= model.config.max_seq_len:
            input_ids = input_ids[:, -model.config.max_seq_len :]

        logits, _, _ = model(input_ids)
        next_logits = logits[:, -1, :]

        # 반복 페널티 적용 (Repetition Penalty)
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[1]):
                token = input_ids[0, i].item()
                if next_logits[0, token] < 0:
                    next_logits[0, token] *= repetition_penalty
                else:
                    next_logits[0, token] /= repetition_penalty

        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-K 필터링
        if top_k > 0:
            topk_vals, _ = torch.topk(next_logits, top_k)
            min_val = topk_vals[..., -1, None]
            next_logits[next_logits < min_val] = float("-inf")

        # Top-P (Nucleus) 필터링
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 누적 확률이 top_p를 넘는 토큰들을 마스킹 (첫 번째 토큰은 제외)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = False

            # 원래 인덱스로 매핑하여 마스킹 적용
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_idx, sorted_indices_to_remove
            )
            next_logits[indices_to_remove] = float("-inf")

        # 확률 분포 계산 및 샘플링
        if temperature == 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=False)


def chat_generate(model, tokenizer, user_message, **kwargs):
    """
    대화 포맷으로 감싸서 챗봇 답변을 생성합니다.
    """
    prompt = f"<user>{user_message}<sep><assistant>"
    output = generate(model, tokenizer, prompt, **kwargs)

    # 생성된 전체 텍스트에서 <assistant> 이후의 부분만 추출하고 후처리
    if "<assistant>" in output:
        reply = output.split("<assistant>")[-1]
        for stop in ["<sep>", "</s>", "<user>"]:
            reply = reply.split(stop)[0]
        return reply.strip()
    return output.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="체크포인트 파일 (.pt) 경로"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="tokenizer/korean_output",
        help="토크나이저 디렉토리 경로",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="생성할 시작 프롬프트 또는 유저 메시지",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="대화 인터페이스 사용 여부 (<user>... 형태로 포맷팅)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.2, help="반복 토큰 페널티"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="실행 디바이스 (cuda, mps, cpu 등)"
    )
    args = parser.parse_args()

    # 디바이스 자동 감지
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # 토크나이저 로드
    print(f"Loading tokenizer from {args.tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    # 모델 설정 및 생성
    vocab_size = len(tokenizer)
    config = DenseTransformerConfig(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=8,
        d_ff=3072,
        max_seq_len=512,
        dropout=0.0,
    )
    model = DenseTransformer(config)

    # 체크포인트 로드
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # module. 접두사 제거 로직
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    # 텍스트 생성
    if args.chat:
        print("\n--- Chat Mode ---")
        print(f"User: {args.prompt}")
        reply = chat_generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        print(f"Assistant: {reply}")
    else:
        print("\n--- Standard Generation Mode ---")
        print(f"Prompt: {args.prompt}")
        output = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        print(f"Generated text: {output}")


if __name__ == "__main__":
    main()
