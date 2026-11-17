# Gram detection pilot

서버의 기존 DINOv3 저장소와 가중치를 사용한다. Python 패키지: torch, numpy, Pillow, scikit-learn. VAE 사용 시 diffusers와 해당 의존성이 추가로 필요하다. 모델 저장소의 환경 요구사항도 설치해야 한다.

데이터 경로는 `DATA_ROOT/GENERATOR/.../real/...` 및 `DATA_ROOT/GENERATOR/.../fake/...` 형태로 가정한다. 실 데이터의 디렉터리와 라벨을 먼저 확인해야 한다. 각 라벨에서 독립적으로 표본을 뽑으므로 ImageNet 클래스 균형은 자동으로 맞추지 않는다. 중복 Real 이미지도 재사용하지 않는다.

```bash
python experiment.py --name pilot_01 --data-root /data/GenImage --repo /path/to/dinov3 --model MODEL_ENTRY --weights /path/to/weights.pth --samples 50 --size 512 --device cuda:0
python analyze.py --run results/pilot_01
```

VAE를 포함하려면 `--transforms noise,local_noise,wavelet,vae --vae-path /path/to/vae`를 지정한다. 이 VAE는 diffusers의 AutoencoderKL 형식이며 입력을 [-1, 1]로 변환하고 잠재 분포의 mode로 재구성한다. DINO 모델은 로컬 torch.hub의 `get_intermediate_layers(..., return_class_token=True, norm=True)` 인터페이스가 필요하다. 모델별 patch size에 맞는 입력 크기를 사용한다.

결과는 `results/이름/`에 저장하며 같은 이름은 덮어쓰지 않는다. Feature는 저장하지 않는다. 점수는 변형별로 기록하고 중단 시 부분 결과가 남지만 자동 재개는 지원하지 않는다. 실행 인자, 선택 목록, 이미지별 점수, Generator별 AUROC 및 Pearson 상관관계를 기록한다. AUROC는 Fake 점수가 높다는 방향으로 고정한다. ACC 및 그래프는 아직 구현하지 않았다.

계산 정의: 입력은 정사각형으로 직접 리사이즈한다. Patch token을 L2 정규화한 Gram의 대각을 제외하고 상삼각만 사용한다. Static은 하위 10% 평균, Gram 변화는 절댓값 평균 및 부호 평균이다. Near는 패치 격자 거리 <= 한 변의 1/4, Far는 > 한 변의 1/2이다. B–B는 변형된 픽셀을 포함하지 않는 패치 쌍의 절대 변화 평균이다. CKA는 이 정규화 Gram을 중심화한 유사도로 계산한다. CLS와 Patch baseline은 원본 Feature의 유클리드 변화량이다. Wavelet은 1단계 Haar의 고주파를 감쇠한다. Local noise는 이미지 면적 비율로 지정한 단일 사각형에 적용된다. 이 정의와 강도는 Pilot 설정이며 연구 목적에 맞게 검토해야 한다.

Pilot과 최종 데이터가 겹치지 않도록 manifest를 보관해 별도 분할을 구성해야 한다. Score 방향과 임계값을 평가 데이터에 맞춰 선택하지 않는다. 변형 난수는 seed, 이미지 절대 경로, 변형 이름으로 결정되므로 파일을 다른 경로로 옮기면 결과가 달라진다.
