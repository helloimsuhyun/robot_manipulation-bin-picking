# RGB-D 카메라 기반 실시간 6D Pose 추정

Intel RealSense D455 카메라로 산업용 부품 3종(cylinder / hole / cross)을 실시간 인식하고,
**FoundationPose**를 이용해 각 물체의 정확한 **6D Pose (위치 + 자세)**를 ROS2로 publish하는 시스템입니다.

---

## 파이프라인

```
RealSense D455 (RGB + Depth)
    → YOLO-seg (객체 감지 + segmentation mask)
    → Depth 기반 우선순위 정렬 (가까운 물체 선택)
    → FoundationPose (CAD mesh + mask + RGBD → 6D pose)
    → ROS2 publish (/object_poses, /insert_poses)
```

FoundationPose 미설치 또는 실패 시 **Depth PCA fallback**으로 자동 전환됩니다.

---

## 환경 요구사항

- Ubuntu 22.04
- Python 3.10
- ROS2 Humble
- NVIDIA GPU (VRAM 8GB 이상 권장) + CUDA Toolkit
- Intel RealSense D455

---

## 설치

### 1. Python 환경 준비

```bash
cd ~/rgbd_camera
source venv/bin/activate
pip install ultralytics pyrealsense2 opencv-python numpy scipy trimesh
```

### 2. FoundationPose 설치

```bash
# CUDA Toolkit 설치 확인
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 설치 스크립트 실행
bash setup_foundationpose.sh
```

스크립트가 자동으로 수행하는 작업:
1. `~/FoundationPose` 클론
2. PyTorch, nvdiffrast, pytorch3d 설치
3. FoundationPose 내부 의존성 빌드
4. 사전학습 가중치 다운로드 (`~/FoundationPose/weights/`)

> **주의**: PyTorch CUDA 버전과 nvcc CUDA 버전이 일치해야 합니다.
> 불일치 시 스크립트가 오류 메시지와 함께 종료됩니다.

### 3. CAD 메쉬 배치

```
rgbd_camera/CAD/
├── cross.stl
├── cylinder.stl
└── hole.stl
```

CAD 파일이 **mm 단위**이면 기본값(`CAD_MESH_SCALE=0.001`) 그대로 사용.
**m 단위**이면:

```bash
export CAD_MESH_SCALE=1.0
```

---

## YOLO 모델 학습 (이미 학습된 경우 생략)

```bash
# 1. 데이터 수집 (RealSense 연결 후)
python data_collector.py
# r키: 0.5초 간격 자동 촬영 / s키: 수동 1장 저장 / ESC: 종료

# 2. Roboflow에서 polygon segmentation 라벨링 후 데이터셋 export

# 3. 학습
python train_yolo.py
```

학습 결과는 `runs/segment/train/weights/best.pt`에 저장됩니다.

---

## 실행

### 실행 전 확인

```bash
cd ~/rgbd_camera
source venv/bin/activate
source /opt/ros/humble/setup.bash

# GPU / CUDA 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# RealSense 연결 확인
rs-enumerate-devices

# FoundationPose 소스 import 경로
export PYTHONPATH=$HOME/FoundationPose:$PYTHONPATH
export MPLCONFIGDIR=/tmp/matplotlib
```

`torch.cuda.is_available()`가 `True`이고 RealSense D455가 인식되어야 메인 노드가 정상 실행됩니다.

### FoundationPose 6D Pose 추정 (메인)

```bash
python foundation_pose_node.py
```

- RealSense에서 RGB+Depth 수신 (10Hz)
- YOLO로 물체 detect → 가장 가까운 물체 1개 선택
- FoundationPose로 6D pose 계산
- ROS2 토픽으로 결과 publish
- 화면에 3D 좌표축 오버레이 시각화 (ESC로 종료)

### Depth PCA 방식 (FoundationPose 없을 때)

```bash
cd ~/rgbd_camera
source venv/bin/activate
source /opt/ros/humble/setup.bash
python detect_3d_pose.py
```

FoundationPose 없이 point cloud + PCA로 위치/방향 추출. 정확도는 낮지만 의존성 없이 동작.

---

## ROS2 토픽

| 토픽 | 메시지 타입 | 내용 |
|------|------------|------|
| `/object_poses` | `std_msgs/String` | object 모드 결과 JSON |
| `/insert_poses` | `std_msgs/String` | insert 모드 결과 JSON |
| `/object_pose_stamped` | `geometry_msgs/PoseStamped` | object 모드 6D pose |
| `/insert_pose_stamped` | `geometry_msgs/PoseStamped` | insert 모드 6D pose |
| `/detect_mode` | `std_msgs/String` | 모드 전환 (subscribe) |

### 모드 전환

```bash
# object 모드 (cross / cylinder / hole 감지)
ros2 topic pub --once /detect_mode std_msgs/msg/String "data: 'object'"

# insert 모드 (cross_insert / cylinder_insert / hole_insert 감지)
ros2 topic pub --once /detect_mode std_msgs/msg/String "data: 'insert'"
```

### JSON 출력 예시

```json
{
  "mode": "object",
  "target": {
    "class": "cylinder",
    "confidence": 0.923,
    "position": {"x": 0.012, "y": -0.045, "z": 0.382},
    "orientation": {"x": 0.01, "y": 0.02, "z": 0.0, "w": 0.9997},
    "pose_matrix": [[...], [...], [...], [...]],
    "priority": {
      "selected": true,
      "reason": "nearest_depth",
      "depth_median_m": 0.382,
      "detected_count": 2
    }
  },
  "detected_count": 2
}
```

---

## 파일 구조

```
rgbd_camera/
├── foundation_pose_node.py   # FoundationPose 6D pose 추정 메인 노드
├── detect_3d_pose.py         # Depth PCA 방식 (fallback / 구버전)
├── setup_foundationpose.sh   # FoundationPose 설치 스크립트
├── data_collector.py         # RealSense 이미지 수집 도구
├── train_yolo.py             # 데이터셋 병합 + YOLOv8 학습
├── test_3d_pose.py           # pose 추정 테스트
├── pose_publisher.py         # pose publish 유틸
├── analyze_results.py        # 학습 결과 시각화
├── CAD/                      # CAD 메쉬 (cross.stl, cylinder.stl, hole.stl)
├── runs/segment/
│   ├── train/weights/best.pt       # object 모드 YOLO 모델
│   └── insert_seg/weights/best.pt  # insert 모드 YOLO 모델
├── cylinder/                 # cylinder 클래스 데이터셋
├── hole/                     # hole 클래스 데이터셋
├── cross/                    # cross 클래스 데이터셋
└── object/                   # 병합 데이터셋
```

---

## 학습 결과

| Class    | mAP50 (Box) | mAP50 (Mask) | Precision | Recall |
|----------|:-----------:|:------------:|:---------:|:------:|
| cylinder |   0.9950    |    0.9950    |   0.9839  | 1.0000 |
| hole     |   0.8473    |    0.8394    |   0.9458  | 0.8571 |
| cross    |   0.9527    |    0.9527    |   0.9062  | 0.9666 |
| **mean** | **0.9317**  |  **0.9290**  | **0.9453**|**0.9412**|

---

## 개발 환경

- Python 3.10
- Intel RealSense D455
- Ultralytics YOLOv8n-seg
- FoundationPose (NVlabs)
- ROS2 Humble
- CUDA 12.x / NVIDIA RTX 4060 Laptop (8GB)
