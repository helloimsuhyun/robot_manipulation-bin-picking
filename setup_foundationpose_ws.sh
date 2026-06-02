#!/usr/bin/env bash
set -e

ENV_NAME="cource"
WS_DIR="/home/choisuhyun/course/robot_manipulation-bin-picking"
FP_DIR="$WS_DIR/FoundationPose"

echo "============================================================"
echo "[0] Conda env / ROS2 환경 확인"
echo "============================================================"

source /opt/ros/humble/setup.bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$FP_DIR:${PYTHONPATH:-}"

# PyTorch3D/nvdiffrast 빌드 시 conda 가짜 compiler 경로 문제 방지
unset NVCC_PREPEND_FLAGS || true
unset NVCC_APPEND_FLAGS || true
unset NVCC_PREPEND_FLAGS_BACKUP || true
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

echo "CONDA_PREFIX=$CONDA_PREFIX"
which python
python --version
which pip
pip --version
echo "CUDA_HOME=$CUDA_HOME"
which nvcc
nvcc --version

echo ""
echo "============================================================"
echo "[1] PyTorch / CUDA 확인"
echo "============================================================"

python - <<'PY'
import re
import subprocess
import sys
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

out = subprocess.check_output(["nvcc", "--version"], text=True)
m = re.search(r"release\s+(\d+\.\d+)", out)
nvcc_cuda = m.group(1) if m else None
print("nvcc cuda:", nvcc_cuda)

if torch.version.cuda is None:
    sys.exit("ERROR: torch가 CUDA build가 아님")

torch_mm = ".".join(torch.version.cuda.split(".")[:2])
if torch_mm != nvcc_cuda:
    sys.exit(f"ERROR: torch CUDA({torch.version.cuda})와 nvcc CUDA({nvcc_cuda})가 다름")

if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() == False")

print("OK: torch CUDA == nvcc CUDA and GPU available")
PY

echo ""
echo "============================================================"
echo "[2] ROS2 rclpy 확인"
echo "============================================================"

python -c "import rclpy; print('rclpy OK')"

echo ""
echo "============================================================"
echo "[3] FoundationPose repo 확인"
echo "============================================================"

if [ ! -d "$FP_DIR" ]; then
    echo "FoundationPose repo가 없어서 clone합니다: $FP_DIR"
    git clone https://github.com/NVlabs/FoundationPose.git "$FP_DIR"
else
    echo "이미 존재: $FP_DIR"
fi

# colcon이 FoundationPose 내부 setup.py를 ROS package로 오인하지 않게 막기
touch "$FP_DIR/COLCON_IGNORE"

echo ""
echo "============================================================"
echo "[4] Python dependency 설치"
echo "============================================================"

pip install -U pip setuptools wheel
pip install trimesh pillow scipy scikit-learn open3d
pip install einops transformers
pip install PyOpenGL PyOpenGL_accelerate
pip install pyyaml imageio joblib ruamel.yaml
pip install fastapi uvicorn python-multipart requests
pip install transformations kornia omegaconf

echo ""
echo "============================================================"
echo "[5] PyTorch3D 확인/설치"
echo "============================================================"

if python -c "import pytorch3d" >/dev/null 2>&1; then
    echo "pytorch3d already installed"
else
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="8.9"
    export MAX_JOBS=2

    pip install --no-build-isolation --no-cache-dir \
      "git+https://github.com/facebookresearch/pytorch3d.git"
fi

echo ""
echo "============================================================"
echo "[6] nvdiffrast 확인/설치"
echo "============================================================"

if python -c "import nvdiffrast.torch" >/dev/null 2>&1; then
    echo "nvdiffrast already installed"
else
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="8.9"
    export MAX_JOBS=2

    pip install --no-build-isolation --no-cache-dir \
      "git+https://github.com/NVlabs/nvdiffrast.git"
fi

echo ""
echo "============================================================"
echo "[7] FoundationPose requirements 설치"
echo "============================================================"

cd "$FP_DIR"
pip install -r requirements.txt || true

echo ""
echo "============================================================"
echo "[8] FoundationPose weights 다운로드"
echo "============================================================"

WEIGHTS_DIR="$FP_DIR/weights"
mkdir -p "$WEIGHTS_DIR"

NEED_WEIGHTS=0
for f in \
  "$WEIGHTS_DIR/2023-10-28-18-33-37/config.yml" \
  "$WEIGHTS_DIR/2023-10-28-18-33-37/model_best.pth" \
  "$WEIGHTS_DIR/2024-01-11-20-02-45/config.yml" \
  "$WEIGHTS_DIR/2024-01-11-20-02-45/model_best.pth"
do
    if [ ! -f "$f" ]; then
        NEED_WEIGHTS=1
    fi
done

if [ "$NEED_WEIGHTS" -eq 0 ]; then
    echo "weights already exist"
else
    echo "weights missing. Hugging Face mirror로 다운로드 시도"
    pip install -U huggingface_hub

    cd "$FP_DIR"
    python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="gpue/foundationpose-weights",
    repo_type="model",
    local_dir="weights",
    local_dir_use_symlinks=False,
)
PY
fi

echo ""
echo "============================================================"
echo "[9] weights 확인"
echo "============================================================"

ls "$WEIGHTS_DIR/2023-10-28-18-33-37/config.yml"
ls "$WEIGHTS_DIR/2023-10-28-18-33-37/model_best.pth"
ls "$WEIGHTS_DIR/2024-01-11-20-02-45/config.yml"
ls "$WEIGHTS_DIR/2024-01-11-20-02-45/model_best.pth"

echo ""
echo "============================================================"
echo "[10] FoundationPose import / model init 확인"
echo "============================================================"

cd "$WS_DIR"
export PYTHONPATH="$FP_DIR:${PYTHONPATH:-}"

python - <<'PY'
import sys
sys.path.insert(0, "/home/choisuhyun/course/robot_manipulation-bin-picking/FoundationPose")

import torch
import nvdiffrast.torch as dr
import pytorch3d
import trimesh
import kornia
import transformations
import estimater
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("imports OK")

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()

print("FoundationPose model init OK")
PY

echo ""
echo "============================================================"
echo "[11] RealSense 연결 확인"
echo "============================================================"

python - <<'PY'
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    print("RealSense num devices:", len(devices))
    for d in devices:
        print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
    if len(devices) == 0:
        print("WARN: RealSense가 연결되어 있지 않음. launch 실행 시 No device connected가 날 수 있음.")
except Exception as e:
    print("WARN: RealSense check failed:", e)
PY

echo ""
echo "============================================================"
echo "설치/검증 완료"
echo "============================================================"
echo ""
echo "다음 실행:"
echo "  cd $WS_DIR"
echo "  source /opt/ros/humble/setup.bash"
echo "  conda activate $ENV_NAME"
echo "  export CUDA_HOME=\"\$CONDA_PREFIX\""
echo "  export PATH=\"\$CONDA_PREFIX/bin:\$CUDA_HOME/bin:\$PATH\""
echo "  export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}\""
echo "  export PYTHONPATH=\"$FP_DIR:\${PYTHONPATH:-}\""
echo "  source install/setup.bash"
echo "  ros2 launch sixd_pose_vision mixed_pose_vision.launch.py"
echo ""
