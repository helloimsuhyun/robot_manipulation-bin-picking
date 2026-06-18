<div align="center">

# Robot Manipulation - Bin Picking

**RGB-D Camera · 6D Pose Estimation · Robot Grasping · Force-Controlled Peg-in-Hole**

Intel RealSense D455 RGB-D 카메라와 RB5 협동로봇을 사용하여
물체의 6D pose를 추정하고, 로봇이 grasp 및 peg-in-hole insertion을 수행하는 프로젝트입니다.

<br/>

<img src="https://img.shields.io/badge/ROS2-Humble-blue">
<img src="https://img.shields.io/badge/Camera-RealSense%20D455-green">
<img src="https://img.shields.io/badge/Robot-RB5-orange">
<img src="https://img.shields.io/badge/6D%20Pose-FoundationPose-purple">
<img src="https://img.shields.io/badge/Detection-YOLOv8-red">

<br/>
<br/>

*학부 수업 **로봇공학실험** 프로젝트로 수행되었습니다.*

</div>

---

## 🎥 Demo

<div align="center">

<a href="https://youtu.be/lljcPwNJVUM?si=-M68miIis1vrQdBE">
  <img src="https://img.youtube.com/vi/lljcPwNJVUM/0.jpg" width="70%">
</a>

<br/>
<br/>

<img src="task_def_0.png" width="45%">
<img src="task_def_1.png" width="45%">

</div>

---

## 🧩 System Pipeline

```mermaid
flowchart TD
    A[RealSense D455 RGB-D Input] --> B[Hand-Eye Calibration]
    B --> C[YOLOv8 Segmentation]
    C --> D[FoundationPose 6D Pose Estimation]
    D --> E[Grasp Pose Generation]
    E --> F[Template Matching for Insert Target]
    F --> G[Force-Controlled Peg-in-Hole Insertion]
```

---

## 🛠️ Method

### 1. Hand-Eye Calibration

카메라에서 추정한 물체 pose를 로봇이 사용할 수 있도록,
**camera frame**과 **robot base frame** 사이의 변환 행렬을 계산합니다.

초기 위치에서 calibration board의 위치를 추정하고,
샘플링된 여러 지점에서 로봇 TCP가 보드 방향을 바라보도록 이동한 뒤 이미지를 촬영하여 hand-eye calibration을 수행합니다.

---

### 2. 6D Pose Estimation

YOLOv8 segmentation으로 물체 mask를 추출한 뒤,
RGB-D image, mask, CAD model을 FoundationPose에 입력하여 물체의 6D pose를 추정합니다.

추정된 pose는 hand-eye calibration 결과를 이용해
camera frame에서 robot base frame으로 변환됩니다.

---

### 3. Grasp Pose Generation

추정된 object pose를 parallel gripper가 접근 가능한 grasp pose로 변환합니다.

물체의 지면 대비 기울기와 그리퍼 접근 방향을 기준으로
grasp 가능 여부를 판단하고, 안정적인 grasp 방향을 선택합니다.

파지가 어려운 자세의 경우 물체를 빈 공간으로 옮겨 재배치한 뒤,
다른 면을 기준으로 다시 grasp를 시도합니다.

---

### 4. Insert Detection & Force Control

Insert target은 template matching을 이용해 검출합니다.

로봇은 검출된 위치로 이동한 뒤,
최종 삽입 단계에서는 force control을 사용해 peg-in-hole 작업을 수행합니다.

---

## ⚙️ Tech Stack

| Category           | Tool / Device                  |
| ------------------ | ------------------------------ |
| OS                 | Ubuntu 22.04                   |
| Middleware         | ROS2 Humble                    |
| Camera             | Intel RealSense D455           |
| Robot              | RB5 Collaborative Robot        |
| GPU                | NVIDIA GeForce RTX 4060 Laptop |
| Segmentation       | YOLOv8 Segmentation            |
| 6D Pose Estimation | FoundationPose                 |
| Insert Detection   | Template Matching              |
| Robot Control      | Force Control                  |

---

