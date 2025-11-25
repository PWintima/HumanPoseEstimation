# Pose Recognition System - User Guide

## ✅ What Was Fixed and Added

### 1. **Fixed Errors**
- ✅ Fixed indentation errors in `PoseEstimation.py`
- ✅ Fixed syntax errors in `integrated_pose_system.py`
- ✅ All code now compiles without errors

### 2. **Added Pose Recognition**
- ✅ Created `PoseClassifier` class that analyzes keypoint positions
- ✅ Real-time pose identification and labeling
- ✅ Visual pose name display on screen
- ✅ Enhanced skeleton visualization with thicker lines

### 3. **Pose Detection Capabilities**

The system can now identify the following poses:

1. **T-Pose** - Arms extended horizontally
2. **Arms Raised** - Both arms raised above shoulders
3. **Arms Raised Up** - Both arms straight up
4. **Hands on Hips** - Wrists positioned near hips
5. **Right Arm Raised** - Only right arm raised
6. **Left Arm Raised** - Only left arm raised
7. **Sitting/Bent Knees** - Knees bent position
8. **Standing Straight** - Upright standing position
9. **Standing** - Default standing pose

---

## 🚀 How to Run

### Option 1: Using MediaPipe (Fast, No Training Required)
```bash
python3 integrated_pose_system.py --use_mediapipe
```

### Option 2: Using Trained Model
```bash
python3 integrated_pose_system.py --model_path outputs/checkpoint_best.pth
```

### Option 3: Hybrid Mode (Compare Both)
```bash
python3 integrated_pose_system.py --model_path outputs/checkpoint_best.pth --hybrid
```

---

## 📺 What You'll See

When the webcam opens, you'll see:

1. **Skeleton Overlay** - Green lines connecting your body joints
2. **Joint Points** - Circles marking each detected keypoint
3. **Pose Name** - Large text at the top showing what pose you're making
4. **FPS Counter** - Performance metrics
5. **Inference Time** - How fast the system is processing

---

## 🎯 How It Works

1. **Keypoint Detection**: The system detects 16 keypoints on your body:
   - Ankles, knees, hips
   - Pelvis, thorax, neck, head
   - Wrists, elbows, shoulders

2. **Pose Analysis**: The `PoseClassifier` analyzes:
   - Joint positions and angles
   - Distances between keypoints
   - Spatial relationships

3. **Pose Identification**: Based on geometric analysis, it identifies:
   - Arm positions (raised, horizontal, etc.)
   - Body posture (standing, sitting)
   - Gesture patterns

4. **Visual Display**: The identified pose name is displayed prominently on screen

---

## 🎮 Controls

- **q** - Quit the application
- **s** - Save current frame
- **m** - Toggle between methods (in hybrid mode)

---

## 🔧 Technical Details

### Pose Classification Logic

The system uses geometric analysis:
- **Angle Calculation**: Measures angles between joints (e.g., shoulder-elbow-wrist)
- **Distance Calculation**: Measures distances between keypoints
- **Position Analysis**: Compares vertical/horizontal positions
- **Visibility Check**: Only uses visible keypoints (confidence > threshold)

### Keypoint Mapping

The system uses MPII format with 16 joints:
- 0-5: Legs (ankles, knees, hips)
- 6-9: Torso and head
- 10-15: Arms (wrists, elbows, shoulders)

---

## 📝 Example Usage

```bash
# Start webcam with MediaPipe (recommended for testing)
python3 integrated_pose_system.py --use_mediapipe --camera 0

# The system will:
# 1. Open your webcam
# 2. Detect your pose in real-time
# 3. Display the pose name (e.g., "T-Pose", "Arms Raised")
# 4. Draw skeleton overlay
```

---

## ✨ Features

- ✅ Real-time pose recognition
- ✅ Visual feedback with skeleton overlay
- ✅ Pose name display
- ✅ Multiple pose detection
- ✅ Works with MediaPipe (fast) or trained models
- ✅ Performance metrics (FPS, inference time)

---

## 🎯 Goal Achieved

✅ **Map out keypoints on your body** - Done! Skeleton visualization shows all 16 keypoints

✅ **Estimate the pose you're making** - Done! Real-time pose classification and labeling

The system now identifies and displays what pose/gesture you're making in real-time!

