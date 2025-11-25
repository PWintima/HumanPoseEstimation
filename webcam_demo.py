"""
Simple webcam demo wrapper for PoseInference

Usage example:
    powershell> & .venv\Scripts\python.exe webcam_demo.py --model_path outputs/training_20251124_222336/checkpoint.pth --camera 0

This script opens the webcam, runs pose estimation in real-time, and displays the output.
"""

import argparse
import torch
from inference import PoseInference


def main():
    parser = argparse.ArgumentParser(description="Webcam demo for pose estimation")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", default="simplebaseline", choices=["hrnet", "simplebaseline"],
                        help="Model type to use")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold for drawing")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on")

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Initialize inference
    inference = PoseInference(model_path=args.model_path, model_type=args.model_type, device=device)

    # Start webcam processing
    inference.process_webcam(confidence_threshold=args.confidence, camera_index=args.camera)


if __name__ == '__main__':
    main()
