"""
Simple MediaPipe Pose Estimation Script

This is a basic example script demonstrating MediaPipe pose estimation:
- Static image processing
- Video file processing
- Real-time webcam processing (commented out)

Originally created by MediaPipe and modified by Augmented Startups.
This script provides a simple starting point for pose estimation using
MediaPipe without requiring model training.

Usage:
    - For static images: Set the image path in the code
    - For videos: Set the video path in cv2.VideoCapture()
    - For webcam: Uncomment cap = cv2.VideoCapture(0)
"""

# Created by MediaPipe
# Modified by Augmented Startups 2021
# Pose-Estimation in 5 Minutes
# Watch 5 Minute Tutorial at www.augmentedstartups.info/YouTube

import cv2
import mediapipe as mp
import time
import os

# Initialize MediaPipe drawing utilities and pose solution
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# ============================================================================
# STATIC IMAGE PROCESSING
# ============================================================================
# Process a single static image and save the annotated result
with mp_pose.Pose(
    static_image_mode=True,  # Process as static image (not video)
    model_complexity=2,  # Use highest complexity model (0, 1, or 2)
    min_detection_confidence=0.5) as pose:  # Minimum confidence for detection
    
    # Load input image (change path as needed)
    image = cv2.imread('4.jpg')  # Insert your Image Here
    
    if image is not None:
        image_height, image_width, _ = image.shape
        
        # Convert BGR to RGB (MediaPipe expects RGB format)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw pose landmarks on a copy of the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
        
        # Save annotated image
        cv2.imwrite(r'4.png', annotated_image)
        print("Static image processed and saved.")

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
# Process video file frame by frame with pose estimation
# For webcam input, uncomment: cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("1.mp4")  # For Video input

prevTime = 0  # Track previous frame time for FPS calculation

with mp_pose.Pose(
    min_detection_confidence=0.5,  # Minimum confidence for initial detection
    min_tracking_confidence=0.5) as pose:  # Minimum confidence for tracking
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Convert BGR image to RGB (MediaPipe expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Mark image as not writeable to pass by reference (performance optimization)
        image.flags.writeable = False
        
        # Process image with MediaPipe pose estimation
        results = pose.process(image)

        # Mark image as writeable again for drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks and connections on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Calculate and display FPS
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), 
                   cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        
        # Display the annotated frame
        cv2.imshow('BlazePose', image)
        
        # Press ESC (27) to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Learn more AI in Computer Vision by Enrolling in our AI_CV Nano Degree:
# https://bit.ly/AugmentedAICVPRO