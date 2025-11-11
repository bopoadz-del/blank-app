#!/usr/bin/env python3
"""
Test script to demonstrate improved camera error handling
This simulates camera failure scenarios to show the improved diagnostics
"""

import sys
import os
from pathlib import Path

# Add jetson-rtdetr to path
jetson_path = Path(__file__).parent / 'jetson-rtdetr'
sys.path.insert(0, str(jetson_path))

# Import the functions we want to test
sys.path.insert(0, str(jetson_path / 'examples'))

import cv2
import time

# Copy the functions from video_inference for testing
def print_camera_diagnostics(camera_index):
    """Print diagnostic information for camera troubleshooting"""
    print("\n" + "="*60)
    print("CAMERA DIAGNOSTICS")
    print("="*60)
    
    # Check available video devices on Linux
    if sys.platform.startswith('linux'):
        print("\nAvailable video devices:")
        video_devices = []
        for i in range(10):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                video_devices.append(device_path)
                print(f"  ✓ {device_path} exists")
        
        if not video_devices:
            print("  ✗ No video devices found in /dev/")
    
    # Check OpenCV build info
    print(f"\nOpenCV version: {cv2.__version__}")
    
    print("\nTroubleshooting suggestions:")
    print("  1. Check if camera is physically connected")
    print("  2. Verify camera permissions: ls -l /dev/video*")
    print("  3. Add user to video group: sudo usermod -a -G video $USER")
    print("  4. Check if camera is in use: sudo lsof /dev/video*")
    print("  5. Try different camera indices (0, 1, 2, etc.)")
    print("  6. For USB cameras, try unplugging and reconnecting")
    print("  7. Check dmesg for hardware errors: dmesg | grep -i video")
    print("="*60 + "\n")


def open_video_source(source, max_retries=3, retry_delay=1.0):
    """
    Open video source with retry mechanism and detailed error reporting
    
    Args:
        source: Video source (camera index as int or string, or video file path)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        cv2.VideoCapture object if successful, None otherwise
    """
    is_camera = False
    source_desc = source
    
    # Determine if source is a camera index or file
    if isinstance(source, str) and source.isdigit():
        source = int(source)
        is_camera = True
        source_desc = f"camera {source}"
    elif isinstance(source, int):
        is_camera = True
        source_desc = f"camera {source}"
    else:
        source_desc = f"video file '{source}'"
    
    print(f"\nAttempting to open {source_desc}...")
    
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"\nRetry attempt {attempt}/{max_retries}...")
            time.sleep(retry_delay)
        
        # Try to open video source
        cap = cv2.VideoCapture(source)
        
        # Check if successfully opened
        if cap.isOpened():
            # Verify we can actually read a frame
            ret, frame = cap.read()
            if ret:
                # Reset to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"✓ Successfully opened {source_desc}")
                return cap
            else:
                print(f"✗ Opened {source_desc} but cannot read frames")
                cap.release()
        else:
            print(f"✗ Failed to open {source_desc} (attempt {attempt}/{max_retries})")
    
    # All retries failed
    print(f"\n{'='*60}")
    print(f"ERROR: Failed to open {source_desc} after {max_retries} attempts")
    print(f"{'='*60}")
    
    if is_camera:
        print_camera_diagnostics(source)
    else:
        print("\nPossible issues with video file:")
        print(f"  1. File does not exist: {source}")
        print(f"  2. File format not supported by OpenCV")
        print(f"  3. File is corrupted or incomplete")
        print(f"  4. Missing codecs for this video format")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .webm")
        
        if not os.path.exists(source):
            print(f"\n✗ File not found: {source}")
        else:
            print(f"\n✓ File exists: {source}")
            file_size = os.path.getsize(source)
            print(f"  File size: {file_size:,} bytes")
    
    return None


def test_camera_failure():
    """Test camera failure scenarios"""
    print("="*70)
    print("TESTING CAMERA DIAGNOSTICS - Simulating Camera Failure")
    print("="*70)
    
    # Test 1: Non-existent camera index
    print("\n\nTest 1: Attempting to open non-existent camera (index 99)")
    print("-"*70)
    cap = open_video_source(99, max_retries=2, retry_delay=0.5)
    if cap is None:
        print("✓ Test passed: Properly handled non-existent camera with diagnostics")
    else:
        print("✗ Test failed: Should not have opened camera")
        cap.release()
    
    # Test 2: Non-existent video file
    print("\n\nTest 2: Attempting to open non-existent video file")
    print("-"*70)
    cap = open_video_source("/tmp/nonexistent_video.mp4", max_retries=2, retry_delay=0.5)
    if cap is None:
        print("✓ Test passed: Properly handled non-existent file with diagnostics")
    else:
        print("✗ Test failed: Should not have opened file")
        cap.release()
    
    print("\n\n" + "="*70)
    print("TESTS COMPLETED")
    print("="*70)
    print("\nThe improved error handling provides:")
    print("  ✓ Detailed diagnostic information")
    print("  ✓ Retry mechanism for transient failures")
    print("  ✓ Helpful troubleshooting suggestions")
    print("  ✓ System information for debugging")
    print("  ✓ Clear error messages with actionable steps")
    print("="*70)


if __name__ == "__main__":
    test_camera_failure()
