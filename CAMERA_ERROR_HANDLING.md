# Camera Error Handling Improvements

## Overview
This document describes the improvements made to camera/video capture error handling in the RT-DETR video inference system.

## Problem Statement
The original implementation had minimal error handling when camera/video sources failed to open. Users would see a simple error message and the script would exit, leaving them without guidance on how to troubleshoot the issue.

## Solution

### 1. Retry Mechanism
- **Feature**: Automatic retry with configurable attempts
- **Default**: 3 retry attempts with 1-second delay
- **Benefit**: Handles transient failures (e.g., camera initializing)

```python
cap = open_video_source(args.input, max_retries=3, retry_delay=1.0)
```

### 2. Detailed Diagnostics

#### For Camera Failures:
- Lists available video devices on Linux (`/dev/video*`)
- Shows OpenCV version and build information
- Provides 7 specific troubleshooting steps:
  1. Check physical connection
  2. Verify permissions (`ls -l /dev/video*`)
  3. Add user to video group
  4. Check if camera is in use
  5. Try different camera indices
  6. Reconnect USB cameras
  7. Check system logs (`dmesg`)

#### For Video File Failures:
- Checks if file exists
- Shows file size if available
- Lists supported video formats
- Identifies common issues:
  - Missing file
  - Unsupported format
  - Corrupted file
  - Missing codecs

### 3. Improved User Feedback
- **Progress indicators**: Shows each retry attempt
- **Success confirmation**: Clear message when source opens
- **Failure diagnostics**: Comprehensive error information
- **Exit codes**: Returns proper exit codes for automation

### 4. Frame Verification
The improved implementation doesn't just check if VideoCapture opens, but also verifies that frames can actually be read:

```python
if cap.isOpened():
    # Verify we can actually read a frame
    ret, frame = cap.read()
    if ret:
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap
```

## Code Changes

### Modified Files
- `jetson-rtdetr/examples/video_inference.py`

### New Functions
1. **`print_camera_diagnostics(camera_index)`**
   - Prints comprehensive diagnostic information
   - Platform-specific checks (Linux video devices)
   - Troubleshooting suggestions

2. **`open_video_source(source, max_retries, retry_delay)`**
   - Handles video source opening with retry logic
   - Distinguishes between cameras and video files
   - Provides detailed error messages

### Enhanced main() Function
- Uses new `open_video_source()` function
- Proper exit codes
- Enhanced video property display

## Example Output

### Before (Original Code):
```
Error: Failed to open video source: 0
```

### After (Improved Code):
```
Attempting to open camera 0...
✗ Failed to open camera 0 (attempt 1/3)

Retry attempt 2/3...
✗ Failed to open camera 0 (attempt 2/3)

Retry attempt 3/3...
✗ Failed to open camera 0 (attempt 3/3)

============================================================
ERROR: Failed to open camera 0 after 3 attempts
============================================================

============================================================
CAMERA DIAGNOSTICS
============================================================

Available video devices:
  ✗ No video devices found in /dev/

OpenCV version: 4.8.1
Video I/O backends: [detailed build info]

Troubleshooting suggestions:
  1. Check if camera is physically connected
  2. Verify camera permissions: ls -l /dev/video*
  3. Add user to video group: sudo usermod -a -G video $USER
  4. Check if camera is in use: sudo lsof /dev/video*
  5. Try different camera indices (0, 1, 2, etc.)
  6. For USB cameras, try unplugging and reconnecting
  7. Check dmesg for hardware errors: dmesg | grep -i video
============================================================

✗ Cannot proceed without valid video source
Please check the diagnostics above and try again.
```

## Testing

### Unit Tests
Created comprehensive unit tests in `test_camera_improvements.py`:
- Camera index detection
- Retry logic verification
- Error message validation
- Diagnostic completeness
- User experience improvements

All 11 tests pass successfully.

## Benefits

1. **Reduced Support Burden**: Users get actionable troubleshooting steps
2. **Better Reliability**: Retry mechanism handles transient failures
3. **Easier Debugging**: Comprehensive diagnostics save time
4. **Professional UX**: Clear, informative error messages
5. **Platform Awareness**: Platform-specific diagnostics (Linux)

## Backward Compatibility
- All changes are backward compatible
- Same command-line interface
- Original functionality preserved
- Only enhanced error handling added

## Future Enhancements
Potential future improvements:
- Support for RTSP streams
- Automatic camera discovery
- Performance metrics for camera initialization
- Configuration file for retry settings
- Integration with logging frameworks
