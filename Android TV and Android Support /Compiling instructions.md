# Compiling Instructions for those who don't like prebuilt .apk files.
# Spatial Audio System for Android TV

## Overview

A complete implementation of the unified spatial audio equation with computer vision-enhanced motion tracking for Android TV. This system creates immersive 360° audio with emergent elevation from horizontal speaker arrays using advanced linear algebra and signal processing.

## Core Equation

The complete system implements:

```
Ψ_s(t) = Σ_r Σ_b ∫ G_s(θ_r, φ_r) · H(ω_b, θ_r, φ_r, d_r, α, r) · 
         B_b(ω_b) · D_s(t - τ_r) · C(E_a, E_m, λ) · M(x, v, t) · 
         S(t, ω) · clip[tanh(·)] dω
```

## Key Features

### 1. **VBAP with cos² Law** (`VBAPProcessor.kt`)
- Energy-preserving vector base amplitude panning
- Emergent 360° elevation from horizontal speakers
- Second-order spherical harmonics
- Triangle-based speaker coverage

### 2. **Room Acoustics** (`TransferFunctionProcessor.kt`)
- Distance attenuation with near-field compensation
- Frequency-dependent air absorption (ISO 9613-1)
- Multi-order reflections with energy loss
- Directivity patterns

### 3. **Multi-Band Processing** (`MultiBandFilterBank.kt`)
- 7 octave bands: 62.5 - 4000 Hz
- Butterworth filters (Q=0.707)
- Implicit sinc reconstruction
- Perfect frequency domain summation

### 4. **Delay Lines** (`CircularBuffer.kt`)
- Fractional delay with linear interpolation
- 96kHz sample rate support
- Distance-based time alignment

### 5. **Computer Vision** (`CVMotionTracker.kt`)
- Real-time motion detection
- Object tracking with velocity estimation
- Audio-motion correlation (lag search ±5 frames)
- Screen-to-3D audio space mapping

### 6. **Spectral Processing** (`FFTProcessor.kt`)
- 2048-point FFT
- Hann windowing
- Spectral enhancement
- Cooley-Tukey radix-2 algorithm

### 7. **Audio Engine** (`SpatialAudioEngine.kt`)
- 96kHz @ 32-bit float processing
- 7.1 surround output
- Real-time processing pipeline
- Automatic surround detection

## Project Structure

```
app/src/main/java/com/spatialaudiosystem/
├── MainActivity.kt                    # Main UI and lifecycle
├── SpatialAudioEngine.kt             # Core audio processing
├── CVMotionTracker.kt                # Computer vision system
├── VBAPProcessor.kt                  # VBAP implementation
├── TransferFunctionProcessor.kt      # Room acoustics
├── MultiBandFilterBank.kt            # Filter bank processing
├── CircularBuffer.kt                 # Delay line implementation
└── FFTProcessor.kt                   # Spectral analysis
```

## Build Requirements

- **Minimum SDK**: 24 (Android 7.0)
- **Target SDK**: 34 (Android 14)
- **Kotlin**: 1.9+
- **Compose**: Latest BOM

### Dependencies

```kotlin
// Core
implementation("androidx.core:core-ktx:1.12.0")
implementation("androidx.compose.material3:material3")

// Camera
implementation("androidx.camera:camera-core:1.3.1")
implementation("androidx.camera:camera-camera2:1.3.1")
implementation("androidx.camera:camera-lifecycle:1.3.1")

// Coroutines
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
```

## Permissions Required

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
```

## Hardware Requirements

### Recommended
- **Audio**: 7.1 surround output
- **Camera**: Front-facing camera for motion tracking
- **Processing**: Multi-core CPU for real-time processing
- **Low-latency audio**: Hardware support preferred

### Minimum
- 2-channel stereo output (will upmix)
- Software audio processing
- 1080p camera

## Audio Configuration

### Sample Rate: 96 kHz
- Nyquist: 48 kHz (well above human hearing)
- Allows accurate high-frequency processing
- Better temporal resolution for delays

### Bit Depth: 32-bit Float
- Extended dynamic range
- Prevents quantization errors
- Native format for DSP operations

### Buffer Size: 4096 samples
- ~43ms latency at 96kHz
- Balance between latency and processing time

## Component Details

### VBAP Gain Calculation
```kotlin
// cos² law for energy preservation
val dotProduct = max(0f, dot(speakerPos, sourceDir))
val gain = (dotProduct * dotProduct) / sqrt(sumPow4)
```

### Transfer Function
```kotlin
val H = distanceAtten * airAbsorption * reflectionLoss * directivity
// where:
// distanceAtten = 1 / (1 + 0.5 * d^1.2)
// airAbsorption = exp(-ω * d / 50000)
// reflectionLoss = (1 - α)^r
```

### Multi-Band Filter
```kotlin
// Butterworth bandpass per octave
val alpha = sin(omega) / (2 * Q)
val output = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
```

### Audio-Motion Correlation
```kotlin
// Lag search for synchronization
maxCorr = max over λ∈[-5,5] { (1/N) Σ E_a[i] · E_m[i + λ] }
```

## Usage

### Basic Operation
1. Launch app on Android TV
2. Grant audio and camera permissions
3. Press "Start" to begin processing
4. Audio sources will be spatially positioned based on visual motion

### CV Toggle
- **Enabled**: Motion tracking affects audio positioning
- **Disabled**: Static spatial positioning
- **Auto**: Enabled for stereo/mono, disabled for true surround

### Performance Monitoring
- Real-time correlation display
- Detected object count
- Processing status indicators

## Calibration

### Speaker Layout
Default 7.1 configuration can be modified in `VBAPProcessor.kt`:
```kotlin
SpeakerLayout.create71Layout()
```

### Room Acoustics
Adjust in `TransferFunctionProcessor.kt`:
```kotlin
TransferFunctionProcessor(
    roomAbsorption = 0.3f,  // 0.0 = reflective, 1.0 = absorptive
    speedOfSound = 343f     // m/s at 20°C
)
```

### Filter Bank
Modify center frequencies in `MultiBandFilterBank.kt`:
```kotlin
62.5f * 2f.pow(b)  // Octave spacing
```

## Testing

### Audio Test Signals
1. Sine sweep: 20 Hz - 20 kHz
2. Pink noise: Verify flat response
3. Impulse: Check delay accuracy
4. Stereo test: Verify VBAP panning

### CV Test Scenarios
1. Static object: No motion = no CV effect
2. Moving object: Position tracks motion
3. Multiple objects: Highest velocity wins
4. Surround content: CV auto-disables

## Performance Optimization

### CPU Usage
- Multi-threading for parallel processing
- SIMD operations where possible
- Efficient buffer management

### Memory
- Pre-allocated buffers
- Circular buffers for delays
- Minimal allocations in audio thread

### Latency
- Direct audio path: ~43ms
- With CV: +16ms (camera frame delay)
- Total: ~60ms (acceptable for TV)

## Troubleshooting

### No Audio Output
- Check 7.1 surround support
- Verify audio permissions
- Test with stereo fallback

### High CPU Usage
- Reduce buffer size
- Decrease reflection order
- Lower sample rate (48kHz)

### CV Not Working
- Check camera permissions
- Verify camera availability
- Test motion detection threshold

## Future Enhancements

1. **GPU Acceleration**: OpenCL/RenderScript for FFT
2. **Machine Learning**: Deep learning for object recognition
3. **HRTF**: Head-related transfer functions for headphones
4. **Ambisonics**: Higher-order spatial encoding
5. **Network Audio**: Multi-device synchronization

## Mathematical Background

### Emergent Elevation
The cos² law in VBAP creates second-order spherical harmonics:
```
Y₂⁰(θ,φ) ∝ (3cos²φ - 1)
```
This naturally encodes elevation even with horizontal-only speakers.

### Sinc Reconstruction
The filter bank sum approximates ideal sinc interpolation:
```
Σ_b B_b(ω) ≈ sinc(ω/ω_s) * rect(ω/ω_s)
```

### Correlation Analysis
Cross-correlation finds optimal time alignment:
```
R(λ) = Σ_i x[i] · y[i + λ]
```

## License

Copyright 2025. All rights reserved.

## Credits

Based on research in:
- VBAP (Pulkki, 1997)
- Room acoustics (Kuttruff)
- Filter bank design (Oppenheim & Schafer)
- Computer vision (OpenCV algorithms)

## Contact

For issues and contributions, please submit via the project repository.
