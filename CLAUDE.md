# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a macOS SwiftUI application called "microphone_FFT" that provides real-time audio spectrum analysis using FFT (Fast Fourier Transform). The app captures microphone input and displays a live frequency spectrum visualization.

## Architecture

### Core Components

- **AudioEngineManager**: ObservableObject class that manages AVAudioEngine, performs FFT processing using vDSP/Accelerate framework, and publishes frequency/amplitude data
- **SpektrumView**: SwiftUI view that renders the real-time frequency spectrum as a bar chart
- **ContentView**: Main UI containing the spectrum visualization and audio controls
- **SpektrumAnalysatorApp**: SwiftUI App entry point with hidden title bar window style

### Key Technologies

- **AVFoundation**: Audio capture via AVAudioEngine
- **Accelerate/vDSP**: High-performance FFT calculations (2048-point FFT)
- **SwiftUI**: Modern declarative UI framework
- **Core Data**: Data persistence (basic setup via Persistence.swift)
- **Combine**: Reactive data flow through @Published properties

### Audio Processing Flow

1. AVAudioEngine captures microphone input at 44.1 kHz sample rate
2. Audio buffer (2048 samples) processed with Hanning window
3. FFT performed using vDSP for frequency domain conversion
4. Magnitude spectrum calculated and published to UI
5. SwektrumView renders real-time frequency bars (20 Hz - 20 kHz range)

## Development Commands

### Building and Running
```bash
# Open in Xcode
open microphone_FFT.xcodeproj

# Build from command line
xcodebuild -project microphone_FFT.xcodeproj -scheme microphone_FFT build

# Run tests
xcodebuild test -project microphone_FFT.xcodeproj -scheme microphone_FFT -destination 'platform=macOS'
```

### Testing
- Unit tests: `microphone_FFTTests/microphone_FFTTests.swift`
- UI tests: `microphone_FFTUITests/microphone_FFTUITests.swift`

## Key Implementation Notes

- FFT size: 2048 samples providing ~21.5 Hz frequency resolution
- Sample rate: 44.1 kHz standard audio rate
- Real-time processing with installTap on audio input node
- German language UI text throughout the application
- Window style set to hidden title bar for cleaner appearance
- Audio permissions required for microphone access

## File Organization

- `ContentView.swift`: Contains all main application code (484 lines)
- `Persistence.swift`: Core Data stack setup
- `Assets.xcassets/`: App icons and color assets
- `microphone_FFT.xcdatamodeld/`: Core Data model definition