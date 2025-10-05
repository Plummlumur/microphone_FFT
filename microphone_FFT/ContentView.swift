import SwiftUI
import AVFoundation
import Accelerate
import Combine
import CoreAudio
import AppKit

// MARK: - MFCC Feature Extractor
final class MFCCExtractor {
    let fftSize: Int
    let sampleRate: Double
    let nMelFilters: Int = 26
    let nCoefficients: Int = 13
    
    private var melFilterbank: [[Double]]!

    init(fftSize: Int, sampleRate: Double) {
        self.fftSize = fftSize
        self.sampleRate = sampleRate
        self.melFilterbank = createMelFilterbank()
    }
    
    private func hzToMel(_ hz: Double) -> Double {
        2595.0 * log10(1.0 + hz / 700.0)
    }
    private func melToHz(_ mel: Double) -> Double {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }
    
    private func createMelFilterbank() -> [[Double]] {
        let minMel: Double = hzToMel(0.0)
        let maxMel: Double = hzToMel(sampleRate / 2.0)

        let filterCount: Int = nMelFilters
        let pointCount: Int = filterCount + 2 // nMelFilters + 2 points define nMelFilters triangular filters

        // Evenly spaced points on the mel scale
        let melStep: Double = (maxMel - minMel) / Double(filterCount + 1)
        var melPoints = [Double](repeating: 0.0, count: pointCount)
        for i in 0..<pointCount {
            melPoints[i] = minMel + Double(i) * melStep
        }

        // Convert mel points to Hz
        var hzPoints = [Double](repeating: 0.0, count: pointCount)
        for i in 0..<pointCount {
            hzPoints[i] = melToHz(melPoints[i])
        }

        // Map Hz to FFT bin indices
        let nBins: Int = fftSize / 2
        let hzToBinScale: Double = Double(fftSize / 2 + 1) / sampleRate
        var bins = [Int](repeating: 0, count: pointCount)
        for i in 0..<pointCount {
            let raw = floor(hzPoints[i] * hzToBinScale)
            var idx = Int(raw)
            if idx < 0 { idx = 0 }
            if idx > nBins { idx = nBins }
            bins[i] = idx
        }

        // Build triangular filterbank
        var filterbank = Array(repeating: Array(repeating: 0.0, count: nBins), count: filterCount)
        for i in 0..<filterCount {
            let start = bins[i]
            let center = bins[i + 1]
            let end = bins[i + 2]
            if !(start < center && center < end) { continue }

            if center > start {
                let upper = min(center, nBins)
                if start < upper {
                    for j in start..<upper {
                        filterbank[i][j] = Double(j - start) / Double(center - start)
                    }
                }
            }
            if end > center {
                let upper = min(end, nBins)
                if center < upper {
                    for j in center..<upper {
                        filterbank[i][j] = Double(end - j) / Double(end - center)
                    }
                }
            }
        }

        return filterbank
    }
    
    func extractMFCCs(powerSpectrum: [Float]) -> [Double] {
        let powerD = powerSpectrum.map { Double($0) }
        
        var melSpectrum = [Double](repeating: 0, count: nMelFilters)
        for (i, filter) in melFilterbank.enumerated() {
            var sum = 0.0
            for j in 0..<min(filter.count, powerD.count) {
                sum += powerD[j] * filter[j]
            }
            melSpectrum[i] = log(max(sum, 1e-12))
        }
        
        var mfccs = [Double](repeating: 0, count: nCoefficients)
        let N = Double(melSpectrum.count)
        for i in 0..<nCoefficients {
            var s = 0.0
            for (j, mel) in melSpectrum.enumerated() {
                s += mel * cos(.pi * Double(i) * (Double(j) + 0.5) / N)
            }
            mfccs[i] = s
        }
        return mfccs
    }
}

// MARK: - Speaker Verification Model
@MainActor
final class SpeakerVerificationModel: ObservableObject {
    @Published var isVerified: Bool = false
    @Published var confidence: Double = 0.0
    @Published var status: String = "Bereit"
    @Published var userSampleCount: Int = 0
    @Published var otherSampleCount: Int = 0
    
    private var userMFCCsRaw: [[Double]] = [] { didSet { userSampleCount = userMFCCsRaw.count } }
    private var otherMFCCsRaw: [[Double]] = [] { didSet { otherSampleCount = otherMFCCsRaw.count } }
    
    private(set) var userMean: [Double] = Array(repeating: 0, count: 13)
    private(set) var userStd:  [Double] = Array(repeating: 1, count: 13)
    
    var threshold: Double = 0.65
    var continuousLearning: Bool = true
    private let minSamplesForTraining = 30
    
    private func maybeUpdateCMVN() {
        guard userMFCCsRaw.count >= minSamplesForTraining else { return }
        if userMFCCsRaw.count % 10 == 0 || userMFCCsRaw.count == minSamplesForTraining {
            computeUserCMVN()
        }
    }
    private func computeUserCMVN() {
        guard !userMFCCsRaw.isEmpty else { return }
        let count = Double(userMFCCsRaw.count)
        var mean = [Double](repeating: 0, count: userMean.count)
        var varAcc = [Double](repeating: 0, count: userMean.count)
        
        for v in userMFCCsRaw {
            for i in 0..<min(v.count, mean.count) { mean[i] += v[i] }
        }
        for i in 0..<mean.count { mean[i] /= count }
        for v in userMFCCsRaw {
            for i in 0..<min(v.count, varAcc.count) {
                let d = v[i] - mean[i]
                varAcc[i] += d*d
            }
        }
        for i in 0..<varAcc.count {
            userMean[i] = mean[i]
            userStd[i]  = max(sqrt(varAcc[i] / max(count, 1.0)), 1e-6)
        }
    }
    private func applyCMVN(_ v: [Double]) -> [Double] {
        guard userMFCCsRaw.count >= minSamplesForTraining else { return v }
        var out = v
        for i in 0..<min(v.count, userMean.count) {
            out[i] = (v[i] - userMean[i]) / userStd[i]
        }
        return out
    }
    private func cosine(_ a: [Double], _ b: [Double]) -> Double {
        let n = min(a.count, b.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<n { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
        if na <= 0 || nb <= 0 { return 0 }
        return dot / (sqrt(na)*sqrt(nb))
    }
    
    func addUserSample(mfccs: [Double]) {
        userMFCCsRaw.append(mfccs)
        if userMFCCsRaw.count > 300 { userMFCCsRaw.removeFirst() }
        maybeUpdateCMVN()
        if userMFCCsRaw.count % 25 == 0 { print("📊 User Samples: \(userMFCCsRaw.count)") }
    }
    func addOtherSample(mfccs: [Double]) {
        otherMFCCsRaw.append(mfccs)
        if otherMFCCsRaw.count > 300 { otherMFCCsRaw.removeFirst() }
        if otherMFCCsRaw.count % 25 == 0 { print("📊 Other Samples: \(otherMFCCsRaw.count)") }
    }
    
    func canTrain() -> Bool { userMFCCsRaw.count >= minSamplesForTraining }
    
    func verify(mfccs: [Double]) -> (verified: Bool, confidence: Double, shouldLearn: Bool) {
        guard canTrain() else { return (false, 0.0, false) }
        let q = applyCMVN(mfccs)
        
        let k = min(7, userMFCCsRaw.count)
        var sims: [Double] = []
        sims.reserveCapacity(userMFCCsRaw.count)
        for s in userMFCCsRaw {
            sims.append(cosine(q, applyCMVN(s)))
        }
        sims.sort(by: >)
        let userScore = sims.prefix(k).reduce(0, +) / Double(k)
        
        var otherScore = -1.0
        if otherMFCCsRaw.count > 10 {
            var os: [Double] = []
            let kOther = min(5, otherMFCCsRaw.count)
            for s in otherMFCCsRaw { os.append(cosine(q, applyCMVN(s))) }
            os.sort(by: >)
            otherScore = os.prefix(kOther).reduce(0, +) / Double(kOther)
        }
        
        let userSim01  = max(0, min(1, (userScore + 1) * 0.5))
        var finalConf = userSim01
        if otherScore > -1 {
            let other01 = max(0, min(1, (otherScore + 1) * 0.5))
            finalConf = max(0, min(1, userSim01 * (1.0 - other01) * 2.0))
        }
        let verified = finalConf > threshold
        let shouldLearn = continuousLearning && verified && finalConf > 0.8
        
        if Int.random(in: 0..<20) == 0 {
            print(String(format:"🎯 Cos=%.2f (other=%.2f) → conf=%.2f ✓=%@", userScore, otherScore, finalConf, verified.description))
        }
        return (verified, finalConf, shouldLearn)
    }
    
    func reset() {
        userMFCCsRaw.removeAll()
        otherMFCCsRaw.removeAll()
        userMean = Array(repeating: 0, count: 13)
        userStd  = Array(repeating: 1, count: 13)
        isVerified = false
        confidence = 0.0
        status = "Bereit"
        print("🔄 Model zurückgesetzt")
    }
    
    func saveModel() {
        let data = ModelData(
            schemaVersion: 2,
            userMFCCs: userMFCCsRaw,
            otherMFCCs: otherMFCCsRaw,
            threshold: threshold,
            userMean: userMean,
            userStd: userStd
        )
        let encoder = JSONEncoder()
        if let encoded = try? encoder.encode(data) {
            UserDefaults.standard.set(encoded, forKey: "SpeakerModelData")
            print("💾 Model gespeichert - User: \(userMFCCsRaw.count), Other: \(otherMFCCsRaw.count)")
        } else {
            print("❌ Fehler beim Speichern")
        }
    }
    func loadModel() {
        if let data = UserDefaults.standard.data(forKey: "SpeakerModelData") {
            let decoder = JSONDecoder()
            if let decoded = try? decoder.decode(ModelData.self, from: data) {
                userMFCCsRaw = decoded.userMFCCs
                otherMFCCsRaw = decoded.otherMFCCs
                threshold = decoded.threshold
                if decoded.schemaVersion >= 2 {
                    userMean = decoded.userMean
                    userStd  = decoded.userStd
                } else {
                    computeUserCMVN()
                }
                print("📂 Model geladen - User: \(userMFCCsRaw.count), Other: \(otherMFCCsRaw.count)")
            } else {
                print("⚠️ Fehler beim Laden")
            }
        } else {
            print("ℹ️ Kein gespeichertes Model gefunden")
        }
    }
    
    struct ModelData: Codable {
        let schemaVersion: Int
        let userMFCCs: [[Double]]
        let otherMFCCs: [[Double]]
        let threshold: Double
        let userMean: [Double]
        let userStd: [Double]
    }
}

// MARK: - Audio Engine Manager
final class AudioEngineManager: ObservableObject {
    @Published var frequenzen: [Double] = []
    @Published var amplituden: [Double] = []
    @Published var peakAmplituden: [Double] = []
    @Published var peakFrequenz: Double = 0.0
    @Published var peakAmplitude: Double = -100.0
    @Published var peakTrailFrequencies: [Double] = []
    @Published var waterfallHistory: [[Double]] = []
    @Published var waveformSamples: [Float] = []

    private let audioEngine = AVAudioEngine()
    private let fftSize = 2048
    
    // FFT (klassisch, real-to-complex)
    private var fftSetup: FFTSetup?
    private var log2n: vDSP_Length = 0
    private var realp: [Float] = []
    private var imagp: [Float] = []
    private var powerBuf: [Float] = []
    
    var sampleRate: Double = 44100.0
    
    var glaettungsFaktor: Double = 0.3
    private var geglätteteAmplituden: [Double] = []
    
    var peakHoldAktiv: Bool = true
    var peakFallRate: Double = 3.0
    private var letzteUpdateZeit: Date = Date()
    
    // Preallocated buffers
    private var hannWindow: [Float] = []
    
    // MFCC
    private var mfccExtractor: MFCCExtractor?
    
    // Speaker Verification
    var verificationModel: SpeakerVerificationModel?
    var verificationAktiv: Bool = false
    var enrollmentMode: EnrollmentMode = .none
    
    // VAD Parameter
    var vadEnergyDbThreshold: Double = -55.0
    var vadFlatnessMax: Double = 0.7

    // Analog dB (SPL) display options
    var analogDbAktiv: Bool = false
    var analogKalibrierungOffset: Double = 0.0
    var analogAWeightingAktiv: Bool = false

    // Peak marker trail
    var peakMarkerTrailSeconds: TimeInterval = 2.0
    private var peakMarkerTrail: [(time: Date, frequency: Double)] = []
    private var lastPeakFrequencyEmitted: Double = 0.0

    // Silence reset for verification status
    var silenceResetDelay: TimeInterval = 1.0
    private var lastSpeechTime: Date? = nil
    
    enum EnrollmentMode { case none, user, other }
    
    init() {
        setupFFT()
        checkAudioDevices()
    }
    
    private func checkAudioDevices() {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = AudioDeviceID()
        var propertySize = UInt32(MemoryLayout<AudioDeviceID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress, 0, nil, &propertySize, &deviceID
        )
        if status == noErr {
            print("\n📱 Standard-Eingabegerät ID: \(deviceID)")
            propertyAddress.mSelector = kAudioDevicePropertyDeviceNameCFString
            propertyAddress.mScope = kAudioDevicePropertyScopeInput
            var deviceName: Unmanaged<CFString>?
            propertySize = UInt32(MemoryLayout<Unmanaged<CFString>>.size)
            withUnsafeMutablePointer(to: &deviceName) { ptr in
                ptr.withMemoryRebound(to: CFString.self, capacity: 1) { cfPtr in
                    _ = AudioObjectGetPropertyData(deviceID, &propertyAddress, 0, nil, &propertySize, cfPtr)
                }
            }
            if let name = deviceName?.takeUnretainedValue() {
                print("📱 Gerätename: \(name)")
            }
        }
    }
    
    private func setupFFT() {
        log2n = vDSP_Length(log2(Double(fftSize)))
        fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))
        
        hannWindow = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&hannWindow, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        
        realp = [Float](repeating: 0, count: fftSize/2)
        imagp = [Float](repeating: 0, count: fftSize/2)
        powerBuf = [Float](repeating: 0, count: fftSize/2)
    }
    
    func startAudioEngine() {
        print("\n=== AUDIO ENGINE START ===\n")
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            print("✓ Mikrofon-Berechtigung: ERTEILT")
        case .notDetermined:
            print("⚠️ Mikrofon-Berechtigung: NICHT ANGEFRAGT")
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                if granted {
                    print("✓ Mikrofon-Berechtigung wurde erteilt")
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        self.startAudioEngine()
                    }
                } else {
                    print("✗ Mikrofon-Berechtigung wurde verweigert!")
                }
            }
            return
        case .denied:
            print("✗ Mikrofon-Berechtigung: VERWEIGERT"); return
        case .restricted:
            print("✗ Mikrofon-Berechtigung: EINGESCHRÄNKT"); return
        @unknown default:
            print("⚠️ Mikrofon-Berechtigung: UNBEKANNT")
        }
        
        checkAudioDevices()
        
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        sampleRate = inputFormat.sampleRate
        
        mfccExtractor = MFCCExtractor(fftSize: fftSize, sampleRate: sampleRate)
        print("\n🎵 Audio-Format: \(inputFormat.sampleRate) Hz, \(inputFormat.channelCount) ch")
        
        inputNode.installTap(onBus: 0, bufferSize: UInt32(fftSize), format: inputFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }
        
        audioEngine.prepare()
        do {
            try audioEngine.start()
            letzteUpdateZeit = Date()
            print("✓ Audio Engine gestartet\n")
        } catch {
            print("✗ Fehler: \(error)")
        }
    }
    
    func stopAudioEngine() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let fftSetup = fftSetup,
              let channelData = buffer.floatChannelData?[0] else { return }
        
        let frameCount = Int(buffer.frameLength)
        var samples = [Float](repeating: 0, count: fftSize)
        let nCopy = min(frameCount, fftSize)
        samples.withUnsafeMutableBufferPointer { dst in
            channelData.withMemoryRebound(to: Float.self, capacity: nCopy) { src in
                dst.baseAddress!.update(from: src, count: nCopy)
            }
        }
        vDSP.multiply(samples, hannWindow, result: &samples)
        
        // Real-FFT mit zrip
        var interleaved = [DSPComplex](repeating: DSPComplex(), count: fftSize/2)
        let limit = min(fftSize, samples.count)
        let pairCount = limit / 2
        if pairCount > 0 {
            for i in 0..<pairCount {
                let re = samples[2*i]
                let im = samples[2*i + 1]
                interleaved[i] = DSPComplex(real: re, imag: im)
            }
        }
        realp.withUnsafeMutableBufferPointer { rBuf in
            imagp.withUnsafeMutableBufferPointer { iBuf in
                var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                interleaved.withUnsafeBufferPointer { src in
                    vDSP_ctoz(src.baseAddress!, 2, &split, 1, vDSP_Length(pairCount))
                }
                vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                vDSP_zvmags(&split, 1, &powerBuf, 1, vDSP_Length(fftSize/2))
            }
        }
        
        // Normierung + dB
        let scale: Float = 1.0 / Float(fftSize * fftSize)
        vDSP.multiply(scale, powerBuf, result: &powerBuf)
        
        let minDb: Double = -100.0
        let halfSize = fftSize / 2
        var dBValues = [Double](repeating: minDb, count: halfSize)
        for i in 0..<halfSize {
            let p = max(powerBuf[i], 1e-12)
            dBValues[i] = max(10.0 * log10(Double(p)), minDb)
        }
        
        // Frequenzachse (20 Hz - 22 kHz)
        let frequenzAufloesung = sampleRate / Double(fftSize)
        let minIndex = Int(20.0 / frequenzAufloesung)
        let maxIndex = min(Int(22000.0 / frequenzAufloesung), halfSize)
        let filteredFrequenzen = (minIndex..<maxIndex).map { Double($0) * frequenzAufloesung }
        var filteredAmplituden = Array(dBValues[minIndex..<maxIndex])

        // Optional: convert to analog dB (approximate SPL) with calibration and A-weighting
        if analogDbAktiv {
            for i in 0..<filteredAmplituden.count {
                let f = filteredFrequenzen[i]
                let weight = analogAWeightingAktiv ? aWeightingDb(f) : 0.0
                filteredAmplituden[i] = filteredAmplituden[i] + analogKalibrierungOffset + weight
            }
        }
        
        // Glättung
        if geglätteteAmplituden.count != filteredAmplituden.count {
            geglätteteAmplituden = filteredAmplituden
        } else if glaettungsFaktor > 0 {
            for i in 0..<filteredAmplituden.count {
                geglätteteAmplituden[i] = geglätteteAmplituden[i] * glaettungsFaktor +
                                          filteredAmplituden[i] * (1.0 - glaettungsFaktor)
            }
        } else {
            geglätteteAmplituden = filteredAmplituden
        }
        filteredAmplituden = geglätteteAmplituden
        
        // VAD
        let energyDb = filteredAmplituden.max() ?? minDb
        let sfm: Double = {
            let p = powerBuf[minIndex..<maxIndex].map { Double(max($0, 1e-12)) }
            let geo = exp(p.map { log($0) }.reduce(0,+) / Double(p.count))
            let arith = p.reduce(0,+) / Double(p.count)
            return geo / max(arith, 1e-12)
        }()
        let isSpeechLike = (energyDb > vadEnergyDbThreshold) && (sfm < vadFlatnessMax)
        
        // MFCCs
        if let extractor = mfccExtractor {
            let powerForMFCC = Array(powerBuf[0..<halfSize])
            let mfccs = extractor.extractMFCCs(powerSpectrum: powerForMFCC)
            
            if let model = verificationModel {
                switch enrollmentMode {
                case .user where isSpeechLike:
                    model.addUserSample(mfccs: mfccs)
                case .other where isSpeechLike:
                    model.addOtherSample(mfccs: mfccs)
                case .none:
                    if verificationAktiv && model.canTrain() && isSpeechLike {
                        let result = model.verify(mfccs: mfccs)
                        if result.shouldLearn {
                            model.addUserSample(mfccs: mfccs)
                            if model.userSampleCount % 20 == 0 { model.saveModel() }
                        }
                        DispatchQueue.main.async {
                            model.isVerified = result.verified
                            model.confidence = result.confidence
                            model.status = result.verified ? "✓ Verifiziert" : "✗ Nicht erkannt"
                        }
                    }
                default: break
                }
            }
        }
        
        let jetzt = Date()

        // Max
        var maxAmplitude: Double = minDb
        var maxFrequenz: Double = 0.0
        for i in 0..<filteredAmplituden.count {
            if filteredAmplituden[i] > maxAmplitude {
                maxAmplitude = filteredAmplituden[i]
                maxFrequenz = filteredFrequenzen[i]
            }
        }

        // Peak frequency trail management
        if peakMarkerTrailSeconds > 0, maxFrequenz > 0 {
            let freqDelta = max(5.0, frequenzAufloesung * 2.0)
            if abs(maxFrequenz - lastPeakFrequencyEmitted) >= freqDelta {
                peakMarkerTrail.append((time: jetzt, frequency: maxFrequenz))
                lastPeakFrequencyEmitted = maxFrequenz
            }
            let cutoff = jetzt.addingTimeInterval(-peakMarkerTrailSeconds)
            peakMarkerTrail.removeAll { $0.time < cutoff }
        } else {
            peakMarkerTrail.removeAll()
            lastPeakFrequencyEmitted = 0.0
        }

        // Reset verification status after sustained silence
        if isSpeechLike {
            lastSpeechTime = jetzt
        } else if verificationAktiv, let model = verificationModel {
            if let last = lastSpeechTime, jetzt.timeIntervalSince(last) >= silenceResetDelay {
                DispatchQueue.main.async {
                    model.isVerified = false
                    model.confidence = 0.0
                    model.status = "Bereit"
                }
                lastSpeechTime = nil
            }
        }

        let zeitDifferenz = jetzt.timeIntervalSince(letzteUpdateZeit)
        letzteUpdateZeit = jetzt
        
        var neuePeaks: [Double]
        if peakAmplituden.count != filteredAmplituden.count {
            neuePeaks = filteredAmplituden
        } else {
            let fallAmount = peakFallRate * zeitDifferenz
            neuePeaks = peakAmplituden
            for i in 0..<filteredAmplituden.count {
                if filteredAmplituden[i] > neuePeaks[i] {
                    neuePeaks[i] = filteredAmplituden[i]
                } else {
                    neuePeaks[i] = max(neuePeaks[i] - fallAmount, filteredAmplituden[i])
                }
            }
        }
        
        // Downsampling für Canvas
        let (dsFreq, dsAmp, dsPeaks) = downsample(f: filteredFrequenzen, a: filteredAmplituden, p: neuePeaks, target: 1024)

        // Waterfall History (ringbuffer mit max 300 Zeilen)
        var updatedHistory = waterfallHistory
        updatedHistory.append(dsAmp)
        if updatedHistory.count > 300 {
            updatedHistory.removeFirst()
        }

        // Waveform samples für Oszilloskop (direkt aus Buffer)
        let waveformSamples = Array(samples.prefix(1024))

        DispatchQueue.main.async { [weak self] in
            self?.frequenzen = dsFreq
            self?.amplituden = dsAmp
            self?.peakAmplituden = dsPeaks
            self?.peakFrequenz = maxFrequenz
            self?.peakAmplitude = maxAmplitude
            self?.peakTrailFrequencies = self?.peakMarkerTrail.map { $0.frequency } ?? []
            self?.waterfallHistory = updatedHistory
            self?.waveformSamples = waveformSamples
        }
    }
    
    private func downsample(f: [Double], a: [Double], p: [Double], target: Int) -> ([Double],[Double],[Double]) {
        guard f.count > target && target > 0 else { return (f,a,p) }
        let step = Double(f.count) / Double(target)
        var rf: [Double] = []; rf.reserveCapacity(target)
        var ra: [Double] = []; ra.reserveCapacity(target)
        var rp: [Double] = []; rp.reserveCapacity(target)
        for i in 0..<target {
            let start = Int(Double(i) * step)
            let end   = min(Int(Double(i+1) * step), f.count)
            if start >= end { continue }
            let fa = f[start..<end]
            let aa = a[start..<end]
            let pp = p[start..<end]
            rf.append(fa.reduce(0,+) / Double(fa.count))
            ra.append(aa.reduce(0,+) / Double(aa.count))
            rp.append(pp.max() ?? aa.max() ?? -100.0)
        }
        return (rf, ra, rp)
    }
    
    private func aWeightingDb(_ f: Double) -> Double {
        guard f > 0 else { return 0 }
        let f2 = f * f
        let c1 = 20.6, c2 = 107.7, c3 = 737.9, c4 = 12194.0
        let c12 = c1 * c1
        let c22 = c2 * c2
        let c32 = c3 * c3
        let c42 = c4 * c4
        let num = c42 * f2 * f2
        let den = (f2 + c12) * sqrt((f2 + c22) * (f2 + c32)) * (f2 + c42)
        let ra = num / den
        return 20.0 * log10(ra) + 2.0
    }
    
    deinit {
        if let fftSetup { vDSP_destroy_fftsetup(fftSetup) }
        stopAudioEngine()
    }
}

// MARK: - Einstellungen (deprecated, use @AppStorage in Settings)
final class EinstellungenManager: ObservableObject {
    @AppStorage("peakHoldAktiv") var peakHoldAktiv = true
    @AppStorage("glaettungAktiv") var glaettungAktiv = true
    @AppStorage("glaettungsStaerke") var glaettungsStaerke: Double = 0.7
    @AppStorage("peakFallRate") var peakFallRate: Double = 3.0
    @AppStorage("frequenzmarkerAktiv") var frequenzmarkerAktiv = true
    @AppStorage("logarithmischeFrequenz") var logarithmischeFrequenz = false
    @AppStorage("analogDbAktiv") var analogDbAktiv = false
    @AppStorage("analogKalibrierungOffset") var analogKalibrierungOffset: Double = 0.0
    @AppStorage("analogAWeightingAktiv") var analogAWeightingAktiv = false
    @AppStorage("peakMarkerTrailSeconds") var peakMarkerTrailSeconds: Double = 2.0
}

// MARK: - Visualisierungsmodus
enum VisualisierungsModus: String, CaseIterable {
    case balken = "Balken"
    case linie = "Linie"
    case kombination = "Kombination"
    case wasserfall = "Wasserfall"
    case oszilloskop = "Oszilloskop"
}

// MARK: - Spektrum View
struct SpektrumView: View {
    let frequenzen: [Double]
    let amplituden: [Double]
    let peakAmplituden: [Double]
    let peakFrequenz: Double
    let peakAmplitude: Double
    let peakTrailFrequencies: [Double]
    let waterfallHistory: [[Double]]
    let waveformSamples: [Float]
    let modus: VisualisierungsModus
    @ObservedObject var einstellungen: EinstellungenManager

    var body: some View {
        Group {
            switch modus {
            case .wasserfall:
                WaterfallView(history: waterfallHistory, frequenzen: frequenzen, einstellungen: einstellungen)
            case .oszilloskop:
                OszilloskopView(samples: waveformSamples)
            default:
                spektrumCanvas
            }
        }
    }

    private var spektrumCanvas: some View {
        Canvas { context, size in
            guard !frequenzen.isEmpty && !amplituden.isEmpty else { return }
            let padding: CGFloat = 50
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black))
            zeichneAchsen(context: context, size: size, padding: padding)
            if einstellungen.frequenzmarkerAktiv { zeichneFrequenzmarker(context: context, size: size, padding: padding) }
            if !peakTrailFrequencies.isEmpty { zeichnePeakTrail(context: context, size: size, padding: padding) }
            switch modus {
            case .balken:       zeichneBalken(context: context, size: size, padding: padding)
            case .linie:        zeichneLinie(context: context, size: size, padding: padding, color: .green)
            case .kombination:  zeichneBalken(context: context, size: size, padding: padding); zeichneLinie(context: context, size: size, padding: padding, color: .cyan)
            default: break
            }
            if einstellungen.peakHoldAktiv { zeichnePeaks(context: context, size: size, padding: padding) }
            let minDb = einstellungen.analogDbAktiv ? 0.0 : -100.0
            if peakFrequenz > 0 && peakAmplitude > minDb { zeichnePeakFrequenzMarker(context: context, size: size, padding: padding) }
            zeichneBeschriftungen(context: context, size: size, padding: padding)
        }
        .background(Color.black)
    }
    
    private func formatiereFrequenz(_ freq: Double) -> String {
        if freq >= 1000 { return String(format: "%.1f kHz", freq / 1000.0) }
        return String(format: "%.2f kHz", freq / 1000.0)
    }
    private func frequenzZuX(_ freq: Double, width: CGFloat, padding: CGFloat) -> CGFloat {
        let drawWidth = width - 2 * padding
        if einstellungen.logarithmischeFrequenz {
            let minFreq = 20.0, maxFreq = 25000.0
            let logMin = log10(minFreq), logMax = log10(maxFreq), logFreq = log10(max(freq, minFreq))
            let normalized = (logFreq - logMin) / (logMax - logMin)
            return padding + CGFloat(normalized) * drawWidth
        } else {
            let minFreq = frequenzen.first ?? 20.0
            let maxFreq = frequenzen.last ?? 22000.0
            let normalized = (freq - minFreq) / max(1e-9, (maxFreq - minFreq))
            return padding + CGFloat(normalized) * drawWidth
        }
    }
    private func zeichneAchsen(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        var path = Path()
        path.move(to: CGPoint(x: padding, y: size.height - padding))
        path.addLine(to: CGPoint(x: size.width - padding, y: size.height - padding))
        path.move(to: CGPoint(x: padding, y: padding))
        path.addLine(to: CGPoint(x: padding, y: size.height - padding))
        context.stroke(path, with: .color(.white), lineWidth: 2)
    }
    private func zeichneFrequenzmarker(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        let markerFrequenzen: [Double] = einstellungen.logarithmischeFrequenz
            ? [50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 20000.0, 25000.0]
            : [50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 20000.0, 22000.0]
        for freq in markerFrequenzen {
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            var path = Path()
            path.move(to: CGPoint(x: x, y: size.height - padding))
            path.addLine(to: CGPoint(x: x, y: padding))
            context.stroke(path, with: .color(.white.opacity(0.15)), lineWidth: 1)
        }
    }
    private func zeichnePeakTrail(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        for freq in peakTrailFrequencies {
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            var path = Path()
            path.move(to: CGPoint(x: x, y: size.height - padding))
            path.addLine(to: CGPoint(x: x, y: padding))
            context.stroke(path, with: .color(.yellow.opacity(0.35)), lineWidth: 1)
        }
    }
    private func zeichnePeakFrequenzMarker(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        let x = frequenzZuX(peakFrequenz, width: size.width, padding: padding)
        var path = Path()
        path.move(to: CGPoint(x: x, y: size.height - padding))
        path.addLine(to: CGPoint(x: x, y: padding))
        context.stroke(path, with: .color(.yellow), lineWidth: 2)
        let frequenzText = formatiereFrequenz(peakFrequenz)
        let labelY = padding + 30
        let textSize = CGSize(width: 80, height: 25)
        let backgroundRect = CGRect(x: x - textSize.width / 2, y: labelY - textSize.height / 2, width: textSize.width, height: textSize.height)
        context.fill(Path(roundedRect: backgroundRect, cornerRadius: 5), with: .color(.black.opacity(0.7)))
        context.stroke(Path(roundedRect: backgroundRect, cornerRadius: 5), with: .color(.yellow), lineWidth: 1)
        context.draw(Text(frequenzText).foregroundColor(.yellow).font(.system(size: 12, weight: .bold)), at: CGPoint(x: x, y: labelY))
    }
    private func zeichneBalken(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        guard !amplituden.isEmpty else { return }
        let height = size.height - 2 * padding
        let minDb: Double = einstellungen.analogDbAktiv ? 0.0 : -100.0, maxDb: Double = einstellungen.analogDbAktiv ? 120.0 : 0.0
        for (index, amplitude) in amplituden.enumerated() {
            let freq = frequenzen[index]
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            let nextFreq = index < frequenzen.count - 1 ? frequenzen[index + 1] : freq + 1
            let nextX = frequenzZuX(nextFreq, width: size.width, padding: padding)
            let barWidth = max(nextX - x - 1, 1)
            let normalizedHeight = (amplitude - minDb) / (maxDb - minDb)
            let barHeight = CGFloat(max(0, normalizedHeight)) * height
            let y = size.height - padding - barHeight
            let rect = CGRect(x: x, y: y, width: barWidth, height: barHeight)
            let hue = 0.6 - (amplitude - minDb) / (maxDb - minDb) * 0.4
            let color = Color(hue: hue, saturation: 0.8, brightness: 0.9)
            context.fill(Path(rect), with: .color(color))
        }
    }
    private func zeichneLinie(context: GraphicsContext, size: CGSize, padding: CGFloat, color: Color) {
        guard amplituden.count > 1 else { return }
        let height = size.height - 2 * padding
        let minDb: Double = einstellungen.analogDbAktiv ? 0.0 : -100.0, maxDb: Double = einstellungen.analogDbAktiv ? 120.0 : 0.0
        var path = Path()
        for (index, amplitude) in amplituden.enumerated() {
            let freq = frequenzen[index]
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            let normalizedHeight = (amplitude - minDb) / (maxDb - minDb)
            let y = size.height - padding - CGFloat(max(0, normalizedHeight)) * height
            if index == 0 { path.move(to: CGPoint(x: x, y: y)) } else { path.addLine(to: CGPoint(x: x, y: y)) }
        }
        context.stroke(path, with: .color(color), lineWidth: 2)
    }
    private func zeichnePeaks(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        guard peakAmplituden.count == amplituden.count else { return }
        let height = size.height - 2 * padding
        let minDb: Double = einstellungen.analogDbAktiv ? 0.0 : -100.0, maxDb: Double = einstellungen.analogDbAktiv ? 120.0 : 0.0
        for (index, peakAmp) in peakAmplituden.enumerated() {
            let freq = frequenzen[index]
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            let nextFreq = index < frequenzen.count - 1 ? frequenzen[index + 1] : freq + 1
            let nextX = frequenzZuX(nextFreq, width: size.width, padding: padding)
            let lineWidth = max(nextX - x, 1)
            let normalizedHeight = (peakAmp - minDb) / (maxDb - minDb)
            let y = size.height - padding - CGFloat(max(0, normalizedHeight)) * height
            var path = Path()
            path.move(to: CGPoint(x: x, y: y))
            path.addLine(to: CGPoint(x: x + lineWidth, y: y))
            context.stroke(path, with: .color(.red), lineWidth: 1.5)
        }
    }
    private func zeichneBeschriftungen(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        let frequenzMarker: [(Double, String)] = einstellungen.logarithmischeFrequenz
            ? [(50,"50"),(100,"100"),(500,"500"),(1000,"1k"),(5000,"5k"),(10000,"10k"),(20000,"20k"),(25000,"25k")]
            : [(50,"50"),(100,"100"),(500,"500"),(1000,"1k"),(5000,"5k"),(10000,"10k"),(20000,"20k"),(22000,"22k")]
        for (freq, label) in frequenzMarker {
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            context.draw(Text(label).foregroundColor(.white).font(.system(size: 10)),
                         at: CGPoint(x: x, y: size.height - padding + 20))
        }
        let minDb: Double = einstellungen.analogDbAktiv ? 0.0 : -100.0
        let maxDb: Double = einstellungen.analogDbAktiv ? 120.0 : 0.0
        let dbMarker: [Int] = einstellungen.analogDbAktiv ? [40, 60, 80, 100, 120] : [-80, -60, -40, -20, 0]
        let height = size.height - 2 * padding
        for db in dbMarker {
            let normalizedY = (Double(db) - minDb) / (maxDb - minDb)
            let y = size.height - padding - CGFloat(normalizedY) * height
            context.draw(Text("\(db) dB").foregroundColor(.white).font(.system(size: 10)),
                         at: CGPoint(x: padding - 30, y: y))
        }
    }
}

// MARK: - Waterfall View (Spectrogram)
struct WaterfallView: View {
    let history: [[Double]]
    let frequenzen: [Double]
    @ObservedObject var einstellungen: EinstellungenManager

    var body: some View {
        Canvas { context, size in
            guard !history.isEmpty else { return }
            let padding: CGFloat = 50
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black))

            let drawWidth = size.width - 2 * padding
            let drawHeight = size.height - 2 * padding
            let rowHeight = drawHeight / CGFloat(history.count)

            let minDb: Double = einstellungen.analogDbAktiv ? 0.0 : -100.0
            let maxDb: Double = einstellungen.analogDbAktiv ? 120.0 : 0.0

            // Zeichne Historie von oben nach unten (neuste oben)
            for (rowIndex, row) in history.reversed().enumerated() {
                let y = padding + CGFloat(rowIndex) * rowHeight
                guard !row.isEmpty else { continue }

                for (i, amplitude) in row.enumerated() {
                    let x = padding + (CGFloat(i) / CGFloat(row.count)) * drawWidth
                    let width = max(drawWidth / CGFloat(row.count), 1)

                    // Farbe basierend auf Amplitude
                    let normalized = (amplitude - minDb) / (maxDb - minDb)
                    let color = colorForIntensity(normalized)

                    let rect = CGRect(x: x, y: y, width: width, height: max(rowHeight, 1))
                    context.fill(Path(rect), with: .color(color))
                }
            }

            // Achsen
            var path = Path()
            path.move(to: CGPoint(x: padding, y: padding))
            path.addLine(to: CGPoint(x: padding, y: size.height - padding))
            path.addLine(to: CGPoint(x: size.width - padding, y: size.height - padding))
            context.stroke(path, with: .color(.white), lineWidth: 2)

            // Frequenz-Beschriftungen
            let frequenzMarker: [(Double, String)] = [(100,"100"),(1000,"1k"),(5000,"5k"),(10000,"10k"),(20000,"20k"),(22000,"22k")]
            for (freq, label) in frequenzMarker {
                guard !frequenzen.isEmpty else { continue }
                let minFreq = frequenzen.first ?? 20.0
                let maxFreq = frequenzen.last ?? 22000.0
                let normalized = (freq - minFreq) / max(1e-9, (maxFreq - minFreq))
                let x = padding + CGFloat(normalized) * drawWidth
                context.draw(Text(label).foregroundColor(.white).font(.system(size: 10)),
                             at: CGPoint(x: x, y: size.height - padding + 20))
            }

            // Zeit-Beschriftungen (oben = neu, unten = alt)
            context.draw(Text("↑ Neu").foregroundColor(.white).font(.system(size: 10)),
                         at: CGPoint(x: padding - 30, y: padding + 10))
            context.draw(Text("↓ Alt").foregroundColor(.white).font(.system(size: 10)),
                         at: CGPoint(x: padding - 30, y: size.height - padding - 10))
        }
        .background(Color.black)
    }

    private func colorForIntensity(_ intensity: Double) -> Color {
        let clamped = max(0, min(1, intensity))
        // Farbverlauf: Blau (kalt) → Grün → Gelb → Rot (heiß)
        if clamped < 0.25 {
            let t = clamped / 0.25
            return Color(red: 0, green: 0, blue: 0.5 + t * 0.5)
        } else if clamped < 0.5 {
            let t = (clamped - 0.25) / 0.25
            return Color(red: 0, green: t, blue: 1.0 - t * 0.5)
        } else if clamped < 0.75 {
            let t = (clamped - 0.5) / 0.25
            return Color(red: t, green: 1.0, blue: 0.5 - t * 0.5)
        } else {
            let t = (clamped - 0.75) / 0.25
            return Color(red: 1.0, green: 1.0 - t, blue: 0)
        }
    }
}

// MARK: - Oszilloskop View
struct OszilloskopView: View {
    let samples: [Float]

    var body: some View {
        Canvas { context, size in
            guard !samples.isEmpty else { return }
            let padding: CGFloat = 50
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black))

            let drawWidth = size.width - 2 * padding
            let drawHeight = size.height - 2 * padding
            let centerY = size.height / 2

            // Gitter
            var gridPath = Path()
            // Horizontale Linien
            for i in 0...4 {
                let y = padding + CGFloat(i) * (drawHeight / 4)
                gridPath.move(to: CGPoint(x: padding, y: y))
                gridPath.addLine(to: CGPoint(x: size.width - padding, y: y))
            }
            // Vertikale Linien
            for i in 0...8 {
                let x = padding + CGFloat(i) * (drawWidth / 8)
                gridPath.move(to: CGPoint(x: x, y: padding))
                gridPath.addLine(to: CGPoint(x: x, y: size.height - padding))
            }
            context.stroke(gridPath, with: .color(.white.opacity(0.15)), lineWidth: 1)

            // Achsen
            var path = Path()
            path.move(to: CGPoint(x: padding, y: centerY))
            path.addLine(to: CGPoint(x: size.width - padding, y: centerY))
            path.move(to: CGPoint(x: padding, y: padding))
            path.addLine(to: CGPoint(x: padding, y: size.height - padding))
            context.stroke(path, with: .color(.white), lineWidth: 2)

            // Waveform
            var wavePath = Path()
            for (i, sample) in samples.enumerated() {
                let x = padding + (CGFloat(i) / CGFloat(samples.count)) * drawWidth
                let amplitude = CGFloat(sample)
                let y = centerY - amplitude * (drawHeight / 2)
                if i == 0 {
                    wavePath.move(to: CGPoint(x: x, y: y))
                } else {
                    wavePath.addLine(to: CGPoint(x: x, y: y))
                }
            }
            context.stroke(wavePath, with: .color(.green), lineWidth: 2)

            // Beschriftungen
            context.draw(Text("+1.0").foregroundColor(.white).font(.system(size: 10)),
                         at: CGPoint(x: padding - 25, y: padding))
            context.draw(Text("0.0").foregroundColor(.white).font(.system(size: 10)),
                         at: CGPoint(x: padding - 25, y: centerY))
            context.draw(Text("-1.0").foregroundColor(.white).font(.system(size: 10)),
                         at: CGPoint(x: padding - 25, y: size.height - padding))
            context.draw(Text("Zeit →").foregroundColor(.white).font(.system(size: 10)),
                         at: CGPoint(x: size.width / 2, y: size.height - padding + 20))
        }
        .background(Color.black)
    }
}

// MARK: - Export Manager
final class ExportManager {
    static func exportiereAlsCSV(frequenzen: [Double], amplituden: [Double]) -> URL? {
        var csvText = "Frequenz [Hz],Amplitude [dB]\n"
        for i in 0..<min(frequenzen.count, amplituden.count) {
            csvText += String(format: "%.2f,%.2f\n", frequenzen[i], amplituden[i])
        }
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "spektrum_\(Date().timeIntervalSince1970).csv"
        let fileURL = tempDir.appendingPathComponent(fileName)
        do { try csvText.write(to: fileURL, atomically: true, encoding: .utf8); return fileURL }
        catch { print("Fehler beim CSV-Export: \(error)"); return nil }
    }
    
    @MainActor
    static func exportiereAlsPNG(view: some View, size: CGSize) -> URL? {
        let renderer = ImageRenderer(content: view.frame(width: size.width, height: size.height))
        renderer.scale = 2.0
        guard let nsImage = renderer.nsImage,
              let tiffData = nsImage.tiffRepresentation,
              let bitmapRep = NSBitmapImageRep(data: tiffData),
              let pngData = bitmapRep.representation(using: .png, properties: [:]) else { return nil }
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "spektrum_\(Date().timeIntervalSince1970).png"
        let fileURL = tempDir.appendingPathComponent(fileName)
        do { try pngData.write(to: fileURL); return fileURL }
        catch { print("Fehler beim PNG-Export: \(error)"); return nil }
    }
}

// MARK: - Main Content View
struct ContentView: View {
    @StateObject private var audioManager = AudioEngineManager()
    @StateObject private var einstellungen = EinstellungenManager()
    @StateObject private var verificationModel = SpeakerVerificationModel()
    
    @State private var isRunning = false
    @State private var visualisierungsModus: VisualisierungsModus = .balken
    @State private var zeigeEinstellungen = false
    @State private var zeigeStimmerkennung = false
    @State private var exportErfolg = false
    @State private var exportNachricht = ""
    
    var body: some View {
        VStack(spacing: 15) {
            Picker("Visualisierung", selection: $visualisierungsModus) {
                ForEach(VisualisierungsModus.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 40)
            .padding(.top, 10)
            
            SpektrumView(
                frequenzen: audioManager.frequenzen,
                amplituden: audioManager.amplituden,
                peakAmplituden: audioManager.peakAmplituden,
                peakFrequenz: audioManager.peakFrequenz,
                peakAmplitude: audioManager.peakAmplitude,
                peakTrailFrequencies: audioManager.peakTrailFrequencies,
                waterfallHistory: audioManager.waterfallHistory,
                waveformSamples: audioManager.waveformSamples,
                modus: visualisierungsModus,
                einstellungen: einstellungen
            )
            .frame(height: 400)
            .cornerRadius(10)
            .padding(.horizontal)
            
            if audioManager.verificationAktiv {
                verificationStatus()
            }

            if exportErfolg {
                Text(exportNachricht).foregroundColor(.green).font(.caption)
            }

            infoBar()
        }
        .padding()
        .frame(minWidth: 900, minHeight: 750)
        .sheet(isPresented: $zeigeEinstellungen) {
            EinstellungenView(einstellungen: einstellungen, audioManager: audioManager)
        }
        .sheet(isPresented: $zeigeStimmerkennung) {
            StimmerkennungView(audioManager: audioManager, verificationModel: verificationModel)
        }
        .onChange(of: einstellungen.glaettungAktiv) { _, aktiv in
            audioManager.glaettungsFaktor = aktiv ? einstellungen.glaettungsStaerke : 0.0
        }
        .onChange(of: einstellungen.glaettungsStaerke) { _, wert in
            if einstellungen.glaettungAktiv { audioManager.glaettungsFaktor = wert }
        }
        .onChange(of: einstellungen.peakHoldAktiv) { _, aktiv in
            audioManager.peakHoldAktiv = aktiv
        }
        .onChange(of: einstellungen.peakFallRate) { _, rate in
            audioManager.peakFallRate = rate
        }
        .onChange(of: einstellungen.analogDbAktiv) { _, aktiv in
            audioManager.analogDbAktiv = aktiv
        }
        .onChange(of: einstellungen.analogKalibrierungOffset) { _, offset in
            audioManager.analogKalibrierungOffset = offset
        }
        .onChange(of: einstellungen.analogAWeightingAktiv) { _, aktiv in
            audioManager.analogAWeightingAktiv = aktiv
        }
        .onChange(of: einstellungen.peakMarkerTrailSeconds) { _, sekunden in
            audioManager.peakMarkerTrailSeconds = sekunden
        }
        .onAppear {
            audioManager.verificationModel = verificationModel
            audioManager.analogDbAktiv = einstellungen.analogDbAktiv
            audioManager.analogKalibrierungOffset = einstellungen.analogKalibrierungOffset
            audioManager.analogAWeightingAktiv = einstellungen.analogAWeightingAktiv
            audioManager.peakMarkerTrailSeconds = einstellungen.peakMarkerTrailSeconds
            verificationModel.loadModel()
            setupNotifications()
        }
        .onDisappear {
            NotificationCenter.default.removeObserver(self)
        }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button(action: toggleAudioEngine) {
                    Label(isRunning ? "Stoppen" : "Starten",
                          systemImage: isRunning ? "stop.circle.fill" : "play.circle.fill")
                }
                .help(isRunning ? "Audio-Engine stoppen" : "Audio-Engine starten")
                .tint(isRunning ? .red : .green)
            }

            ToolbarItemGroup(placement: .automatic) {
                Button(action: { zeigeStimmerkennung.toggle() }) {
                    Label("Stimmerkennung", systemImage: "person.wave.2")
                }
                .help("Stimmerkennung öffnen")
                .badge(verificationModel.isVerified ? "✓" : nil)

                Divider()

                Button(action: exportiereCSV) {
                    Label("CSV Export", systemImage: "arrow.down.doc")
                }
                .help("Als CSV exportieren (⌘E)")
                .disabled(!isRunning)

                Button(action: exportierePNG) {
                    Label("PNG Export", systemImage: "camera")
                }
                .help("Als PNG exportieren (⇧⌘E)")
                .disabled(!isRunning)
            }
        }
        .navigationTitle("Spektrumanalysator Pro")
    }
    
    @ViewBuilder
    private func infoBar() -> some View {
        HStack(spacing: 30) {
            VStack(alignment: .leading, spacing: 4) {
                Text("FFT: 2048 Samples").font(.caption)
                Text("Sample-Rate: \(Int(audioManager.sampleRate)) Hz").font(.caption)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Bereich: 20 Hz - 22 kHz").font(.caption)
                let peakText = audioManager.peakFrequenz > 0
                    ? formatiereFrequenz(audioManager.peakFrequenz)
                    : "---"
                Text("Peak: \(peakText)")
                    .font(.caption).foregroundColor(.yellow)
            }
        }
        .foregroundColor(.secondary)
    }
    
    @ViewBuilder
    private func verificationStatus() -> some View {
        HStack(spacing: 15) {
            Image(systemName: verificationModel.isVerified ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.title)
                .foregroundColor(verificationModel.isVerified ? .green : .red)
            VStack(alignment: .leading, spacing: 3) {
                Text(verificationModel.status).font(.headline)
                Text("Confidence: \(Int(verificationModel.confidence * 100))%")
                    .font(.caption).foregroundColor(.secondary)
                if verificationModel.continuousLearning && verificationModel.isVerified {
                    Text("🔄 Auto-Learning aktiv").font(.caption2).foregroundColor(.blue)
                }
            }
            Spacer()
            Text("\(verificationModel.userSampleCount) Samples")
                .font(.caption).foregroundColor(.secondary)
        }
        .padding()
        .background(Color.gray.opacity(0.2))
        .cornerRadius(10)
    }
    
    private func formatiereFrequenz(_ freq: Double) -> String {
        if freq >= 1000 { return String(format: "%.1f kHz", freq / 1000.0) }
        return String(format: "%.2f kHz", freq / 1000.0)
    }
    private func toggleAudioEngine() {
        if isRunning { audioManager.stopAudioEngine() }
        else { audioManager.startAudioEngine() }
        isRunning.toggle()
    }
    private func exportiereCSV() {
        if let url = ExportManager.exportiereAlsCSV(frequenzen: audioManager.frequenzen, amplituden: audioManager.amplituden) {
            NSWorkspace.shared.open(url.deletingLastPathComponent())
            exportNachricht = "CSV exportiert: \(url.lastPathComponent)"
            exportErfolg = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { exportErfolg = false }
        }
    }
    private func exportierePNG() {
        let spektrumView = SpektrumView(
            frequenzen: audioManager.frequenzen,
            amplituden: audioManager.amplituden,
            peakAmplituden: audioManager.peakAmplituden,
            peakFrequenz: audioManager.peakFrequenz,
            peakAmplitude: audioManager.peakAmplitude,
            peakTrailFrequencies: audioManager.peakTrailFrequencies,
            waterfallHistory: audioManager.waterfallHistory,
            waveformSamples: audioManager.waveformSamples,
            modus: visualisierungsModus,
            einstellungen: einstellungen
        )
        if let url = ExportManager.exportiereAlsPNG(view: spektrumView, size: CGSize(width: 1200, height: 600)) {
            NSWorkspace.shared.open(url.deletingLastPathComponent())
            exportNachricht = "PNG exportiert: \(url.lastPathComponent)"
            exportErfolg = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { exportErfolg = false }
        }
    }

    private func setupNotifications() {
        NotificationCenter.default.addObserver(
            forName: .exportCSV,
            object: nil,
            queue: .main
        ) { [self] _ in
            exportiereCSV()
        }

        NotificationCenter.default.addObserver(
            forName: .exportPNG,
            object: nil,
            queue: .main
        ) { [self] _ in
            exportierePNG()
        }

        NotificationCenter.default.addObserver(
            forName: .setVisualizationMode,
            object: nil,
            queue: .main
        ) { [self] notification in
            if let mode = notification.object as? VisualisierungsModus {
                visualisierungsModus = mode
            }
        }
    }
}

// MARK: - Stimmerkennung View
struct StimmerkennungView: View {
    @ObservedObject var audioManager: AudioEngineManager
    @ObservedObject var verificationModel: SpeakerVerificationModel
    @Environment(\.dismiss) var dismiss
    
    @State private var enrollmentProgress: Double = 0.0
    @State private var enrollmentTimer: Timer?
    @State private var enrollmentType: String = ""
    @State private var enrollmentStartTime: Date?
    
    var body: some View {
        VStack(spacing: 25) {
            Text("Stimmerkennung").font(.title).fontWeight(.bold)
            Divider()
            VStack(spacing: 10) {
                HStack { Text("User-Samples:"); Spacer(); Text("\(verificationModel.userSampleCount)").foregroundColor(.blue) }
                HStack { Text("Other-Samples:"); Spacer(); Text("\(verificationModel.otherSampleCount)").foregroundColor(.orange) }
                HStack {
                    Text("Training bereit:"); Spacer()
                    Text("\(verificationModel.canTrain() ? "✓ Ja" : "✗ Nein")")
                        .foregroundColor(verificationModel.canTrain() ? .green : .orange)
                }
                Divider()
                HStack { Text("Verification:"); Spacer()
                    Toggle("", isOn: $audioManager.verificationAktiv).disabled(!verificationModel.canTrain())
                }
                HStack { Text("Auto-Learning:"); Spacer()
                    Toggle("", isOn: $verificationModel.continuousLearning)
                }.help("Speichert automatisch neue Samples bei hoher Confidence")
                VStack(alignment: .leading, spacing: 5) {
                    HStack { Text("Threshold:"); Spacer(); Text("\(Int(verificationModel.threshold * 100))%") }
                    Slider(value: $verificationModel.threshold, in: 0.3...0.9, step: 0.05)
                }
            }
            .padding().background(Color.gray.opacity(0.1)).cornerRadius(10)
            
            VStack(spacing: 15) {
                Text("Training").font(.headline)
                Text("Sprich 10 Sekunden um deine Stimme zu lernen")
                    .font(.caption).foregroundColor(.secondary).multilineTextAlignment(.center)
                
                Button(action: startUserEnrollment) {
                    HStack { Image(systemName: "mic.fill"); Text("Meine Stimme lernen") }
                    .frame(maxWidth: .infinity).padding()
                    .background(Color.green).foregroundColor(.white).cornerRadius(10)
                }.disabled(audioManager.enrollmentMode != .none)
                
                Button(action: startOtherEnrollment) {
                    HStack { Image(systemName: "person.2"); Text("Andere Stimmen lernen") }
                    .frame(maxWidth: .infinity).padding()
                    .background(Color.orange).foregroundColor(.white).cornerRadius(10)
                }.disabled(audioManager.enrollmentMode != .none)
                
                if audioManager.enrollmentMode != .none {
                    VStack(spacing: 10) {
                        HStack { Image(systemName: "record.circle").foregroundColor(.red); Text("Aufnahme läuft: \(enrollmentType)").foregroundColor(.red) }
                        Text("\(Int(enrollmentProgress * 10)) / 10 Sekunden").font(.caption)
                        ProgressView(value: enrollmentProgress, total: 1.0).progressViewStyle(.linear)
                        Button("Abbrechen") { stopEnrollment() }
                            .foregroundColor(.red).padding(.top, 5)
                    }
                    .padding().background(Color.red.opacity(0.1)).cornerRadius(10)
                }
            }
            .padding().background(Color.gray.opacity(0.1)).cornerRadius(10)
            
            VStack(spacing: 15) {
                Text("Verwaltung").font(.headline)
                HStack(spacing: 15) {
                    Button(action: { verificationModel.reset() }) {
                        VStack { Image(systemName: "trash").font(.title2); Text("Zurücksetzen").font(.caption) }
                            .frame(maxWidth: .infinity).padding()
                            .background(Color.red.opacity(0.8)).foregroundColor(.white).cornerRadius(10)
                    }
                    Button(action: { verificationModel.saveModel() }) {
                        VStack { Image(systemName: "square.and.arrow.down").font(.title2); Text("Speichern").font(.caption) }
                            .frame(maxWidth: .infinity).padding()
                            .background(Color.blue).foregroundColor(.white).cornerRadius(10)
                    }
                }
            }
            .padding().background(Color.gray.opacity(0.1)).cornerRadius(10)
            
            Spacer()
            Button("Schließen") { dismiss() }.buttonStyle(.borderedProminent).padding(.bottom)
        }
        .padding().frame(width: 550, height: 800)
    }
    
    private func startUserEnrollment() {
        enrollmentType = "Deine Stimme"
        enrollmentProgress = 0.0
        enrollmentStartTime = Date()
        audioManager.enrollmentMode = .user
        audioManager.verificationAktiv = false
        scheduleEnrollmentTimer()
    }
    private func startOtherEnrollment() {
        enrollmentType = "Andere Stimmen"
        enrollmentProgress = 0.0
        enrollmentStartTime = Date()
        audioManager.enrollmentMode = .other
        audioManager.verificationAktiv = false
        scheduleEnrollmentTimer()
    }
    private func scheduleEnrollmentTimer() {
        enrollmentTimer?.invalidate()
        enrollmentTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            guard let startTime = enrollmentStartTime else { return }
            let elapsed = Date().timeIntervalSince(startTime)
            enrollmentProgress = min(elapsed / 10.0, 1.0)
            if elapsed >= 10.0 { stopEnrollment() }
        }
    }
    private func stopEnrollment() {
        enrollmentTimer?.invalidate(); enrollmentTimer = nil
        audioManager.enrollmentMode = .none
        enrollmentProgress = 0.0
        enrollmentStartTime = nil
        verificationModel.saveModel()
        print("✅ Training abgeschlossen - Model automatisch gespeichert")
    }
}

// MARK: - Einstellungen View
struct EinstellungenView: View {
    @ObservedObject var einstellungen: EinstellungenManager
    @ObservedObject var audioManager: AudioEngineManager
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Einstellungen").font(.title).fontWeight(.bold)
            Form {
                Section("Peak-Hold") {
                    Toggle("Peak-Hold aktivieren", isOn: $einstellungen.peakHoldAktiv)
                    if einstellungen.peakHoldAktiv {
                        VStack(alignment: .leading) {
                            Text("Fall-Rate: \(einstellungen.peakFallRate, specifier: "%.1f") dB/s")
                            Slider(value: $einstellungen.peakFallRate, in: 1...10, step: 0.5)
                        }
                    }
                }
                Section("Glättung") {
                    Toggle("Glättung aktivieren", isOn: $einstellungen.glaettungAktiv)
                    if einstellungen.glaettungAktiv {
                        VStack(alignment: .leading) {
                            Text("Glättungsstärke: \(Int(einstellungen.glaettungsStaerke * 100))%")
                            Slider(value: $einstellungen.glaettungsStaerke, in: 0.1...0.95)
                        }
                    }
                }
                Section("Anzeige") {
                    Toggle("Frequenzmarker anzeigen", isOn: $einstellungen.frequenzmarkerAktiv)
                    Toggle("Logarithmische Frequenzachse", isOn: $einstellungen.logarithmischeFrequenz)
                    Toggle("Analoge dB (SPL) anzeigen", isOn: $einstellungen.analogDbAktiv)
                    if einstellungen.analogDbAktiv {
                        Toggle("A-Bewertung (A-Weighting)", isOn: $einstellungen.analogAWeightingAktiv)
                        Stepper(value: $einstellungen.analogKalibrierungOffset, in: -60...60, step: 1) {
                            Text("Kalibrierung: \(Int(einstellungen.analogKalibrierungOffset)) dB")
                        }
                        Text("Kalibriere mit einem bekannten Pegel (z. B. 94 dB bei 1 kHz)")
                            .font(.caption).foregroundColor(.secondary)
                    }
                    Stepper(value: $einstellungen.peakMarkerTrailSeconds, in: 0...10, step: 0.5) {
                        Text("Peak-Marker Nachleuchten: \(einstellungen.peakMarkerTrailSeconds, specifier: "%.1f") s")
                    }
                    Text("0 s = aus. Letzte Höchstpegel werden als blasse Linien angezeigt.")
                        .font(.caption).foregroundColor(.secondary)
                }
                Section("Stimmerkennung (VAD)") {
                    Stepper(value: $audioManager.vadEnergyDbThreshold, in: (-80.0)...(-20.0), step: 1) {
                        Text("Energie-Schwelle: \(Int(audioManager.vadEnergyDbThreshold)) dB")
                    }
                    Slider(value: $audioManager.vadFlatnessMax, in: 0.2...0.95) {
                        Text("Spektrale Flachheit max.")
                    } minimumValueLabel: { Text("0.2") } maximumValueLabel: { Text("0.95") }
                    Text("Flatness < \(audioManager.vadFlatnessMax, specifier: "%.2f") gilt als sprachähnlich")
                        .font(.caption).foregroundColor(.secondary)
                }
            }
            .formStyle(.grouped)
            Button("Schließen") { dismiss() }.buttonStyle(.borderedProminent)
        }
        .padding().frame(width: 500, height: 540)
    }
}

// MARK: - App Entry Point
@main
struct SpektrumAnalysatorApp: App {
    @AppStorage("windowWidth") private var windowWidth: Double = 900
    @AppStorage("windowHeight") private var windowHeight: Double = 750

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                ContentView()
            }
            .frame(
                minWidth: 900, idealWidth: windowWidth, maxWidth: .infinity,
                minHeight: 750, idealHeight: windowHeight, maxHeight: .infinity
            )
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified(showsTitle: true))
        .defaultSize(width: windowWidth, height: windowHeight)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("Über Spektrumanalysator Pro") {
                    NSApplication.shared.orderFrontStandardAboutPanel()
                }
            }

            CommandGroup(replacing: .newItem) {
                EmptyView()
            }

            CommandMenu("Ablage") {
                Button("CSV exportieren...") {
                    NotificationCenter.default.post(name: .exportCSV, object: nil)
                }
                .keyboardShortcut("e", modifiers: .command)

                Button("PNG exportieren...") {
                    NotificationCenter.default.post(name: .exportPNG, object: nil)
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])

                Divider()

                Button("Fenster schließen") {
                    NSApplication.shared.keyWindow?.close()
                }
                .keyboardShortcut("w", modifiers: .command)
            }

            CommandMenu("Darstellung") {
                Button("Balken") {
                    NotificationCenter.default.post(name: .setVisualizationMode, object: VisualisierungsModus.balken)
                }
                .keyboardShortcut("1", modifiers: .command)

                Button("Linie") {
                    NotificationCenter.default.post(name: .setVisualizationMode, object: VisualisierungsModus.linie)
                }
                .keyboardShortcut("2", modifiers: .command)

                Button("Kombination") {
                    NotificationCenter.default.post(name: .setVisualizationMode, object: VisualisierungsModus.kombination)
                }
                .keyboardShortcut("3", modifiers: .command)

                Button("Wasserfall") {
                    NotificationCenter.default.post(name: .setVisualizationMode, object: VisualisierungsModus.wasserfall)
                }
                .keyboardShortcut("4", modifiers: .command)

                Button("Oszilloskop") {
                    NotificationCenter.default.post(name: .setVisualizationMode, object: VisualisierungsModus.oszilloskop)
                }
                .keyboardShortcut("5", modifiers: .command)
            }
        }

        Settings {
            SettingsView()
        }
    }
}

// MARK: - Settings View (macOS Standard)
struct SettingsView: View {
    var body: some View {
        TabView {
            AllgemeinSettingsTab()
                .tabItem {
                    Label("Allgemein", systemImage: "gear")
                }
                .tag(0)

            DarstellungSettingsTab()
                .tabItem {
                    Label("Darstellung", systemImage: "waveform")
                }
                .tag(1)

            AudioSettingsTab()
                .tabItem {
                    Label("Audio", systemImage: "speaker.wave.3")
                }
                .tag(2)

            StimmerkennungSettingsTab()
                .tabItem {
                    Label("Stimmerkennung", systemImage: "person.wave.2")
                }
                .tag(3)

            PresetsSettingsTab()
                .tabItem {
                    Label("Presets", systemImage: "folder")
                }
                .tag(4)
        }
        .frame(width: 500, height: 450)
    }
}

// MARK: - Settings Tabs
struct AllgemeinSettingsTab: View {
    @AppStorage("appVersion") private var appVersion = "1.0.0"

    var body: some View {
        Form {
            Section {
                LabeledContent("Version", value: appVersion)
                LabeledContent("Build", value: Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1")
            }

            Section("Über") {
                Text("Spektrumanalysator Pro ist ein Echtzeit-Audio-Analysator mit FFT und Stimmerkennung.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct DarstellungSettingsTab: View {
    @AppStorage("peakHoldAktiv") private var peakHoldAktiv = true
    @AppStorage("peakFallRate") private var peakFallRate = 3.0
    @AppStorage("glaettungAktiv") private var glaettungAktiv = true
    @AppStorage("glaettungsStaerke") private var glaettungsStaerke = 0.7
    @AppStorage("frequenzmarkerAktiv") private var frequenzmarkerAktiv = true
    @AppStorage("logarithmischeFrequenz") private var logarithmischeFrequenz = false
    @AppStorage("analogDbAktiv") private var analogDbAktiv = false
    @AppStorage("analogKalibrierungOffset") private var analogKalibrierungOffset = 0.0
    @AppStorage("analogAWeightingAktiv") private var analogAWeightingAktiv = false
    @AppStorage("peakMarkerTrailSeconds") private var peakMarkerTrailSeconds = 2.0

    var body: some View {
        Form {
            Section("Peak-Hold") {
                Toggle("Peak-Hold aktivieren", isOn: $peakHoldAktiv)
                if peakHoldAktiv {
                    VStack(alignment: .leading) {
                        Text("Fall-Rate: \(peakFallRate, specifier: "%.1f") dB/s")
                        Slider(value: $peakFallRate, in: 1...10, step: 0.5)
                    }
                }
            }

            Section("Glättung") {
                Toggle("Glättung aktivieren", isOn: $glaettungAktiv)
                if glaettungAktiv {
                    VStack(alignment: .leading) {
                        Text("Glättungsstärke: \(Int(glaettungsStaerke * 100))%")
                        Slider(value: $glaettungsStaerke, in: 0.1...0.95)
                    }
                }
            }

            Section("Anzeige") {
                Toggle("Frequenzmarker anzeigen", isOn: $frequenzmarkerAktiv)
                Toggle("Logarithmische Frequenzachse", isOn: $logarithmischeFrequenz)
                Toggle("Analoge dB (SPL) anzeigen", isOn: $analogDbAktiv)

                if analogDbAktiv {
                    Toggle("A-Bewertung (A-Weighting)", isOn: $analogAWeightingAktiv)
                    Stepper(value: $analogKalibrierungOffset, in: -60...60, step: 1) {
                        Text("Kalibrierung: \(Int(analogKalibrierungOffset)) dB")
                    }
                    Text("Kalibriere mit einem bekannten Pegel (z. B. 94 dB bei 1 kHz)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Stepper(value: $peakMarkerTrailSeconds, in: 0...10, step: 0.5) {
                    Text("Peak-Marker Nachleuchten: \(peakMarkerTrailSeconds, specifier: "%.1f") s")
                }
                Text("0 s = aus. Letzte Höchstpegel werden als blasse Linien angezeigt.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct AudioSettingsTab: View {
    @AppStorage("vadEnergyDbThreshold") private var vadEnergyDbThreshold = -55.0
    @AppStorage("vadFlatnessMax") private var vadFlatnessMax = 0.7

    var body: some View {
        Form {
            Section("Voice Activity Detection (VAD)") {
                Stepper(value: $vadEnergyDbThreshold, in: -80.0...(-20.0), step: 1) {
                    Text("Energie-Schwelle: \(Int(vadEnergyDbThreshold)) dB")
                }

                Slider(value: $vadFlatnessMax, in: 0.2...0.95) {
                    Text("Spektrale Flachheit max.")
                } minimumValueLabel: {
                    Text("0.2")
                } maximumValueLabel: {
                    Text("0.95")
                }

                Text("Flatness < \(vadFlatnessMax, specifier: "%.2f") gilt als sprachähnlich")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct StimmerkennungSettingsTab: View {
    @AppStorage("verificationThreshold") private var verificationThreshold = 0.65
    @AppStorage("continuousLearning") private var continuousLearning = true

    var body: some View {
        Form {
            Section("Verifikation") {
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text("Threshold:")
                        Spacer()
                        Text("\(Int(verificationThreshold * 100))%")
                    }
                    Slider(value: $verificationThreshold, in: 0.3...0.9, step: 0.05)
                }

                Toggle("Auto-Learning aktivieren", isOn: $continuousLearning)
                    .help("Speichert automatisch neue Samples bei hoher Confidence")
            }

            Section {
                Text("Sprich 10 Sekunden um deine Stimme zu trainieren. Verwende den Stimmerkennung-Dialog für das Training.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

struct PresetsSettingsTab: View {
    @State private var presetName: String = ""
    @State private var savedPresets: [String] = []
    @State private var showAlert = false
    @State private var alertMessage = ""

    var body: some View {
        Form {
            Section("Preset Speichern") {
                TextField("Preset-Name", text: $presetName)
                Button("Aktuelles Preset speichern") {
                    saveCurrentPreset()
                }
                .disabled(presetName.isEmpty)
            }

            Section("Gespeicherte Presets") {
                if savedPresets.isEmpty {
                    Text("Keine Presets gespeichert")
                        .foregroundColor(.secondary)
                        .font(.caption)
                } else {
                    ForEach(savedPresets, id: \.self) { preset in
                        HStack {
                            Text(preset)
                            Spacer()
                            Button("Laden") {
                                loadPreset(preset)
                            }
                            .buttonStyle(.bordered)
                            Button("Löschen") {
                                deletePreset(preset)
                            }
                            .buttonStyle(.bordered)
                            .tint(.red)
                        }
                    }
                }
            }

            Section("Vordefinierte Presets") {
                Button("Standard (Balanced)") { loadDefaultPreset() }
                Button("Hochauflösend (High Detail)") { loadHighDetailPreset() }
                Button("Performance (Low CPU)") { loadPerformancePreset() }
            }
        }
        .formStyle(.grouped)
        .padding()
        .onAppear {
            loadPresetsList()
        }
        .alert("Preset", isPresented: $showAlert) {
            Button("OK") { }
        } message: {
            Text(alertMessage)
        }
    }

    private func saveCurrentPreset() {
        let preset = SettingsPreset(
            name: presetName,
            peakHoldAktiv: UserDefaults.standard.bool(forKey: "peakHoldAktiv"),
            peakFallRate: UserDefaults.standard.double(forKey: "peakFallRate"),
            glaettungAktiv: UserDefaults.standard.bool(forKey: "glaettungAktiv"),
            glaettungsStaerke: UserDefaults.standard.double(forKey: "glaettungsStaerke"),
            frequenzmarkerAktiv: UserDefaults.standard.bool(forKey: "frequenzmarkerAktiv"),
            logarithmischeFrequenz: UserDefaults.standard.bool(forKey: "logarithmischeFrequenz"),
            analogDbAktiv: UserDefaults.standard.bool(forKey: "analogDbAktiv"),
            analogKalibrierungOffset: UserDefaults.standard.double(forKey: "analogKalibrierungOffset"),
            analogAWeightingAktiv: UserDefaults.standard.bool(forKey: "analogAWeightingAktiv"),
            peakMarkerTrailSeconds: UserDefaults.standard.double(forKey: "peakMarkerTrailSeconds")
        )

        if let encoded = try? JSONEncoder().encode(preset) {
            UserDefaults.standard.set(encoded, forKey: "preset_\(presetName)")
            loadPresetsList()
            alertMessage = "Preset '\(presetName)' gespeichert"
            showAlert = true
            presetName = ""
        }
    }

    private func loadPreset(_ name: String) {
        if let data = UserDefaults.standard.data(forKey: "preset_\(name)"),
           let preset = try? JSONDecoder().decode(SettingsPreset.self, from: data) {
            UserDefaults.standard.set(preset.peakHoldAktiv, forKey: "peakHoldAktiv")
            UserDefaults.standard.set(preset.peakFallRate, forKey: "peakFallRate")
            UserDefaults.standard.set(preset.glaettungAktiv, forKey: "glaettungAktiv")
            UserDefaults.standard.set(preset.glaettungsStaerke, forKey: "glaettungsStaerke")
            UserDefaults.standard.set(preset.frequenzmarkerAktiv, forKey: "frequenzmarkerAktiv")
            UserDefaults.standard.set(preset.logarithmischeFrequenz, forKey: "logarithmischeFrequenz")
            UserDefaults.standard.set(preset.analogDbAktiv, forKey: "analogDbAktiv")
            UserDefaults.standard.set(preset.analogKalibrierungOffset, forKey: "analogKalibrierungOffset")
            UserDefaults.standard.set(preset.analogAWeightingAktiv, forKey: "analogAWeightingAktiv")
            UserDefaults.standard.set(preset.peakMarkerTrailSeconds, forKey: "peakMarkerTrailSeconds")
            alertMessage = "Preset '\(name)' geladen"
            showAlert = true
        }
    }

    private func deletePreset(_ name: String) {
        UserDefaults.standard.removeObject(forKey: "preset_\(name)")
        loadPresetsList()
    }

    private func loadPresetsList() {
        savedPresets = UserDefaults.standard.dictionaryRepresentation().keys
            .filter { $0.hasPrefix("preset_") }
            .map { $0.replacingOccurrences(of: "preset_", with: "") }
            .sorted()
    }

    private func loadDefaultPreset() {
        UserDefaults.standard.set(true, forKey: "peakHoldAktiv")
        UserDefaults.standard.set(3.0, forKey: "peakFallRate")
        UserDefaults.standard.set(true, forKey: "glaettungAktiv")
        UserDefaults.standard.set(0.7, forKey: "glaettungsStaerke")
        UserDefaults.standard.set(true, forKey: "frequenzmarkerAktiv")
        UserDefaults.standard.set(false, forKey: "logarithmischeFrequenz")
        alertMessage = "Standard-Preset geladen"
        showAlert = true
    }

    private func loadHighDetailPreset() {
        UserDefaults.standard.set(true, forKey: "peakHoldAktiv")
        UserDefaults.standard.set(1.0, forKey: "peakFallRate")
        UserDefaults.standard.set(true, forKey: "glaettungAktiv")
        UserDefaults.standard.set(0.9, forKey: "glaettungsStaerke")
        UserDefaults.standard.set(true, forKey: "frequenzmarkerAktiv")
        UserDefaults.standard.set(true, forKey: "logarithmischeFrequenz")
        UserDefaults.standard.set(5.0, forKey: "peakMarkerTrailSeconds")
        alertMessage = "Hochauflösend-Preset geladen"
        showAlert = true
    }

    private func loadPerformancePreset() {
        UserDefaults.standard.set(false, forKey: "peakHoldAktiv")
        UserDefaults.standard.set(5.0, forKey: "peakFallRate")
        UserDefaults.standard.set(true, forKey: "glaettungAktiv")
        UserDefaults.standard.set(0.5, forKey: "glaettungsStaerke")
        UserDefaults.standard.set(false, forKey: "frequenzmarkerAktiv")
        UserDefaults.standard.set(0.0, forKey: "peakMarkerTrailSeconds")
        alertMessage = "Performance-Preset geladen"
        showAlert = true
    }
}

struct SettingsPreset: Codable {
    let name: String
    let peakHoldAktiv: Bool
    let peakFallRate: Double
    let glaettungAktiv: Bool
    let glaettungsStaerke: Double
    let frequenzmarkerAktiv: Bool
    let logarithmischeFrequenz: Bool
    let analogDbAktiv: Bool
    let analogKalibrierungOffset: Double
    let analogAWeightingAktiv: Bool
    let peakMarkerTrailSeconds: Double
}

// MARK: - Notification Names
extension Notification.Name {
    static let exportCSV = Notification.Name("exportCSV")
    static let exportPNG = Notification.Name("exportPNG")
    static let setVisualizationMode = Notification.Name("setVisualizationMode")
}

// MARK: - kleine Hilfen
fileprivate extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        return Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}

