import SwiftUI
import AVFoundation
import Accelerate
import Combine
import CoreAudio
import CoreML
import CreateML

// MARK: - MFCC Feature Extractor
class MFCCExtractor {
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
        return 2595.0 * log10(1.0 + hz / 700.0)
    }
    
    private func melToHz(_ mel: Double) -> Double {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }
    
    private func createMelFilterbank() -> [[Double]] {
        let minMel = hzToMel(0)
        let maxMel = hzToMel(sampleRate / 2)
        
        // Mel-Punkte berechnen
        var melPoints: [Double] = []
        for i in 0...(nMelFilters + 1) {
            let point = minMel + (Double(i) * (maxMel - minMel) / Double(nMelFilters + 1))
            melPoints.append(point)
        }
        
        // Hz-Punkte berechnen
        var hzPoints: [Double] = []
        for mel in melPoints {
            hzPoints.append(melToHz(mel))
        }
        
        // Bins berechnen
        var bins: [Int] = []
        for hz in hzPoints {
            let bin = Int(floor(Double(fftSize / 2 + 1) * hz / sampleRate))
            bins.append(bin)
        }
        
        var filterbank = [[Double]](repeating: [Double](repeating: 0, count: fftSize/2), count: nMelFilters)
        
        for i in 0..<nMelFilters {
            let start = bins[i]
            let center = bins[i+1]
            let end = bins[i+2]
            
            for j in start..<min(center, fftSize/2) {
                let diff = center - start
                if diff > 0 {
                    filterbank[i][j] = Double(j - start) / Double(diff)
                }
            }
            for j in center..<min(end, fftSize/2) {
                let diff = end - center
                if diff > 0 {
                    filterbank[i][j] = Double(end - j) / Double(diff)
                }
            }
        }
        
        return filterbank
    }
    
    func extractMFCCs(magnitudes: [Float]) -> [Double] {
        let magnitudesDouble = magnitudes.map { Double($0) }
        
        // Mel-Filterbank anwenden
        var melSpectrum = [Double](repeating: 0, count: nMelFilters)
        for (i, filter) in melFilterbank.enumerated() {
            var sum = 0.0
            for (j, mag) in magnitudesDouble.prefix(fftSize/2).enumerated() {
                sum += mag * filter[j]
            }
            melSpectrum[i] = log(max(sum, 1e-10))
        }
        
        // DCT
        var mfccs = [Double](repeating: 0, count: nCoefficients)
        let N = Double(melSpectrum.count)
        
        for i in 0..<nCoefficients {
            var sum = 0.0
            for (j, mel) in melSpectrum.enumerated() {
                sum += mel * cos(Double.pi * Double(i) * (Double(j) + 0.5) / N)
            }
            mfccs[i] = sum
        }
        
        return mfccs
    }
}

// MARK: - Speaker Verification Model
class SpeakerVerificationModel: ObservableObject {
    @Published var isVerified: Bool = false
    @Published var confidence: Double = 0.0
    @Published var status: String = "Bereit"
    @Published var userSampleCount: Int = 0
    @Published var otherSampleCount: Int = 0
    
    private var userMFCCs: [[Double]] = [] {
        didSet {
            userSampleCount = userMFCCs.count
        }
    }
    private var otherMFCCs: [[Double]] = [] {
        didSet {
            otherSampleCount = otherMFCCs.count
        }
    }
    
    var threshold: Double = 0.55  // Niedriger Threshold für bessere Erkennung
    var continuousLearning: Bool = true
    private let minSamplesForTraining = 30  // Weniger Samples nötig
    
    func addUserSample(mfccs: [Double]) {
        let normalized = normalizeMFCCs(mfccs)
        userMFCCs.append(normalized)
        if userMFCCs.count > 300 {
            userMFCCs.removeFirst()
        }
        print("📊 User Samples: \(userMFCCs.count)")
    }
    
    func addOtherSample(mfccs: [Double]) {
        let normalized = normalizeMFCCs(mfccs)
        otherMFCCs.append(normalized)
        if otherMFCCs.count > 300 {
            otherMFCCs.removeFirst()
        }
        print("📊 Other Samples: \(otherMFCCs.count)")
    }
    
    private func normalizeMFCCs(_ mfccs: [Double]) -> [Double] {
        // Z-Score Normalisierung
        let mean = mfccs.reduce(0, +) / Double(mfccs.count)
        let variance = mfccs.map { pow($0 - mean, 2) }.reduce(0, +) / Double(mfccs.count)
        let std = sqrt(variance)
        
        if std > 0 {
            return mfccs.map { ($0 - mean) / std }
        }
        return mfccs
    }
    
    func canTrain() -> Bool {
        return userMFCCs.count >= minSamplesForTraining
    }
    
    func verify(mfccs: [Double]) -> (verified: Bool, confidence: Double, shouldLearn: Bool) {
        guard userMFCCs.count >= minSamplesForTraining else {
            return (false, 0.0, false)
        }
        
        let normalized = normalizeMFCCs(mfccs)
        
        // K-NN mit k=7 für robustere Erkennung
        let k = min(7, userMFCCs.count)
        var userDistances: [Double] = []
        for userSample in userMFCCs {
            let dist = euclideanDistance(a: normalized, b: userSample)
            userDistances.append(dist)
        }
        userDistances.sort()
        let avgUserDist = userDistances.prefix(k).reduce(0, +) / Double(k)
        
        // Wenn wir "Andere"-Samples haben, vergleiche damit
        var avgOtherDist = Double.infinity
        if otherMFCCs.count > 10 {
            var otherDistances: [Double] = []
            for otherSample in otherMFCCs {
                let dist = euclideanDistance(a: normalized, b: otherSample)
                otherDistances.append(dist)
            }
            otherDistances.sort()
            let kOther = min(5, otherMFCCs.count)
            avgOtherDist = otherDistances.prefix(kOther).reduce(0, +) / Double(kOther)
        }
        
        // Relative Confidence: wie viel näher ist User vs. Other?
        let maxDist = 50.0
        let userSimilarity = max(0, 1.0 - (avgUserDist / maxDist))
        
        // Wenn wir Other-Samples haben, nutze Ratio
        var finalConfidence = userSimilarity
        if avgOtherDist != Double.infinity {
            let ratio = avgOtherDist / max(avgUserDist, 0.1)
            finalConfidence = min(1.0, userSimilarity * ratio * 0.5)
        }
        
        let verified = finalConfidence > threshold
        
        // Kontinuierliches Learning: Bei hoher Confidence User-Sample hinzufügen
        let shouldLearn = continuousLearning && verified && finalConfidence > 0.75
        
        if Int.random(in: 0..<20) == 0 {
            print("🎯 Verification - Confidence: \(String(format: "%.2f", finalConfidence)), User Dist: \(String(format: "%.2f", avgUserDist)), Other Dist: \(String(format: "%.2f", avgOtherDist)), Verified: \(verified)")
        }
        
        return (verified, finalConfidence, shouldLearn)
    }
    
    private func euclideanDistance(a: [Double], b: [Double]) -> Double {
        var sum = 0.0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    func reset() {
        userMFCCs.removeAll()
        otherMFCCs.removeAll()
        isVerified = false
        confidence = 0.0
        status = "Bereit"
        print("🔄 Model zurückgesetzt")
    }
    
    func saveModel() {
        let data = ModelData(userMFCCs: userMFCCs, otherMFCCs: otherMFCCs, threshold: threshold)
        let encoder = JSONEncoder()
        if let encoded = try? encoder.encode(data) {
            UserDefaults.standard.set(encoded, forKey: "SpeakerModelData")
            print("💾 Model gespeichert - User: \(userMFCCs.count), Other: \(otherMFCCs.count)")
        } else {
            print("❌ Fehler beim Speichern")
        }
    }
    
    func loadModel() {
        if let data = UserDefaults.standard.data(forKey: "SpeakerModelData") {
            let decoder = JSONDecoder()
            if let decoded = try? decoder.decode(ModelData.self, from: data) {
                userMFCCs = decoded.userMFCCs
                otherMFCCs = decoded.otherMFCCs
                threshold = decoded.threshold
                print("📂 Model geladen - User: \(userMFCCs.count), Other: \(otherMFCCs.count)")
            } else {
                print("⚠️ Fehler beim Laden")
            }
        } else {
            print("ℹ️ Kein gespeichertes Model gefunden")
        }
    }
    
    struct ModelData: Codable {
        let userMFCCs: [[Double]]
        let otherMFCCs: [[Double]]
        let threshold: Double
    }
}

// MARK: - Audio Engine Manager
class AudioEngineManager: ObservableObject {
    @Published var frequenzen: [Double] = []
    @Published var amplituden: [Double] = []
    @Published var peakAmplituden: [Double] = []
    @Published var peakFrequenz: Double = 0.0
    @Published var peakAmplitude: Double = -100.0
    
    private let audioEngine = AVAudioEngine()
    private let fftSize = 2048
    private var fftSetup: vDSP.FFT<DSPSplitComplex>?
    var sampleRate: Double = 44100.0
    
    var glaettungsFaktor: Double = 0.3
    private var geglätteteAmplituden: [Double] = []
    
    var peakHoldAktiv: Bool = true
    var peakFallRate: Double = 3.0
    private var letzteUpdateZeit: Date = Date()
    
    // MFCC Extraktor
    private var mfccExtractor: MFCCExtractor?
    
    // Speaker Verification
    var verificationModel: SpeakerVerificationModel?
    var verificationAktiv: Bool = false
    var enrollmentMode: EnrollmentMode = .none
    
    enum EnrollmentMode {
        case none
        case user
        case other
    }
    
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
            &propertyAddress,
            0,
            nil,
            &propertySize,
            &deviceID
        )
        
        if status == noErr {
            print("\n📱 Standard-Eingabegerät ID: \(deviceID)")
            
            propertyAddress.mSelector = kAudioDevicePropertyDeviceNameCFString
            propertyAddress.mScope = kAudioDevicePropertyScopeInput
            
            var deviceName: CFString = "" as CFString
            propertySize = UInt32(MemoryLayout<CFString>.size)
            
            let nameStatus = AudioObjectGetPropertyData(
                deviceID,
                &propertyAddress,
                0,
                nil,
                &propertySize,
                &deviceName
            )
            
            if nameStatus == noErr {
                print("📱 Gerätename: \(deviceName)")
            }
        }
    }
        
    private func setupFFT() {
        let log2n = vDSP_Length(log2(Double(fftSize)))
        fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)
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
            print("✗ Mikrofon-Berechtigung: VERWEIGERT")
            return
        case .restricted:
            print("✗ Mikrofon-Berechtigung: EINGESCHRÄNKT")
            return
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
        
        // MFCC Extraktor initialisieren
        mfccExtractor = MFCCExtractor(fftSize: fftSize, sampleRate: sampleRate)
        
        print("\n🎵 Audio-Format: \(inputFormat.sampleRate) Hz, \(inputFormat.channelCount) ch")
        
        inputNode.installTap(onBus: 0, bufferSize: UInt32(fftSize), format: inputFormat) { [weak self] buffer, time in
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
        guard let channelData = buffer.floatChannelData?[0],
              let fftSetup = fftSetup else { return }
        
        let frameCount = Int(buffer.frameLength)
        var samples = [Float](repeating: 0, count: fftSize)
        let samplesToCopy = min(frameCount, fftSize)
        
        for i in 0..<samplesToCopy {
            samples[i] = channelData[i]
        }
        
        var window = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        vDSP_vmul(samples, 1, window, 1, &samples, 1, vDSP_Length(fftSize))
        
        let halfSize = fftSize / 2
        var realParts = [Float](repeating: 0, count: halfSize)
        var imagParts = [Float](repeating: 0, count: halfSize)
        
        samples.withUnsafeBytes { (samplesPtr: UnsafeRawBufferPointer) in
            samplesPtr.baseAddress?.withMemoryRebound(to: DSPComplex.self, capacity: halfSize) { complexPtr in
                realParts.withUnsafeMutableBufferPointer { realBuf in
                    imagParts.withUnsafeMutableBufferPointer { imagBuf in
                        var splitComplex = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(halfSize))
                        fftSetup.forward(input: splitComplex, output: &splitComplex)
                    }
                }
            }
        }
        
        var magnitudes = [Float](repeating: 0, count: halfSize)
        realParts.withUnsafeBufferPointer { realBuf in
            imagParts.withUnsafeBufferPointer { imagBuf in
                var splitComplex = DSPSplitComplex(realp: UnsafeMutablePointer(mutating: realBuf.baseAddress!),
                                                   imagp: UnsafeMutablePointer(mutating: imagBuf.baseAddress!))
                vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(halfSize))
            }
        }
        
        // MFCCs extrahieren für Stimmerkennung
        if let extractor = mfccExtractor {
            let mfccs = extractor.extractMFCCs(magnitudes: magnitudes)
            
            // Speaker Verification
            if let model = verificationModel {
                switch enrollmentMode {
                case .user:
                    model.addUserSample(mfccs: mfccs)
                case .other:
                    model.addOtherSample(mfccs: mfccs)
                case .none:
                    if verificationAktiv && model.canTrain() {
                        let result = model.verify(mfccs: mfccs)
                        
                        // Kontinuierliches Learning: Bei hoher Confidence nachtrainieren
                        if result.shouldLearn {
                            model.addUserSample(mfccs: mfccs)
                            // Automatisch speichern alle 20 Samples
                            if model.userSampleCount % 20 == 0 {
                                model.saveModel()
                            }
                        }
                        
                        DispatchQueue.main.async {
                            model.isVerified = result.verified
                            model.confidence = result.confidence
                            model.status = result.verified ? "✓ Verifiziert" : "✗ Nicht erkannt"
                        }
                    }
                }
            }
        }
        
        // Rest der FFT-Verarbeitung für Spektrum
        var dBValues = magnitudes.map { magnitude in
            let normalized = magnitude / Float(fftSize)
            let db = 20 * log10(max(normalized, 1e-10))
            return max(Double(db), -100.0)
        }
        
        let frequenzAufloesung = sampleRate / Double(fftSize)
        let minIndex = Int(20.0 / frequenzAufloesung)
        let maxIndex = min(Int(20000.0 / frequenzAufloesung), halfSize)
        
        let filteredFrequenzen = (minIndex..<maxIndex).map { Double($0) * frequenzAufloesung }
        var filteredAmplituden = Array(dBValues[minIndex..<maxIndex])
        
        if geglätteteAmplituden.isEmpty {
            geglätteteAmplituden = filteredAmplituden
        } else {
            for i in 0..<filteredAmplituden.count {
                geglätteteAmplituden[i] = geglätteteAmplituden[i] * glaettungsFaktor +
                                          filteredAmplituden[i] * (1.0 - glaettungsFaktor)
            }
            filteredAmplituden = geglätteteAmplituden
        }
        
        let jetzt = Date()
        let zeitDifferenz = jetzt.timeIntervalSince(letzteUpdateZeit)
        letzteUpdateZeit = jetzt
        
        var neuePeaks: [Double]
        if peakAmplituden.isEmpty || peakAmplituden.count != filteredAmplituden.count {
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
        
        var maxAmplitude: Double = -100.0
        var maxFrequenz: Double = 0.0
        for i in 0..<filteredAmplituden.count {
            if filteredAmplituden[i] > maxAmplitude {
                maxAmplitude = filteredAmplituden[i]
                maxFrequenz = filteredFrequenzen[i]
            }
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.frequenzen = filteredFrequenzen
            self?.amplituden = filteredAmplituden
            self?.peakAmplituden = neuePeaks
            self?.peakFrequenz = maxFrequenz
            self?.peakAmplitude = maxAmplitude
        }
    }
    
    deinit {
        stopAudioEngine()
    }
}

// MARK: - Einstellungen
class EinstellungenManager: ObservableObject {
    @Published var peakHoldAktiv = true
    @Published var glaettungAktiv = true
    @Published var glaettungsStaerke: Double = 0.7
    @Published var peakFallRate: Double = 3.0
    @Published var frequenzmarkerAktiv = true
    @Published var logarithmischeFrequenz = false
}

// MARK: - Visualisierungsmodus
enum VisualisierungsModus: String, CaseIterable {
    case balken = "Balken"
    case linie = "Linie"
    case kombination = "Kombination"
}

// MARK: - Spektrum View
struct SpektrumView: View {
    let frequenzen: [Double]
    let amplituden: [Double]
    let peakAmplituden: [Double]
    let peakFrequenz: Double
    let peakAmplitude: Double
    let modus: VisualisierungsModus
    @ObservedObject var einstellungen: EinstellungenManager
    
    var body: some View {
        Canvas { context, size in
            guard !frequenzen.isEmpty && !amplituden.isEmpty else { return }
            
            let padding: CGFloat = 50
            
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black))
            
            zeichneAchsen(context: context, size: size, padding: padding)
            
            if einstellungen.frequenzmarkerAktiv {
                zeichneFrequenzmarker(context: context, size: size, padding: padding)
            }
            
            switch modus {
            case .balken:
                zeichneBalken(context: context, size: size, padding: padding)
            case .linie:
                zeichneLinie(context: context, size: size, padding: padding, color: .green)
            case .kombination:
                zeichneBalken(context: context, size: size, padding: padding)
                zeichneLinie(context: context, size: size, padding: padding, color: .cyan)
            }
            
            if einstellungen.peakHoldAktiv {
                zeichnePeaks(context: context, size: size, padding: padding)
            }
            
            if peakFrequenz > 0 && peakAmplitude > -90.0 {
                zeichnePeakFrequenzMarker(context: context, size: size, padding: padding)
            }
            
            zeichneBeschriftungen(context: context, size: size, padding: padding)
        }
        .background(Color.black)
    }
    
    private func formatiereFrequenz(_ freq: Double) -> String {
        if freq >= 1000 {
            return String(format: "%.1f kHz", freq / 1000.0)
        } else {
            return String(format: "%.2f kHz", freq / 1000.0)
        }
    }
    
    private func frequenzZuX(_ freq: Double, width: CGFloat, padding: CGFloat) -> CGFloat {
        let drawWidth = width - 2 * padding
        
        if einstellungen.logarithmischeFrequenz {
            let minFreq = 20.0
            let maxFreq = 20000.0
            let logMin = log10(minFreq)
            let logMax = log10(maxFreq)
            let logFreq = log10(max(freq, minFreq))
            let normalized = (logFreq - logMin) / (logMax - logMin)
            return padding + CGFloat(normalized) * drawWidth
        } else {
            let minFreq = frequenzen.first ?? 20.0
            let maxFreq = frequenzen.last ?? 20000.0
            let normalized = (freq - minFreq) / (maxFreq - minFreq)
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
        let markerFrequenzen = [50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 20000.0]
        
        for freq in markerFrequenzen {
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            
            var path = Path()
            path.move(to: CGPoint(x: x, y: size.height - padding))
            path.addLine(to: CGPoint(x: x, y: padding))
            
            context.stroke(path, with: .color(.white.opacity(0.15)), lineWidth: 1)
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
        let backgroundRect = CGRect(
            x: x - textSize.width / 2,
            y: labelY - textSize.height / 2,
            width: textSize.width,
            height: textSize.height
        )
        
        context.fill(
            Path(roundedRect: backgroundRect, cornerRadius: 5),
            with: .color(.black.opacity(0.7))
        )
        
        context.stroke(
            Path(roundedRect: backgroundRect, cornerRadius: 5),
            with: .color(.yellow),
            lineWidth: 1
        )
        
        context.draw(
            Text(frequenzText)
                .foregroundColor(.yellow)
                .font(.system(size: 12, weight: .bold)),
            at: CGPoint(x: x, y: labelY)
        )
    }
    
    private func zeichneBalken(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        guard !amplituden.isEmpty else { return }
        
        let height = size.height - 2 * padding
        let minDb: Double = -100.0
        let maxDb: Double = 0.0
        
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
        let minDb: Double = -100.0
        let maxDb: Double = 0.0
        
        var path = Path()
        
        for (index, amplitude) in amplituden.enumerated() {
            let freq = frequenzen[index]
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            let normalizedHeight = (amplitude - minDb) / (maxDb - minDb)
            let y = size.height - padding - CGFloat(max(0, normalizedHeight)) * height
            
            if index == 0 {
                path.move(to: CGPoint(x: x, y: y))
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
        
        context.stroke(path, with: .color(color), lineWidth: 2)
    }
    
    private func zeichnePeaks(context: GraphicsContext, size: CGSize, padding: CGFloat) {
        guard peakAmplituden.count == amplituden.count else { return }
        
        let height = size.height - 2 * padding
        let minDb: Double = -100.0
        let maxDb: Double = 0.0
        
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
        let frequenzMarker: [(Double, String)] = [
            (50, "50"), (100, "100"), (500, "500"),
            (1000, "1k"), (5000, "5k"), (10000, "10k"), (20000, "20k")
        ]
        
        for (freq, label) in frequenzMarker {
            let x = frequenzZuX(freq, width: size.width, padding: padding)
            let textPosition = CGPoint(x: x, y: size.height - padding + 20)
            context.draw(Text(label).foregroundColor(.white).font(.system(size: 10)), at: textPosition)
        }
        
        let dbMarker = [-80, -60, -40, -20, 0]
        let height = size.height - 2 * padding
        
        for db in dbMarker {
            let normalizedY = (Double(db) - (-100.0)) / (0.0 - (-100.0))
            let y = size.height - padding - CGFloat(normalizedY) * height
            context.draw(Text("\(db) dB").foregroundColor(.white).font(.system(size: 10)),
                       at: CGPoint(x: padding - 30, y: y))
        }
    }
}

// MARK: - Export Manager
class ExportManager {
    static func exportiereAlsCSV(frequenzen: [Double], amplituden: [Double]) -> URL? {
        var csvText = "Frequenz [Hz],Amplitude [dB]\n"
        
        for i in 0..<min(frequenzen.count, amplituden.count) {
            csvText += String(format: "%.2f,%.2f\n", frequenzen[i], amplituden[i])
        }
        
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "spektrum_\(Date().timeIntervalSince1970).csv"
        let fileURL = tempDir.appendingPathComponent(fileName)
        
        do {
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            return fileURL
        } catch {
            print("Fehler beim CSV-Export: \(error)")
            return nil
        }
    }
    
    @MainActor
    static func exportiereAlsPNG(view: some View, size: CGSize) -> URL? {
        let renderer = ImageRenderer(content: view.frame(width: size.width, height: size.height))
        renderer.scale = 2.0
        
        guard let nsImage = renderer.nsImage else { return nil }
        guard let tiffData = nsImage.tiffRepresentation else { return nil }
        guard let bitmapRep = NSBitmapImageRep(data: tiffData) else { return nil }
        guard let pngData = bitmapRep.representation(using: .png, properties: [:]) else { return nil }
        
        let tempDir = FileManager.default.temporaryDirectory
        let fileName = "spektrum_\(Date().timeIntervalSince1970).png"
        let fileURL = tempDir.appendingPathComponent(fileName)
        
        do {
            try pngData.write(to: fileURL)
            return fileURL
        } catch {
            print("Fehler beim PNG-Export: \(error)")
            return nil
        }
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
            HStack {
                Text("Spektrumanalysator Pro")
                    .font(.title)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button(action: { zeigeStimmerkennung.toggle() }) {
                    Image(systemName: "person.wave.2")
                        .font(.title2)
                        .foregroundColor(verificationModel.isVerified ? .green : .primary)
                }
                
                Button(action: { zeigeEinstellungen.toggle() }) {
                    Image(systemName: "gearshape.fill")
                        .font(.title2)
                }
            }
            .padding(.horizontal)
            
            Picker("Visualisierung", selection: $visualisierungsModus) {
                ForEach(VisualisierungsModus.allCases, id: \.self) { modus in
                    Text(modus.rawValue).tag(modus)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 40)
            
            SpektrumView(
                frequenzen: audioManager.frequenzen,
                amplituden: audioManager.amplituden,
                peakAmplituden: audioManager.peakAmplituden,
                peakFrequenz: audioManager.peakFrequenz,
                peakAmplitude: audioManager.peakAmplitude,
                modus: visualisierungsModus,
                einstellungen: einstellungen
            )
            .frame(height: 400)
            .cornerRadius(10)
            .padding(.horizontal)
            
            // Speaker Verification Status
            if audioManager.verificationAktiv {
                HStack(spacing: 15) {
                    Image(systemName: verificationModel.isVerified ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .font(.title)
                        .foregroundColor(verificationModel.isVerified ? .green : .red)
                    
                    VStack(alignment: .leading, spacing: 3) {
                        Text(verificationModel.status)
                            .font(.headline)
                        Text("Confidence: \(Int(verificationModel.confidence * 100))%")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        if verificationModel.continuousLearning && verificationModel.isVerified {
                            Text("🔄 Auto-Learning aktiv")
                                .font(.caption2)
                                .foregroundColor(.blue)
                        }
                    }
                    
                    Spacer()
                    
                    Text("\(verificationModel.userSampleCount) Samples")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .background(Color.gray.opacity(0.2))
                .cornerRadius(10)
            }
            
            HStack(spacing: 20) {
                Button(action: toggleAudioEngine) {
                    HStack {
                        Image(systemName: isRunning ? "stop.circle.fill" : "play.circle.fill")
                        Text(isRunning ? "Stoppen" : "Starten")
                    }
                    .frame(width: 150, height: 40)
                    .background(isRunning ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                
                Button(action: exportiereCSV) {
                    HStack {
                        Image(systemName: "arrow.down.doc")
                        Text("CSV")
                    }
                    .frame(width: 100, height: 40)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .disabled(!isRunning)
                
                Button(action: exportierePNG) {
                    HStack {
                        Image(systemName: "camera")
                        Text("PNG")
                    }
                    .frame(width: 100, height: 40)
                    .background(Color.purple)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .disabled(!isRunning)
            }
            
            if exportErfolg {
                Text(exportNachricht)
                    .foregroundColor(.green)
                    .font(.caption)
            }
            
            HStack(spacing: 30) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("FFT: 2048 Samples")
                        .font(.caption)
                    Text("Sample-Rate: \(Int(audioManager.sampleRate)) Hz")
                        .font(.caption)
                }
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Bereich: 20 Hz - 20 kHz")
                        .font(.caption)
                    Text("Peak: \(audioManager.peakFrequenz > 0 ? formatiereFrequenz(audioManager.peakFrequenz) : "---")")
                        .font(.caption)
                        .foregroundColor(.yellow)
                }
            }
            .foregroundColor(.secondary)
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
            if einstellungen.glaettungAktiv {
                audioManager.glaettungsFaktor = wert
            }
        }
        .onChange(of: einstellungen.peakHoldAktiv) { _, aktiv in
            audioManager.peakHoldAktiv = aktiv
        }
        .onChange(of: einstellungen.peakFallRate) { _, rate in
            audioManager.peakFallRate = rate
        }
        .onAppear {
            audioManager.verificationModel = verificationModel
            verificationModel.loadModel()
        }
    }
    
    private func formatiereFrequenz(_ freq: Double) -> String {
        if freq >= 1000 {
            return String(format: "%.1f kHz", freq / 1000.0)
        } else {
            return String(format: "%.2f kHz", freq / 1000.0)
        }
    }
    
    private func toggleAudioEngine() {
        if isRunning {
            audioManager.stopAudioEngine()
        } else {
            audioManager.startAudioEngine()
        }
        isRunning.toggle()
    }
    
    private func exportiereCSV() {
        if let url = ExportManager.exportiereAlsCSV(
            frequenzen: audioManager.frequenzen,
            amplituden: audioManager.amplituden
        ) {
            NSWorkspace.shared.open(url.deletingLastPathComponent())
            exportNachricht = "CSV exportiert: \(url.lastPathComponent)"
            exportErfolg = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                exportErfolg = false
            }
        }
    }
    
    private func exportierePNG() {
        let spektrumView = SpektrumView(
            frequenzen: audioManager.frequenzen,
            amplituden: audioManager.amplituden,
            peakAmplituden: audioManager.peakAmplituden,
            peakFrequenz: audioManager.peakFrequenz,
            peakAmplitude: audioManager.peakAmplitude,
            modus: visualisierungsModus,
            einstellungen: einstellungen
        )
        
        if let url = ExportManager.exportiereAlsPNG(view: spektrumView, size: CGSize(width: 1200, height: 600)) {
            NSWorkspace.shared.open(url.deletingLastPathComponent())
            exportNachricht = "PNG exportiert: \(url.lastPathComponent)"
            exportErfolg = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                exportErfolg = false
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
            Text("Stimmerkennung")
                .font(.title)
                .fontWeight(.bold)
            
            Divider()
            
            // Status
            VStack(spacing: 10) {
                HStack {
                    Text("User-Samples:")
                    Spacer()
                    Text("\(verificationModel.userSampleCount)")
                        .foregroundColor(.blue)
                }
                
                HStack {
                    Text("Other-Samples:")
                    Spacer()
                    Text("\(verificationModel.otherSampleCount)")
                        .foregroundColor(.orange)
                }
                
                HStack {
                    Text("Training bereit:")
                    Spacer()
                    Text("\(verificationModel.canTrain() ? "✓" : "✗") \(verificationModel.canTrain() ? "Ja" : "Nein")")
                        .foregroundColor(verificationModel.canTrain() ? .green : .orange)
                }
                
                Divider()
                
                HStack {
                    Text("Verification:")
                    Spacer()
                    Toggle("", isOn: $audioManager.verificationAktiv)
                        .disabled(!verificationModel.canTrain())
                }
                
                HStack {
                    Text("Auto-Learning:")
                    Spacer()
                    Toggle("", isOn: $verificationModel.continuousLearning)
                }
                .help("Speichert automatisch neue Samples bei hoher Confidence")
                
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text("Threshold:")
                        Spacer()
                        Text("\(Int(verificationModel.threshold * 100))%")
                    }
                    Slider(value: $verificationModel.threshold, in: 0.3...0.9, step: 0.05)
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
            
            // Training
            VStack(spacing: 15) {
                Text("Training")
                    .font(.headline)
                
                Text("Sprich 10 Sekunden um deine Stimme zu lernen")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                Button(action: startUserEnrollment) {
                    HStack {
                        Image(systemName: "mic.fill")
                        Text("Meine Stimme lernen")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(audioManager.enrollmentMode != .none)
                
                Button(action: startOtherEnrollment) {
                    HStack {
                        Image(systemName: "person.2")
                        Text("Andere Stimmen lernen")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.orange)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(audioManager.enrollmentMode != .none)
                
                if audioManager.enrollmentMode != .none {
                    VStack(spacing: 10) {
                        HStack {
                            Image(systemName: "record.circle")
                                .foregroundColor(.red)
                            Text("Aufnahme läuft: \(enrollmentType)")
                                .foregroundColor(.red)
                        }
                        
                        Text("\(Int(enrollmentProgress * 10)) / 10 Sekunden")
                            .font(.caption)
                        
                        ProgressView(value: enrollmentProgress, total: 1.0)
                            .progressViewStyle(.linear)
                        
                        Button("Abbrechen") {
                            stopEnrollment()
                        }
                        .foregroundColor(.red)
                        .padding(.top, 5)
                    }
                    .padding()
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(10)
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
            
            // Management
            VStack(spacing: 15) {
                Text("Verwaltung")
                    .font(.headline)
                
                HStack(spacing: 15) {
                    Button(action: {
                        verificationModel.reset()
                    }) {
                        VStack {
                            Image(systemName: "trash")
                                .font(.title2)
                            Text("Zurücksetzen")
                                .font(.caption)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.red.opacity(0.8))
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    
                    Button(action: {
                        verificationModel.saveModel()
                    }) {
                        VStack {
                            Image(systemName: "square.and.arrow.down")
                                .font(.title2)
                            Text("Speichern")
                                .font(.caption)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
            
            Spacer()
            
            Button("Schließen") {
                dismiss()
            }
            .buttonStyle(.borderedProminent)
            .padding(.bottom)
        }
        .padding()
        .frame(width: 550, height: 800)
    }
    
    private func startUserEnrollment() {
        enrollmentType = "Deine Stimme"
        enrollmentProgress = 0.0
        enrollmentStartTime = Date()
        audioManager.enrollmentMode = .user
        audioManager.verificationAktiv = false
        
        enrollmentTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [self] timer in
            guard let startTime = enrollmentStartTime else { return }
            
            let elapsed = Date().timeIntervalSince(startTime)
            let progress = min(elapsed / 10.0, 1.0)
            
            DispatchQueue.main.async {
                self.enrollmentProgress = progress
            }
            
            if elapsed >= 10.0 {
                DispatchQueue.main.async {
                    self.stopEnrollment()
                }
            }
        }
    }
    
    private func startOtherEnrollment() {
        enrollmentType = "Andere Stimmen"
        enrollmentProgress = 0.0
        enrollmentStartTime = Date()
        audioManager.enrollmentMode = .other
        audioManager.verificationAktiv = false
        
        enrollmentTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [self] timer in
            guard let startTime = enrollmentStartTime else { return }
            
            let elapsed = Date().timeIntervalSince(startTime)
            let progress = min(elapsed / 10.0, 1.0)
            
            DispatchQueue.main.async {
                self.enrollmentProgress = progress
            }
            
            if elapsed >= 10.0 {
                DispatchQueue.main.async {
                    self.stopEnrollment()
                }
            }
        }
    }
    
    private func stopEnrollment() {
        enrollmentTimer?.invalidate()
        enrollmentTimer = nil
        audioManager.enrollmentMode = .none
        enrollmentProgress = 0.0
        enrollmentStartTime = nil
        
        // Automatisch speichern nach Training
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
            Text("Einstellungen")
                .font(.title)
                .fontWeight(.bold)
            
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
                }
            }
            .formStyle(.grouped)
            
            Button("Schließen") {
                dismiss()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .frame(width: 500, height: 500)
    }
}

// MARK: - App Entry Point
@main
struct SpektrumAnalysatorApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
    }
}
