//
//  microphone_FFTApp.swift
//  microphone_FFT
//
//  Created by Thorsten Rueben on 04.10.25.
//

import SwiftUI
import CoreData

@main
struct microphone_FFTApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
