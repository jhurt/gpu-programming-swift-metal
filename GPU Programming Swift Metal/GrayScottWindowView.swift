// Copyright 2026 Jason Hurt
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import SwiftUI

struct GrayScottWindowView: View {
    // listen for @Published changes
    @ObservedObject var controller = GrayScottController.shared
    
    @State private var viewID = UUID()
    
    var body: some View {
        if let _ = controller.grayScott {
            GrayScottMetalView()
                .navigationTitle("Gray-Scott Simulation")
                .id(viewID)
                .onAppear {
                    viewID = UUID() // force recreate the view
                }
        } else {
            Text("Simulation Not Running")
        }
    }
}
