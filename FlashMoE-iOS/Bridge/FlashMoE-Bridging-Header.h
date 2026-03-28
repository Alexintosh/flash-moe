/*
 * FlashMoE-Bridging-Header.h — Exposes C APIs to Swift
 *
 * Setup in Xcode:
 *   1. Build the Rust static lib:
 *      cd flash-swarm && cargo build --release --target aarch64-apple-ios -p flashswarm-ios
 *   2. In Xcode target Build Settings:
 *      - "Other Linker Flags": add path to libflashswarm_ios.a
 *      - "Header Search Paths": add flash-swarm/crates/flashswarm-ios/include
 *      - Link frameworks: Metal, MetalPerformanceShaders, Accelerate
 */

#import "FlashMoEEngine.h"

// Flash-Swarm worker API (Rust static library)
// Requires libflashswarm_ios.a to be linked.
#include "flashswarm_ios.h"
