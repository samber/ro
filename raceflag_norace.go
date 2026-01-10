//go:build !race

package ro

// RaceEnabled is false when the test binary is NOT built with the race detector.
const RaceEnabled = false
