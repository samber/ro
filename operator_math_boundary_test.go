package ro

import (
	"math"
	"testing"
)

func TestMaxPow10ChunkValue(t *testing.T) {
	if maxPow10Chunk != 308 {
		t.Fatalf("expected maxPow10Chunk == 308, got %d", maxPow10Chunk)
	}

	v := math.Pow10(maxPow10Chunk)
	if math.IsInf(v, 0) || math.IsNaN(v) {
		t.Fatalf("expected math.Pow10(%d) to be finite, got %v", maxPow10Chunk, v)
	}

	v2 := math.Pow10(maxPow10Chunk + 1)
	if !math.IsInf(v2, 1) {
		t.Fatalf("expected math.Pow10(%d) to overflow to +Inf, got %v", maxPow10Chunk+1, v2)
	}
}

func TestChunkCountComputation(t *testing.T) {
	// a moderately large precision should require multiple chunks
	places := 1000
	chunkCount := (places + maxPow10Chunk - 1) / maxPow10Chunk
	if chunkCount <= 1 {
		t.Fatalf("expected chunkCount>1 for places=%d, got %d", places, chunkCount)
	}
	if chunkCount > maxPow10ChunkCount {
		t.Fatalf("expected chunkCount <= maxPow10ChunkCount for places=%d, got %d", places, chunkCount)
	}

	// a huge precision should exceed the chunk count cap
	largePlaces := maxPow10Chunk * (maxPow10ChunkCount + 1)
	chunkCount2 := (largePlaces + maxPow10Chunk - 1) / maxPow10Chunk
	if chunkCount2 <= maxPow10ChunkCount {
		t.Fatalf("expected chunkCount2 > maxPow10ChunkCount for largePlaces, got %d", chunkCount2)
	}
}
