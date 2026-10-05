// Copyright 2025 samber.
//
// Licensed as an Enterprise License (the "License"); you may not use
// this file except in compliance with the License. You may obtain
// a copy of the License at:
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.ee.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


package rolicense

import (
	"regexp"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestGetInstanceID(t *testing.T) {
	// Test that GetInstanceID returns a valid ID
	instanceID1 := GetInstanceID()
	assert.NotEmpty(t, instanceID1)

	// Test that the ID is a valid hex string (16 bytes = 32 hex characters)
	hexPattern := regexp.MustCompile(`^[0-9a-f]{32}$`)
	assert.True(t, hexPattern.MatchString(instanceID1), "GetInstanceID() returned invalid hex string: %s", instanceID1)

	// Test that subsequent calls return the same ID (caching)
	instanceID2 := GetInstanceID()
	assert.Equal(t, instanceID1, instanceID2, "GetInstanceID() returned different IDs on subsequent calls")

	t.Run("id is a hex string or the timestamp fallback", func(t *testing.T) {
		// Test that different test runs (simulated by resetting the instanceIDOnce)
		// would generate different IDs
		// Note: This is a bit tricky to test since we can't easily reset sync.Once
		// But we can test the fallback mechanism by checking the pattern

		// Get the current instance ID
		currentID := GetInstanceID()

		// Test that it's either a hex string (normal case) or a timestamp-based ID (fallback)
		hexPattern := regexp.MustCompile(`^[0-9a-f]{32}$`)
		timestampPattern := regexp.MustCompile(`^instance-\d+$`)

		assert.True(t, hexPattern.MatchString(currentID) || timestampPattern.MatchString(currentID),
			"GetInstanceID() returned invalid format: %s", currentID)
	})

	t.Run("id has the expected length and characters", func(t *testing.T) {
		// Test that the instance ID has the expected format
		instanceID := GetInstanceID()

		// Should be exactly 32 characters (16 bytes in hex)
		assert.Equal(t, 32, len(instanceID), "GetInstanceID() returned ID with wrong length")

		// Should only contain hexadecimal characters
		for _, char := range instanceID {
			assert.True(t, (char >= '0' && char <= '9') || (char >= 'a' && char <= 'f'),
				"GetInstanceID() returned ID with invalid character: %c", char)
		}
	})
}

func TestGetInstanceID_concurrency(t *testing.T) {
	// Test that GetInstanceID is thread-safe
	const numGoroutines = 10
	var wg sync.WaitGroup
	results := make([]string, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			results[index] = GetInstanceID()
		}(i)
	}

	wg.Wait()

	// All results should be the same
	firstResult := results[0]
	for i, result := range results {
		assert.Equal(t, firstResult, result, "GetInstanceID() returned different results in concurrent calls at index %d", i)
	}
}

func TestGetInstanceID_performance(t *testing.T) {
	// Test that GetInstanceID is fast (cached after first call)
	start := time.Now()

	// First call (should be slower due to generation)
	_ = GetInstanceID()
	firstCallDuration := time.Since(start)

	// Second call (should be fast due to caching)
	start = time.Now()
	_ = GetInstanceID()
	secondCallDuration := time.Since(start)

	// The second call should be significantly faster
	if secondCallDuration >= firstCallDuration {
		t.Logf("Warning: Second call to GetInstanceID() was not faster than first call")
		t.Logf("First call duration: %v", firstCallDuration)
		t.Logf("Second call duration: %v", secondCallDuration)
	}
}
