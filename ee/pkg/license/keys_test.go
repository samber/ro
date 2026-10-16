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
	"crypto/elliptic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateKeys(t *testing.T) {
	privateKey, publicKey, err := GenerateKeys()
	require.NoError(t, err)
	assert.NotNil(t, privateKey)
	assert.NotNil(t, publicKey)

	// Verify the keys are using the correct curve
	assert.Equal(t, Curve, privateKey.Curve)
	assert.Equal(t, Curve, publicKey.Curve)

	// Verify the public key matches the private key's public key
	assert.True(t, publicKey.Equal(&privateKey.PublicKey))
}

func TestUnmarshalPrivateKey(t *testing.T) {
	// Generate a test key pair
	privateKey, _, err := GenerateKeys()
	require.NoError(t, err)

	// Encode the private key
	privateKeyPEM, err := MarshalPrivateKey(privateKey)
	require.NoError(t, err)

	tests := []struct {
		name    string
		input   []byte
		wantErr bool
	}{
		{
			name:    "valid private key",
			input:   privateKeyPEM,
			wantErr: false,
		},
		{
			name:    "invalid PEM",
			input:   []byte("invalid-pem-data"),
			wantErr: true,
		},
		{
			name:    "empty data",
			input:   []byte{},
			wantErr: true,
		},
		{
			name:    "nil data",
			input:   nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parsedKey, err := UnmarshalPrivateKey(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, parsedKey)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, parsedKey)
				assert.Equal(t, Curve, parsedKey.Curve)
			}
		})
	}
}

func TestUnmarshalPublicKey(t *testing.T) {
	// Generate a test key pair
	_, publicKey, err := GenerateKeys()
	require.NoError(t, err)

	// Encode the public key
	publicKeyPEM, err := MarshalPublicKey(publicKey)
	require.NoError(t, err)

	tests := []struct {
		name    string
		input   []byte
		wantErr bool
	}{
		{
			name:    "valid public key",
			input:   publicKeyPEM,
			wantErr: false,
		},
		{
			name:    "invalid PEM",
			input:   []byte("invalid-pem-data"),
			wantErr: true,
		},
		{
			name:    "empty data",
			input:   []byte{},
			wantErr: true,
		},
		{
			name:    "nil data",
			input:   nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parsedKey, err := UnmarshalPublicKey(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, parsedKey)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, parsedKey)
				assert.Equal(t, Curve, parsedKey.Curve)
			}
		})
	}
}

func TestMarshalPrivateKey(t *testing.T) {
	// Generate a test private key
	privateKey, _, err := GenerateKeys()
	require.NoError(t, err)

	// Encode the private key
	encoded, err := MarshalPrivateKey(privateKey)
	require.NoError(t, err)
	assert.NotEmpty(t, encoded)

	// Verify the encoded key can be parsed back
	parsedKey, err := UnmarshalPrivateKey(encoded)
	require.NoError(t, err)
	assert.True(t, privateKey.Equal(parsedKey))
}

func TestMarshalPublicKey(t *testing.T) {
	// Generate a test key pair
	_, publicKey, err := GenerateKeys()
	require.NoError(t, err)

	// Encode the public key
	encoded, err := MarshalPublicKey(publicKey)
	require.NoError(t, err)
	assert.NotEmpty(t, encoded)

	// Verify the encoded key can be parsed back
	parsedKey, err := UnmarshalPublicKey(encoded)
	require.NoError(t, err)
	assert.True(t, publicKey.Equal(parsedKey))
}

func TestEncodeDataWithPrivateKey(t *testing.T) {
	// Generate test keys
	privateKey, _, err := GenerateKeys()
	require.NoError(t, err)

	testData := []byte("test data to encode")

	// Encode the data
	encoded, err := EncodeDataWithPrivateKey(testData, privateKey)
	require.NoError(t, err)
	assert.NotEmpty(t, encoded)
}

func TestDecodeDataWithPublicKey(t *testing.T) {
	// Generate test keys
	privateKey, publicKey, err := GenerateKeys()
	require.NoError(t, err)

	testData := []byte("test data to encode and decode")

	// Encode the data
	encoded, err := EncodeDataWithPrivateKey(testData, privateKey)
	require.NoError(t, err)

	// Encode the public key for decoding
	publicKeyPEM, err := MarshalPublicKey(publicKey)
	require.NoError(t, err)

	// Decode the data
	decoded, err := DecodeDataWithPublicKey(encoded, publicKeyPEM)
	require.NoError(t, err)
	assert.Equal(t, testData, decoded)

	// Test with invalid encoded data
	_, err = DecodeDataWithPublicKey("invalid-base64", publicKeyPEM)
	assert.Error(t, err)
}

func TestSignData(t *testing.T) {
	// Generate test keys
	privateKey, _, err := GenerateKeys()
	require.NoError(t, err)

	testData := []byte("test data to sign")

	// Sign the data
	container, err := signData(testData, privateKey)
	require.NoError(t, err)
	assert.NotNil(t, container)
	assert.NotEmpty(t, container.Data)
	assert.NotEmpty(t, container.Signature)
	assert.Equal(t, testData, container.Data)
}

func TestVerifyDataSignature(t *testing.T) {
	// Generate test keys
	privateKey, publicKey, err := GenerateKeys()
	require.NoError(t, err)

	testData := []byte("test data to sign and verify")

	// Sign the data
	container, err := signData(testData, privateKey)
	require.NoError(t, err)

	// Encode the public key for verification
	publicKeyPEM, err := MarshalPublicKey(publicKey)
	require.NoError(t, err)

	t.Run("valid signature", func(t *testing.T) {
		err := verifyDataSignature(container, publicKeyPEM)
		assert.NoError(t, err)
	})

	t.Run("tampered data", func(t *testing.T) {
		tamperedContainer := &signedContainer{
			Data:      []byte("tampered data"),
			Signature: container.Signature,
		}

		err := verifyDataSignature(tamperedContainer, publicKeyPEM)
		assert.Error(t, err)
	})

	t.Run("tampered signature", func(t *testing.T) {
		tamperedContainer := &signedContainer{
			Data:      container.Data,
			Signature: []byte("tampered signature"),
		}

		err := verifyDataSignature(tamperedContainer, publicKeyPEM)
		assert.Error(t, err)
	})
}

func TestCurveConstant(t *testing.T) {
	assert.Equal(t, elliptic.P521(), Curve)
}
