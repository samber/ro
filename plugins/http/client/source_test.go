// Copyright 2025 samber.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.apache.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rohttpclient

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func TestOperatorSpecialHTTPRequest(t *testing.T) {
	t.Parallel()
	// testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "test")
	}))

	req, _ := http.NewRequest(http.MethodGet, server.URL, nil)

	values, err := ro.Collect(
		HTTPRequest(req, http.DefaultClient),
	)
	is.Len(values, 1)
	is.Equal(http.StatusOK, values[0].StatusCode)
	b, _ := io.ReadAll(values[0].Body)
	values[0].Body.Close()
	is.Equal("test\n", string(b))
	is.Nil(err)

	req, _ = http.NewRequest(http.MethodGet, "http://invalid.url", nil)

	values, err = ro.Collect(
		HTTPRequest(req, http.DefaultClient),
	)
	is.Equal([]*http.Response{}, values)
	is.ErrorContains(err, "Get \"http://invalid.url\": dial tcp: lookup invalid.url")

	server.Close()

	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintln(w, "not found")
	}))

	req, _ = http.NewRequest(http.MethodGet, server.URL, nil)

	values, err = ro.Collect(
		HTTPRequest(req, http.DefaultClient),
	)
	is.Len(values, 1)
	is.Equal(http.StatusNotFound, values[0].StatusCode)
	b, _ = io.ReadAll(values[0].Body)
	values[0].Body.Close()
	is.Equal("not found\n", string(b))
	is.Nil(err)

	server.Close()

	// For some reason, removing the following line causes
	// the test to fail (see goleak).
	// See https://github.com/uber-go/goleak/issues/102
	http.DefaultClient.CloseIdleConnections()
}

func TestOperatorSpecialHTTPRequestJSON(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test successful JSON string parsing
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `"test"`)
	}))

	req, _ := http.NewRequest(http.MethodGet, server.URL, nil)

	values, err := ro.Collect(
		HTTPRequestJSON[string](req, http.DefaultClient),
	)
	is.Equal([]string{"test"}, values)
	is.Nil(err)

	server.Close()

	// Test successful JSON object parsing
	type TestResponse struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	}

	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"message": "hello", "code": 200}`)
	}))

	req, _ = http.NewRequest(http.MethodGet, server.URL, nil)

	objValues, err := ro.Collect(
		HTTPRequestJSON[TestResponse](req, http.DefaultClient),
	)
	is.Equal([]TestResponse{{Message: "hello", Code: 200}}, objValues)
	is.Nil(err)

	server.Close()

	// Test invalid JSON
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `invalid json`)
	}))

	req, _ = http.NewRequest(http.MethodGet, server.URL, nil)

	values, err = ro.Collect(
		HTTPRequestJSON[string](req, http.DefaultClient),
	)
	is.Equal([]string{}, values)
	is.Contains(err.Error(), "invalid character")

	server.Close()

	// Test network error
	req, _ = http.NewRequest(http.MethodGet, "http://invalid.url", nil)

	values, err = ro.Collect(
		HTTPRequestJSON[string](req, http.DefaultClient),
	)
	is.Equal([]string{}, values)
	is.ErrorContains(err, "Get \"http://invalid.url\": dial tcp: lookup invalid.url")

	// For some reason, removing the following line causes
	// the test to fail (see goleak).
	// See https://github.com/uber-go/goleak/issues/102
	http.DefaultClient.CloseIdleConnections()

	t.Run("different types", func(t *testing.T) {
		is := assert.New(t)

		// Test with int
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, `42`)
		}))
		defer server.Close()

		req, _ := http.NewRequest(http.MethodGet, server.URL, nil)

		intValues, err := ro.Collect(
			HTTPRequestJSON[int](req, http.DefaultClient),
		)
		is.Equal([]int{42}, intValues)
		is.Nil(err)

		// Test with float
		server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, `3.14`)
		}))
		defer server2.Close()

		req2, _ := http.NewRequest(http.MethodGet, server2.URL, nil)

		floatValues, err := ro.Collect(
			HTTPRequestJSON[float64](req2, http.DefaultClient),
		)
		is.Equal([]float64{3.14}, floatValues)
		is.Nil(err)

		// Test with boolean
		server3 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, `true`)
		}))
		defer server3.Close()

		req3, _ := http.NewRequest(http.MethodGet, server3.URL, nil)

		boolValues, err := ro.Collect(
			HTTPRequestJSON[bool](req3, http.DefaultClient),
		)
		is.Equal([]bool{true}, boolValues)
		is.Nil(err)

		http.DefaultClient.CloseIdleConnections()
	})

	t.Run("slice", func(t *testing.T) {
		is := assert.New(t)

		// Test with slice
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, `["apple", "banana", "cherry"]`)
		}))
		defer server.Close()

		req, _ := http.NewRequest(http.MethodGet, server.URL, nil)

		sliceValues, err := ro.Collect(
			HTTPRequestJSON[[]string](req, http.DefaultClient),
		)
		is.Equal([][]string{{"apple", "banana", "cherry"}}, sliceValues)
		is.Nil(err)

		http.DefaultClient.CloseIdleConnections()
	})

	t.Run("nested struct", func(t *testing.T) {
		is := assert.New(t)

		type Address struct {
			Street string `json:"street"`
			City   string `json:"city"`
		}

		type User struct {
			Name    string  `json:"name"`
			Age     int     `json:"age"`
			Address Address `json:"address"`
		}

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, `{
			"name": "John Doe",
			"age": 30,
			"address": {
				"street": "123 Main St",
				"city": "New York"
			}
		}`)
		}))
		defer server.Close()

		req, _ := http.NewRequest(http.MethodGet, server.URL, nil)

		userValues, err := ro.Collect(
			HTTPRequestJSON[User](req, http.DefaultClient),
		)
		expected := User{
			Name: "John Doe",
			Age:  30,
			Address: Address{
				Street: "123 Main St",
				City:   "New York",
			},
		}
		is.Equal([]User{expected}, userValues)
		is.Nil(err)

		http.DefaultClient.CloseIdleConnections()
	})

	t.Run("HTTP error status is not a stream error", func(t *testing.T) {
		is := assert.New(t)

		// Test that HTTP status codes >= 400 are not considered errors by HTTPRequestJSON
		// (the JSON parsing still happens)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintln(w, `{"error": "not found", "code": 404}`)
		}))
		defer server.Close()

		type ErrorResponse struct {
			Error string `json:"error"`
			Code  int    `json:"code"`
		}

		req, _ := http.NewRequest(http.MethodGet, server.URL, nil)

		errorValues, err := ro.Collect(
			HTTPRequestJSON[ErrorResponse](req, http.DefaultClient),
		)
		expected := ErrorResponse{Error: "not found", Code: 404}
		is.Equal([]ErrorResponse{expected}, errorValues)
		is.Nil(err)

		http.DefaultClient.CloseIdleConnections()
	})

	t.Run("nil client falls back to default client", func(t *testing.T) {
		is := assert.New(t)

		// Test that nil client uses http.DefaultClient
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, `"default client test"`)
		}))
		defer server.Close()

		req, _ := http.NewRequest(http.MethodGet, server.URL, nil)

		values, err := ro.Collect(
			HTTPRequestJSON[string](req, nil), // nil client
		)
		is.Equal([]string{"default client test"}, values)
		is.Nil(err)

		http.DefaultClient.CloseIdleConnections()
	})
}
