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


package roratelimit

import (
	"fmt"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func ExampleNewRateLimiter_basic() {
	// Basic rate limiting: 3 items per second
	observable := ro.Pipe1(
		ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
		NewRateLimiter[int](3, time.Second, func(v int) string {
			return "default"
		}),
	)

	values, err := ro.Collect(observable)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Rate limited values: %v\n", values)
	// Output: Rate limited values: [1 2 3]
}

func ExampleNewRateLimiter_userBased() {
	type UserRequest struct {
		UserID string
		Action string
		Data   string
	}

	// Rate limit by user ID: 2 requests per minute per user
	observable := ro.Pipe1(
		ro.Just(
			UserRequest{UserID: "user1", Action: "login", Data: "data1"},
			UserRequest{UserID: "user2", Action: "login", Data: "data2"},
			UserRequest{UserID: "user1", Action: "logout", Data: "data3"},
			UserRequest{UserID: "user1", Action: "profile", Data: "data4"},
			UserRequest{UserID: "user3", Action: "login", Data: "data5"},
		),
		NewRateLimiter[UserRequest](2, time.Minute, func(req UserRequest) string {
			return req.UserID
		}),
	)

	values, err := ro.Collect(observable)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("User-based rate limited requests: %d\n", len(values))
	// Output: User-based rate limited requests: 4
}

func ExampleNewRateLimiter_ipBased() {
	type APIRequest struct {
		IPAddress string
		Endpoint  string
		Method    string
	}

	// Rate limit by IP address: 5 requests per minute per IP
	observable := ro.Pipe1(
		ro.Just(
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/users", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.2", Endpoint: "/api/posts", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/users", Method: "POST"},
			APIRequest{IPAddress: "192.168.1.3", Endpoint: "/api/comments", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/posts", Method: "PUT"},
		),
		NewRateLimiter[APIRequest](5, time.Minute, func(req APIRequest) string {
			return req.IPAddress
		}),
	)

	values, err := ro.Collect(observable)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("IP-based rate limited requests: %d\n", len(values))
	// Output: IP-based rate limited requests: 5
}

func ExampleNewRateLimiter_endpointBased() {
	type APIRequest struct {
		IPAddress string
		Endpoint  string
		Method    string
	}

	// Rate limit by endpoint: 2 requests per second per endpoint
	observable := ro.Pipe1(
		ro.Just(
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/users", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.2", Endpoint: "/api/posts", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/users", Method: "POST"},
			APIRequest{IPAddress: "192.168.1.3", Endpoint: "/api/comments", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/posts", Method: "PUT"},
		),
		NewRateLimiter[APIRequest](2, time.Second, func(req APIRequest) string {
			return req.Endpoint
		}),
	)

	values, err := ro.Collect(observable)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Endpoint-based rate limited requests: %d\n", len(values))
	// Output: Endpoint-based rate limited requests: 5
}

func ExampleNewRateLimiter_compositeKey() {
	type APIRequest struct {
		IPAddress string
		Endpoint  string
		Method    string
	}

	// Rate limit by IP + endpoint combination: 3 requests per minute per IP-endpoint pair
	observable := ro.Pipe1(
		ro.Just(
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/users", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.2", Endpoint: "/api/users", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/posts", Method: "GET"},
			APIRequest{IPAddress: "192.168.1.1", Endpoint: "/api/users", Method: "POST"},
			APIRequest{IPAddress: "192.168.1.2", Endpoint: "/api/posts", Method: "GET"},
		),
		NewRateLimiter[APIRequest](3, time.Minute, func(req APIRequest) string {
			return req.IPAddress + ":" + req.Endpoint
		}),
	)

	values, err := ro.Collect(observable)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Composite key rate limited requests: %d\n", len(values))
	// Output: Composite key rate limited requests: 5
}

func ExampleNewRateLimiter_realWorld() {
	type LogEntry struct {
		UserID    string
		Action    string
		Timestamp time.Time
		Message   string
	}

	// Simulate log processing with rate limiting
	// Limit: 100 logs per minute per user
	logs := []LogEntry{
		{UserID: "user1", Action: "login", Timestamp: time.Now(), Message: "User logged in"},
		{UserID: "user2", Action: "login", Timestamp: time.Now(), Message: "User logged in"},
		{UserID: "user1", Action: "logout", Timestamp: time.Now(), Message: "User logged out"},
		{UserID: "user3", Action: "login", Timestamp: time.Now(), Message: "User logged in"},
		{UserID: "user1", Action: "profile", Timestamp: time.Now(), Message: "Profile updated"},
	}

	observable := ro.Pipe1(
		ro.Just(logs...),
		NewRateLimiter[LogEntry](100, time.Minute, func(log LogEntry) string {
			return log.UserID
		}),
	)

	values, err := ro.Collect(observable)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Processed log entries: %d\n", len(values))
	// Output: Processed log entries: 5
}

func TestNewRateLimiter_examples(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test basic rate limiting
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Range(0, 10),
			NewRateLimiter[int64](5, time.Second, func(v int64) string {
				return "default"
			}),
		),
	)
	is.NoError(err)
	is.Len(values, 5)
	is.Equal([]int64{0, 1, 2, 3, 4}, values)

	// Test user-based rate limiting
	type UserRequest struct {
		UserID string
		Action string
	}

	userRequests := []UserRequest{
		{UserID: "user1", Action: "login"},
		{UserID: "user2", Action: "login"},
		{UserID: "user1", Action: "logout"},
		{UserID: "user3", Action: "login"},
		{UserID: "user1", Action: "profile"},
	}

	userValues, err := ro.Collect(
		ro.Pipe1(
			ro.Just(userRequests...),
			NewRateLimiter[UserRequest](3, time.Minute, func(req UserRequest) string {
				return req.UserID
			}),
		),
	)
	is.NoError(err)
	is.Len(userValues, 5) // All requests should pass as they're from different users or within limit

	// Test IP-based rate limiting
	type APIRequest struct {
		IPAddress string
		Endpoint  string
	}

	apiRequests := []APIRequest{
		{IPAddress: "192.168.1.1", Endpoint: "/api/users"},
		{IPAddress: "192.168.1.2", Endpoint: "/api/posts"},
		{IPAddress: "192.168.1.1", Endpoint: "/api/users"},
		{IPAddress: "192.168.1.3", Endpoint: "/api/comments"},
		{IPAddress: "192.168.1.1", Endpoint: "/api/posts"},
	}

	apiValues, err := ro.Collect(
		ro.Pipe1(
			ro.Just(apiRequests...),
			NewRateLimiter[APIRequest](2, time.Second, func(req APIRequest) string {
				return req.IPAddress
			}),
		),
	)
	is.NoError(err)
	is.Len(apiValues, 4) // Rate limited: 2 per IP per second, so some requests are filtered
}
