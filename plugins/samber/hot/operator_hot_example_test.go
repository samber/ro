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


package rohot

import (
	"context"
	"fmt"
	"time"

	"github.com/samber/hot"
	"github.com/samber/lo"
	"github.com/samber/ro"
)

// User represents a user in our system
type User struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	Age  int    `json:"age"`
}

// UserService simulates a service that fetches users
type UserService struct {
	users map[string]User
}

func NewUserService() *UserService {
	return &UserService{
		users: map[string]User{
			"user1": {ID: "user1", Name: "Alice", Age: 30},
			"user2": {ID: "user2", Name: "Bob", Age: 25},
			"user3": {ID: "user3", Name: "Charlie", Age: 35},
			"user4": {ID: "user4", Name: "Diana", Age: 28},
		},
	}
}

func (s *UserService) GetUser(id string) (User, error) {
	// Simulate network delay
	time.Sleep(10 * time.Millisecond)

	if user, exists := s.users[id]; exists {
		return user, nil
	}
	return User{}, fmt.Errorf("user not found: %s", id)
}

func ExampleGetOrFetch() {
	// Create a hot cache for users with LRU eviction and 1000 capacity
	cache := hot.NewHotCache[string, User](hot.LRU, 1000).
		WithTTL(5 * time.Minute).
		Build()

	// Populate cache with some initial data
	cache.Set("user1", User{ID: "user1", Name: "Alice", Age: 30})
	cache.Set("user2", User{ID: "user2", Name: "Bob", Age: 25})

	// Create observable with user IDs
	observable := ro.Pipe1(
		ro.Just("user1", "user2", "user3", "user4"),
		GetOrFetch(cache),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(tuple lo.Tuple2[User, bool]) {
				user, found := tuple.Unpack()
				if found {
					fmt.Printf("Found in cache: %s (%s)\n", user.Name, user.ID)
				} else {
					id := user.ID
					if id == "" {
						id = user.Name
					}
					fmt.Printf("Not in cache: %q\n", id)
				}
			},
			func(err error) {
				fmt.Printf("Error: %v\n", err)
			},
			func() {
				fmt.Println("Completed")
			},
		),
	)
	defer subscription.Unsubscribe()

	// Output:
	// Found in cache: Alice (user1)
	// Found in cache: Bob (user2)
	// Not in cache: ""
	// Not in cache: ""
	// Completed
}

func ExampleGetOrFetchOrSkip() {
	// Create a hot cache for users
	cache := hot.NewHotCache[string, User](hot.LRU, 1000).
		WithTTL(5 * time.Minute).
		Build()

	// Populate cache with some data
	cache.Set("user1", User{ID: "user1", Name: "Alice", Age: 30})
	cache.Set("user3", User{ID: "user3", Name: "Charlie", Age: 35})

	// Create observable that only emits users found in cache
	observable := ro.Pipe1(
		ro.Just("user1", "user2", "user3", "user4"),
		GetOrFetchOrSkip(cache),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(user User) {
				fmt.Printf("Found: %s (%s)\n", user.Name, user.ID)
			},
			func(err error) {
				fmt.Printf("Error: %v\n", err)
			},
			func() {
				fmt.Println("Completed")
			},
		),
	)
	defer subscription.Unsubscribe()

	// Output:
	// Found: Alice (user1)
	// Found: Charlie (user3)
	// Completed
}

func ExampleGetOrFetchOrError() {
	// Create a hot cache for users
	cache := hot.NewHotCache[string, User](hot.LRU, 1000).
		WithTTL(5 * time.Minute).
		Build()

	// Populate cache with some data
	cache.Set("user1", User{ID: "user1", Name: "Alice", Age: 30})

	// Create observable that emits errors for missing users
	observable := ro.Pipe1(
		ro.Just("user1", "user2", "user3"),
		GetOrFetchOrError(cache),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(user User) {
				fmt.Printf("Found: %s (%s)\n", user.Name, user.ID)
			},
			func(err error) {
				fmt.Printf("Error: %v\n", err)
			},
			func() {
				fmt.Println("Completed")
			},
		),
	)
	defer subscription.Unsubscribe()

	// Output:
	// Found: Alice (user1)
	// Error: rohot.GetOrFetchOrError: not found
}

func ExampleGetOrFetchMany() {
	// Create a hot cache for users
	cache := hot.NewHotCache[string, User](hot.LRU, 1000).
		WithTTL(5 * time.Minute).
		Build()

	// Populate cache with some data
	cache.Set("user1", User{ID: "user1", Name: "Alice", Age: 30})
	cache.Set("user2", User{ID: "user2", Name: "Bob", Age: 25})
	cache.Set("user3", User{ID: "user3", Name: "Charlie", Age: 35})

	// Create observable with batches of user IDs
	observable := ro.Pipe1(
		ro.Just(
			[]string{"user1", "user2"},
			[]string{"user3", "user4"},
			[]string{"user1", "user3", "user5"},
		),
		GetOrFetchMany(cache),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(users map[string]User) {
				fmt.Printf("Batch found: %d users\n", len(users))
			},
			func(err error) {
				fmt.Printf("Error: %v\n", err)
			},
			func() {
				fmt.Println("Completed")
			},
		),
	)
	defer subscription.Unsubscribe()

	// Output:
	// Batch found: 2 users
	// Batch found: 1 users
	// Batch found: 2 users
	// Completed
}

func ExampleGetOrFetch_withContext() {
	// Create a hot cache for users
	cache := hot.NewHotCache[string, User](hot.LRU, 1000).
		WithTTL(5 * time.Minute).
		Build()

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Populate cache
	cache.Set("user1", User{ID: "user1", Name: "Alice", Age: 30})

	// Create observable
	observable := ro.Pipe1(
		ro.Just("user1", "user2"),
		GetOrFetch(cache),
	)

	subscription := observable.SubscribeWithContext(
		ctx,
		ro.NewObserverWithContext(
			func(ctx context.Context, tuple lo.Tuple2[User, bool]) {
				user, found := tuple.Unpack()
				if found {
					fmt.Printf("Found: %s\n", user.Name)
				} else {
					id := user.ID
					if id == "" {
						id = user.Name
					}
					fmt.Printf("Not found: %q\n", id)
				}
			},
			func(ctx context.Context, err error) {
				fmt.Printf("Error: %v\n", err)
			},
			func(ctx context.Context) {
				fmt.Println("Completed")
			},
		),
	)
	defer subscription.Unsubscribe()

	// Output:
	// Found: Alice
	// Not found: ""
	// Completed
}
