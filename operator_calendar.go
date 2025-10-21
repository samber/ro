package ro

import (
    "context"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "errors"
    "time"
)

var ErrInvalidCalendar = errors.New("invalid calendar item")

// Serialize converts a value into JSON string.
func Serialize[T any]() func(Observable[T]) Observable[string] {
    return func(source Observable[T]) Observable[string] {
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[string]) Teardown {
            sub := source.SubscribeWithContext(
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, v T) {
                        b, err := json.Marshal(v)
                        if err != nil {
                            destination.ErrorWithContext(ctx, err)
                            return
                        }

                        destination.NextWithContext(ctx, string(b))
                    },
                    destination.ErrorWithContext,
                    destination.CompleteWithContext,
                ),
            )

            return sub.Unsubscribe
        })
    }
}

// Unserialize parses JSON string into the target type.
func Unserialize[T any]() func(Observable[string]) Observable[T] {
    return func(source Observable[string]) Observable[T] {
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
            sub := source.SubscribeWithContext(
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, s string) {
                        var out T
                        if err := json.Unmarshal([]byte(s), &out); err != nil {
                            destination.ErrorWithContext(ctx, err)
                            return
                        }

                        destination.NextWithContext(ctx, out)
                    },
                    destination.ErrorWithContext,
                    destination.CompleteWithContext,
                ),
            )

            return sub.Unsubscribe
        })
    }
}

// Validate applies a validator function and forwards only valid items.
func Validate[T any](validator func(ctx context.Context, item T) (context.Context, error)) func(Observable[T]) Observable[T] {
    return func(source Observable[T]) Observable[T] {
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
            sub := source.SubscribeWithContext(
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, v T) {
                        newCtx, err := validator(ctx, v)
                        if err != nil {
                            destination.ErrorWithContext(newCtx, err)
                            return
                        }

                        destination.NextWithContext(newCtx, v)
                    },
                    destination.ErrorWithContext,
                    destination.CompleteWithContext,
                ),
            )

            return sub.Unsubscribe
        })
    }
}

// FilterByParticipant filters calendar string items by a participant identifier.
// It expects the input to be a string containing JSON or ICS; for simplicity we
// filter by substring match.
func FilterByParticipant(participant string) func(Observable[string]) Observable[string] {
    return func(source Observable[string]) Observable[string] {
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[string]) Teardown {
            sub := source.SubscribeWithContext(
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, s string) {
                        if participant == "" || containsParticipant(s, participant) {
                            destination.NextWithContext(ctx, s)
                        }
                    },
                    destination.ErrorWithContext,
                    destination.CompleteWithContext,
                ),
            )

            return sub.Unsubscribe
        })
    }
}

// FilterByTimeWindow filters string payloads by a time window. For simplicity
// it expects the payload to contain RFC3339 timestamps and will check if any
// timestamp falls within the window.
func FilterByTimeWindow(start, end time.Time) func(Observable[string]) Observable[string] {
    return func(source Observable[string]) Observable[string] {
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[string]) Teardown {
            sub := source.SubscribeWithContext(
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, s string) {
                        if timeWindowMatch(s, start, end) {
                            destination.NextWithContext(ctx, s)
                        }
                    },
                    destination.ErrorWithContext,
                    destination.CompleteWithContext,
                ),
            )

            return sub.Unsubscribe
        })
    }
}

// Dedup removes duplicate payloads based on content hash.
func Dedup() func(Observable[string]) Observable[string] {
    return func(source Observable[string]) Observable[string] {
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[string]) Teardown {
            seen := map[string]struct{}{}

            sub := source.SubscribeWithContext(
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, s string) {
                        h := sha256.Sum256([]byte(s))
                        key := hex.EncodeToString(h[:])
                        if _, ok := seen[key]; ok {
                            return
                        }

                        seen[key] = struct{}{}
                        destination.NextWithContext(ctx, s)
                    },
                    destination.ErrorWithContext,
                    destination.CompleteWithContext,
                ),
            )

            return sub.Unsubscribe
        })
    }
}

// helpers (simple implementations)
func containsParticipant(s, participant string) bool {
    return participant == "" || (participant != "" && (contains(s, participant)))
}

func contains(s, sub string) bool {
    return len(s) >= len(sub) && (indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
    // naive implementation
    for i := 0; i+len(sub) <= len(s); i++ {
        if s[i:i+len(sub)] == sub {
            return i
        }
    }
    return -1
}

func timeWindowMatch(s string, start, end time.Time) bool {
    // naive: try to parse any RFC3339 date in the string
    // scan for substrings that look like 4-digit year and attempt parse
    for i := 0; i+20 <= len(s); i++ {
        sub := s[i : i+20]
        if t, err := time.Parse(time.RFC3339, sub); err == nil {
            if (t.Equal(start) || t.After(start)) && (t.Equal(end) || t.Before(end)) {
                return true
            }
        }
    }

    return false
}
