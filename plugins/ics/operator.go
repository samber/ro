package roics

import (
    "context"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "time"

    ics "github.com/arran4/golang-ical"
    "github.com/samber/ro"
)

// FilterVEventByParticipant filters events by participant (attendee) substring.
func FilterVEventByParticipant(participant string) func(ro.Observable[*ics.VEvent]) ro.Observable[*ics.VEvent] {
    return func(source ro.Observable[*ics.VEvent]) ro.Observable[*ics.VEvent] {
        return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[*ics.VEvent]) ro.Teardown {
            sub := source.SubscribeWithContext(ctx, ro.NewObserverWithContext(
                func(ctx context.Context, e *ics.VEvent) {
                    if participant == "" {
                        destination.NextWithContext(ctx, e)
                        return
                    }

                    props := e.GetProperties("ATTENDEE")
                    for _, p := range props {
                        if p != nil && p.Value != "" && contains(p.Value, participant) {
                            destination.NextWithContext(ctx, e)
                            return
                        }
                    }
                },
                destination.ErrorWithContext,
                destination.CompleteWithContext,
            ))

            return sub.Unsubscribe
        })
    }
}

// FilterVEventByTimeWindow filters events whose DTSTART falls within [start,end].
func FilterVEventByTimeWindow(start, end time.Time) func(ro.Observable[*ics.VEvent]) ro.Observable[*ics.VEvent] {
    return func(source ro.Observable[*ics.VEvent]) ro.Observable[*ics.VEvent] {
        return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[*ics.VEvent]) ro.Teardown {
            sub := source.SubscribeWithContext(ctx, ro.NewObserverWithContext(
                func(ctx context.Context, e *ics.VEvent) {
                    dt := e.GetProperty(ics.ComponentPropertyDtStart)
                    if dt == nil || dt.Value == "" {
                        return
                    }

                    // try parse common formats
                    if t, err := parseICSTime(dt.Value); err == nil {
                        if (t.Equal(start) || t.After(start)) && (t.Equal(end) || t.Before(end)) {
                            destination.NextWithContext(ctx, e)
                        }
                    }
                },
                destination.ErrorWithContext,
                destination.CompleteWithContext,
            ))

            return sub.Unsubscribe
        })
    }
}

// DedupVEvents deduplicates events by UID+DTSTART hash.
func DedupVEvents() func(ro.Observable[*ics.VEvent]) ro.Observable[*ics.VEvent] {
    return func(source ro.Observable[*ics.VEvent]) ro.Observable[*ics.VEvent] {
        return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[*ics.VEvent]) ro.Teardown {
            seen := map[string]struct{}{}

            sub := source.SubscribeWithContext(ctx, ro.NewObserverWithContext(
                func(ctx context.Context, e *ics.VEvent) {
                    uid := ""
                    if p := e.GetProperty("UID"); p != nil {
                        uid = p.Value
                    }

                    dt := ""
                    if p := e.GetProperty("DTSTART"); p != nil {
                        dt = p.Value
                    }

                    h := sha256.Sum256([]byte(uid + "|" + dt))
                    key := hex.EncodeToString(h[:])
                    if _, ok := seen[key]; ok {
                        return
                    }

                    seen[key] = struct{}{}
                    destination.NextWithContext(ctx, e)
                },
                destination.ErrorWithContext,
                destination.CompleteWithContext,
            ))

            return sub.Unsubscribe
        })
    }
}

// SerializeVEvent serializes a VEvent into JSON string representing its UID and DTSTART and SUMMARY.
func SerializeVEvent() func(ro.Observable[*ics.VEvent]) ro.Observable[string] {
    return func(source ro.Observable[*ics.VEvent]) ro.Observable[string] {
        return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[string]) ro.Teardown {
            sub := source.SubscribeWithContext(ctx, ro.NewObserverWithContext(
                func(ctx context.Context, e *ics.VEvent) {
                    obj := map[string]string{}
                    if p := e.GetProperty("UID"); p != nil {
                        obj["uid"] = p.Value
                    }
                    if p := e.GetProperty("DTSTART"); p != nil {
                        obj["dtstart"] = p.Value
                    }
                    if p := e.GetProperty("SUMMARY"); p != nil {
                        obj["summary"] = p.Value
                    }

                    if b, err := json.Marshal(obj); err == nil {
                        destination.NextWithContext(ctx, string(b))
                    } else {
                        destination.ErrorWithContext(ctx, err)
                    }
                },
                destination.ErrorWithContext,
                destination.CompleteWithContext,
            ))

            return sub.Unsubscribe
        })
    }
}

// UnserializeVEvent is a noop here (can't build a *ics.VEvent from JSON easily) —
// but we provide it to keep symmetry and possibly deserialize into a lightweight struct.
func UnserializeVEvent() func(ro.Observable[string]) ro.Observable[map[string]string] {
    return func(source ro.Observable[string]) ro.Observable[map[string]string] {
        return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[map[string]string]) ro.Teardown {
            sub := source.SubscribeWithContext(ctx, ro.NewObserverWithContext(
                func(ctx context.Context, s string) {
                    var m map[string]string
                    if err := json.Unmarshal([]byte(s), &m); err != nil {
                        destination.ErrorWithContext(ctx, err)
                        return
                    }

                    destination.NextWithContext(ctx, m)
                },
                destination.ErrorWithContext,
                destination.CompleteWithContext,
            ))

            return sub.Unsubscribe
        })
    }
}

// helpers
func parseICSTime(v string) (time.Time, error) {
    // Try RFC3339 first, then basic formats used in testdata (YYYYMMDD or YYYYMMDDTHHMMSSZ)
    if t, err := time.Parse(time.RFC3339, v); err == nil {
        return t, nil
    }

    // try YYYYMMDD
    if t, err := time.Parse("20060102", v); err == nil {
        return t, nil
    }

    if t, err := time.Parse("20060102T150405Z", v); err == nil {
        return t, nil
    }

    return time.Time{}, &time.ParseError{Layout: "RFC3339/ICSTime", Value: v, LayoutElem: ""}
}

func contains(s, sub string) bool {
    if len(sub) == 0 {
        return true
    }
    for i := 0; i+len(sub) <= len(s); i++ {
        if s[i:i+len(sub)] == sub {
            return true
        }
    }
    return false
}
