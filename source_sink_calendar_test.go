package ro

import (
    "context"
    "io/ioutil"
    "net/http"
    "net/http/httptest"
    "os"
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
)

func TestWatchFileAndWriteToFileAndDedup(t *testing.T) {
    t.Parallel()

    tmp, err := ioutil.TempFile("", "ro_test_*.ics")
    assert.NoError(t, err)
    path := tmp.Name()
    _ = tmp.Close()
    defer os.Remove(path)

    // write initial content
    err = ioutil.WriteFile(path, []byte("BEGIN:VCALENDAR\nUID:1\nEND:VCALENDAR"), 0644)
    assert.NoError(t, err)

    // collect two emissions: initial and after change
    ch := make(chan []string, 1)

    go func() {
        vals, _ := Collect(Pipe1(WatchFile(path, 10*time.Millisecond), Take[string](2)))
        ch <- vals
    }()

    time.Sleep(30 * time.Millisecond)
    // modify file
    err = ioutil.WriteFile(path, []byte("BEGIN:VCALENDAR\nUID:2\nEND:VCALENDAR"), 0644)
    assert.NoError(t, err)

    vals := <-ch
    assert.GreaterOrEqual(t, len(vals), 2)

    // test WriteToFile
    outPath := path + ".out"
    defer os.Remove(outPath)

    values, err := Collect(Pipe1(Of("a","b","a"), WriteToFile(outPath, false, 0644)))
    assert.NoError(t, err)
    assert.Equal(t, []string{"a","b","a"}, values)

    // file contains entries
    b, err := ioutil.ReadFile(outPath)
    assert.NoError(t, err)
    s := string(b)
    assert.Contains(t, s, "a")

    // Dedup
    vals2, err := Collect(Pipe1(Of("x","y","x"), Dedup()))
    assert.NoError(t, err)
    assert.Equal(t, []string{"x","y"}, vals2)
}

func TestWatchURLAndSerializeUnserializeValidateFilter(t *testing.T) {
    t.Parallel()

    // httptest server returning JSON
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte(`{"uid":"u1","ts":"2020-01-01T00:00:00Z"}`))
    }))
    defer srv.Close()

    // WatchURL should emit initial content
    vals, err := Collect(Pipe1(WatchURL(srv.URL, 50*time.Millisecond), Take[string](1)))
    assert.NoError(t, err)
    assert.GreaterOrEqual(t, len(vals), 1)

    // Serialize / Unserialize
    type Item struct{ UID string `json:"uid"` }
    items, err := Collect(Pipe2(Of(Item{UID: "u"}), Serialize[Item](), Unserialize[Item]()))
    assert.NoError(t, err)
    assert.Equal(t, []Item{{UID: "u"}}, items)

    // Validate: accept only UID == u
    validator := func(ctx context.Context, it Item) (context.Context, error) {
        if it.UID != "u" {
            return ctx, ErrInvalidCalendar
        }
        return ctx, nil
    }

    vout, err := Collect(Pipe1(Of(Item{UID: "u"}, Item{UID: "z"}), Validate(validator)))
    assert.Error(t, err)
    // Collect returns collected values before the error
    assert.Equal(t, []Item{{UID: "u"}}, vout)

    // FilterByParticipant simple substring
    fvals, err := Collect(Pipe1(Of("attendee:alice@example.com","other"), FilterByParticipant("alice@example.com")))
    assert.NoError(t, err)
    assert.Equal(t, []string{"attendee:alice@example.com"}, fvals)

    // FilterByTimeWindow: include RFC3339 timestamp
    now := time.Now()
    start := now.Add(-time.Hour)
    end := now.Add(time.Hour)
    payload := now.UTC().Format(time.RFC3339)
    tw, err := Collect(Pipe1(Of("some text " + payload), FilterByTimeWindow(start, end)))
    assert.NoError(t, err)
    assert.Equal(t, []string{"some text " + payload}, tw)
}
