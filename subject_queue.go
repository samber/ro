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

package ro

import "sync"

// notificationQueue delivers a subject's notifications one at a time, in the order
// their state updates were made, and never while the subject's mutex is held.
//
// The first caller drains the queue. A call made while a drain is running (another
// goroutine, or an observer calling back into the subject from inside its own callback)
// only enqueues its notification, and returns: the drainer delivers it after the
// notification in progress. This is what keeps a re-entrant call from re-entering the
// observer that is currently being notified.
//
// The queue is unbounded when producers outpace the observers.
type notificationQueue struct {
	jobs     []func()
	draining bool
}

// run calls update with mu held. update changes the subject state and returns the
// notification to deliver for that change, or nil.
func (q *notificationQueue) run(mu *sync.Mutex, update func() func()) {
	mu.Lock()

	if job := update(); job != nil {
		q.jobs = append(q.jobs, job)
	}

	if q.draining {
		mu.Unlock()
		return
	}

	q.draining = true

	for len(q.jobs) > 0 {
		job := q.jobs[0]
		q.jobs[0] = nil
		q.jobs = q.jobs[1:]

		mu.Unlock()
		q.deliver(mu, job)
		mu.Lock()
	}

	q.jobs = nil
	q.draining = false

	mu.Unlock()
}

// deliver runs job. If it panics, the drain is handed back so the next call can resume
// delivering the notifications still queued.
func (q *notificationQueue) deliver(mu *sync.Mutex, job func()) {
	done := false

	defer func() {
		if !done {
			mu.Lock()
			q.draining = false
			mu.Unlock()
		}
	}()

	job()

	done = true
}
