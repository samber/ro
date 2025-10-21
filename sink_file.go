package ro

import (
    "context"
    "os"
)

// WriteToFile writes each string item emitted by the source Observable to the specified file.
// If append is true, it appends lines; otherwise it truncates the file on first write.
// It emits the written string downstream unchanged.
func WriteToFile(path string, appendMode bool, perm os.FileMode) func(Observable[string]) Observable[string] {
    return func(source Observable[string]) Observable[string] {
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[string]) Teardown {
            var f *os.File
            var opened bool

            openFile := func() error {
                if opened {
                    return nil
                }
                var err error
                flag := os.O_CREATE | os.O_WRONLY
                if appendMode {
                    flag |= os.O_APPEND
                } else {
                    flag |= os.O_TRUNC
                }

                f, err = os.OpenFile(path, flag, perm)
                if err != nil {
                    return err
                }

                opened = true
                return nil
            }

            sub := source.SubscribeWithContext(
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, value string) {
                        if err := openFile(); err != nil {
                            destination.ErrorWithContext(ctx, err)
                            return
                        }

                        if _, err := f.WriteString(value); err != nil {
                            destination.ErrorWithContext(ctx, err)
                            return
                        }

                        // write newline to separate entries
                        if _, err := f.WriteString("\n"); err != nil {
                            destination.ErrorWithContext(ctx, err)
                            return
                        }

                        destination.NextWithContext(ctx, value)
                    },
                    destination.ErrorWithContext,
                    func(ctx context.Context) {
                        if opened {
                            _ = f.Close()
                        }

                        destination.CompleteWithContext(ctx)
                    },
                ),
            )

            return sub.Unsubscribe
        })
    }
}
