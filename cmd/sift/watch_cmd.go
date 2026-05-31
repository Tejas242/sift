package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
	"github.com/tejas242/sift/internal/watcher"
)

func init() {
	rootCmd.AddCommand(&cobra.Command{
		Use:   "watch <dir> [dir...]",
		Short: "Index a directory then watch it for changes",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer stop()

			idx, err := openIndex(ortLib)
			if err != nil {
				return err
			}
			defer idx.Close()

			if err := indexDirs(ctx, idx, args); err != nil {
				return err
			}
			if err := idx.Flush(); err != nil {
				return err
			}
			s := idx.Stats()
			fmt.Fprintf(os.Stderr, "Done. %d chunks indexed. Watching for changes… (Ctrl+C to stop)\n", s.NumChunks)

			w, err := watcher.New(idx)
			if err != nil {
				return err
			}

			done := make(chan struct{})
			go func() {
				<-ctx.Done()
				close(done)
			}()

			for _, dir := range args {
				go func(d string) {
					if err := w.Watch(d, done); err != nil {
						fmt.Fprintf(os.Stderr, "watch error %s: %v\n", d, err)
					}
				}(dir)
			}
			<-done
			return nil
		},
	})
}
