package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

func init() {
	rootCmd.AddCommand(&cobra.Command{
		Use:   "index <dir> [dir...]",
		Short: "Index all supported files in a directory",
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
			fmt.Fprintf(os.Stderr, "Done. %d chunks from %d files indexed.\n", s.NumChunks, s.NumFiles)
			return nil
		},
	})
}
