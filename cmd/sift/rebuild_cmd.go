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
		Use:   "rebuild <dir> [dir...]",
		Short: "Wipe and rebuild the index from scratch (ignores skip-cache)",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer stop()

			idx, err := openIndex(ortLib)
			if err != nil {
				return err
			}
			defer idx.Close()

			for _, dir := range args {
				fmt.Fprintf(os.Stderr, "Rebuilding index for %s…\n", dir)
				if err := idx.RebuildFromDir(ctx, dir); err != nil {
					if !isInterrupted(err) {
						return err
					}
					fmt.Fprintln(os.Stderr, "\nInterrupted — saving partial index…")
				}
			}
			if err := idx.Flush(); err != nil {
				return err
			}
			s := idx.Stats()
			fmt.Fprintf(os.Stderr, "Done. %d chunks from %d files.\n", s.NumChunks, s.NumFiles)
			return nil
		},
	})
}
