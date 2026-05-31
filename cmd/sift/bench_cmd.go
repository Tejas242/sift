package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/tejas242/sift/internal/config"
	"github.com/tejas242/sift/internal/embed"
)

func init() {
	rootCmd.AddCommand(&cobra.Command{
		Use:   "bench",
		Short: "Benchmark tokenizer and ONNX inference speed on this machine",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Fprint(os.Stderr, "Loading model… ")
			resolved := config.ResolveOrtLib(ortLib)
			e, err := embed.New(modelDir, resolved, numThreads)
			if err != nil {
				return err
			}
			defer e.Close()
			fmt.Fprintln(os.Stderr, "ready.")

			texts := []struct {
				label string
				text  string
			}{
				{"short (8 words) ", "the quick brown fox jumps over the lazy dog"},
				{"medium (50 words)", strings.Repeat("the quick brown fox ", 50)},
				{"long (200 words) ", strings.Repeat("the quick brown fox jumps over the lazy dog. ", 20)},
			}

			fmt.Printf("\n%-20s  %10s  %10s  %10s\n", "text size", "tokenize", "inference", "total")
			fmt.Println(strings.Repeat("─", 55))
			for _, tc := range texts {
				tok, inf, tot, err := e.BenchmarkSingle(tc.text)
				if err != nil {
					return fmt.Errorf("bench %s: %w", tc.label, err)
				}
				fmt.Printf("%-20s  %10s  %10s  %10s\n", tc.label,
					tok.Round(time.Millisecond),
					inf.Round(time.Millisecond),
					tot.Round(time.Millisecond))
			}
			fmt.Printf("\nIf inference >500ms, try: sift --threads 1 index <dir>\n")
			fmt.Printf("Set SIFT_DEBUG=1 for per-batch timing during indexing.\n")
			return nil
		},
	})
}
