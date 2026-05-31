package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
	"github.com/tejas242/sift/internal/config"
	"github.com/tejas242/sift/internal/index"
)

var (
	rootCmd = &cobra.Command{
		Use:   "sift",
		Short: "Local semantic search for developers",
		Long:  "sift — fast, offline semantic file search powered by BGE-small-en-v1.5 and HNSW.",
	}

	cfg        *config.Config
	modelDir   string
	ortLib     string
	numThreads int
	maxFileKB  int
)

func init() {
	var err error
	cfg, err = config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading configuration: %v\n", err)
		os.Exit(1)
	}

	rootCmd.PersistentFlags().StringVar(&modelDir, "model-dir", cfg.ModelDir, "directory containing ONNX model files")
	rootCmd.PersistentFlags().StringVar(&ortLib, "ort-lib", cfg.OrtLib, "path to onnxruntime.so (auto-detected if empty)")
	rootCmd.PersistentFlags().IntVar(&numThreads, "threads", cfg.Threads, "ONNX intra-op thread count (0 = auto, usually NumCPU capped at 4)")
	rootCmd.PersistentFlags().IntVar(&maxFileKB, "max-file-kb", cfg.MaxFileKB, "skip indexing files larger than this (in KB)")
}

// Execute executes the root command.
func Execute() error {
	return rootCmd.Execute()
}

func openIndex(ortLibFlag string) (*index.Index, error) {
	fmt.Fprint(os.Stderr, "Loading model… ")
	resolved := config.ResolveOrtLib(ortLibFlag)
	idx, err := index.Open(config.DefaultSiftDir, modelDir, resolved, numThreads, maxFileKB)
	if err != nil {
		fmt.Fprintln(os.Stderr, "")
		return nil, err
	}
	fmt.Fprintln(os.Stderr, "ready.")
	return idx, nil
}

func indexDirs(ctx context.Context, idx *index.Index, dirs []string) error {
	done := make(chan struct{})
	defer close(done)

	go func() {
		select {
		case <-done:
			return // clean exit — do nothing
		case <-ctx.Done():
			fmt.Fprintln(os.Stderr, "\n[sift] stopping — waiting up to 1s for current embed to finish…")
			select {
			case <-done:
				return // finished before timeout
			case <-time.After(time.Second):
				fmt.Fprintln(os.Stderr, "[sift] exiting.")
				os.Exit(130)
			}
		}
	}()

	prog := makeProgressPrinter()
	for _, dir := range dirs {
		fmt.Fprintf(os.Stderr, "Scanning %s…\n", dir)
		err := idx.IndexDirWithProgress(ctx, dir, prog)
		if err != nil {
			if isInterrupted(err) {
				fmt.Fprintln(os.Stderr, "\nInterrupted — saving partial index…")
				return nil
			}
			return err
		}
	}
	return nil
}

func isInterrupted(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func makeProgressPrinter() index.ProgressFunc {
	return func(done, total int, path string, skipped bool) {
		short := filepath.Base(filepath.Dir(path)) + "/" + filepath.Base(path)
		if skipped {
			fmt.Fprintf(os.Stderr, "\r  [%d/%d]  ·   %-50s", done, total, short)
		} else {
			pct := 100 * done / total
			if done < total {
				fmt.Fprintf(os.Stderr, "\r  [%d/%d] %3d%%  %-50s",
					done, total, pct, short)
			} else {
				fmt.Fprintf(os.Stderr, "\r  [%d/%d] 100%%  %-50s\n",
					done, total, short)
			}
		}
	}
}
