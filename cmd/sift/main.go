package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/pelletier/go-toml/v2"
	"github.com/spf13/cobra"

	"github.com/tejas242/sift/internal/embed"
	"github.com/tejas242/sift/internal/index"
	"github.com/tejas242/sift/internal/tui"
	"github.com/tejas242/sift/internal/watcher"
)

var (
	defaultModelDir = "./models"
	defaultSiftDir  = ".sift"
	defaultOrtLib   = "./lib/onnxruntime.so"
	defaultThreads  = 0
	defaultMaxFile  = 512
)

func main() {
	root := &cobra.Command{
		Use:   "sift",
		Short: "Local semantic search for developers",
		Long:  "sift — fast, offline semantic file search powered by BGE-small-en-v1.5 and HNSW.",
	}

	var cfg struct {
		ModelDir  string `toml:"model-dir"`
		OrtLib    string `toml:"ort-lib"`
		Threads   int    `toml:"threads"`
		MaxFileKB int    `toml:"max-file-kb"`
	}

	if b, err := os.ReadFile(".sift.toml"); err == nil {
		if err := toml.Unmarshal(b, &cfg); err == nil {
			if cfg.ModelDir != "" {
				defaultModelDir = cfg.ModelDir
			}
			if cfg.OrtLib != "" {
				defaultOrtLib = cfg.OrtLib
			}
			if cfg.Threads > 0 {
				defaultThreads = cfg.Threads
			}
			if cfg.MaxFileKB > 0 {
				defaultMaxFile = cfg.MaxFileKB
			}
		}
	}

	var modelDir string
	var ortLib string
	var numThreads int
	var maxFileKB int
	root.PersistentFlags().StringVar(&modelDir, "model-dir", defaultModelDir, "directory containing ONNX model files")
	root.PersistentFlags().StringVar(&ortLib, "ort-lib", defaultOrtLib, "path to onnxruntime.so (auto-detected if empty)")
	root.PersistentFlags().IntVar(&numThreads, "threads", defaultThreads, "ONNX intra-op thread count (0 = auto, usually NumCPU capped at 4)")
	root.PersistentFlags().IntVar(&maxFileKB, "max-file-kb", defaultMaxFile, "skip indexing files larger than this (in KB)")

	resolveOrtLib := func(flag string) string {
		if flag != "" {
			return flag
		}
		if exe, err := os.Executable(); err == nil {
			candidate := filepath.Join(filepath.Dir(exe), "lib", "onnxruntime.so")
			if _, err := os.Stat(candidate); err == nil {
				return candidate
			}
		}
		if _, err := os.Stat(defaultOrtLib); err == nil {
			absPath, _ := filepath.Abs(defaultOrtLib)
			return absPath
		}
		return ""
	}

	// openIndex loads the model and index, printing status so the user knows
	// it isn't stuck (model loading can take 1–4s on first run).
	openIndex := func(ortLibFlag string) (*index.Index, error) {
		fmt.Fprint(os.Stderr, "Loading model… ")
		idx, err := index.Open(defaultSiftDir, modelDir, resolveOrtLib(ortLibFlag), numThreads, maxFileKB)
		if err != nil {
			fmt.Fprintln(os.Stderr, "")
			return nil, err
		}
		fmt.Fprintln(os.Stderr, "ready.")
		return idx, nil
	}

	// indexDirs indexes directories using ctx for cancellation.
	// IMPORTANT: session.Run() is a blocking CGo call that Go cannot preempt.
	// We start a hard-exit goroutine so Ctrl+C always terminates the process
	// after a 600ms grace period. A "done" channel cancels the goroutine
	// on clean exit so the interrupt message never prints spuriously.
	indexDirs := func(ctx context.Context, idx *index.Index, dirs []string) error {
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

	// ---- sift index <dir> --------------------------------------------------
	root.AddCommand(&cobra.Command{
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

	// ---- sift search <query> -----------------------------------------------
	var jsonExport bool
	searchCmd := &cobra.Command{
		Use:   "search <query>",
		Short: "Non-interactive semantic search",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			query := strings.Join(args, " ")

			idx, err := openIndex(ortLib)
			if err != nil {
				return err
			}
			defer idx.Close()

			results, err := idx.Search(query, 10)
			if err != nil {
				return err
			}
			if len(results) == 0 {
				if jsonExport {
					fmt.Println("[]")
				} else {
					fmt.Println("no results")
				}
				return nil
			}
			if jsonExport {
				j, err := json.MarshalIndent(results, "", "  ")
				if err != nil {
					return fmt.Errorf("marshal json: %w", err)
				}
				fmt.Println(string(j))
				return nil
			}
			for i, r := range results {
				fmt.Printf("%2d  %.3f  %s:%d\n    %s\n\n",
					i+1, r.Score, r.Meta.Path, r.Meta.LineNum, r.Meta.Text)
			}
			return nil
		},
	}
	searchCmd.Flags().BoolVar(&jsonExport, "json", false, "output search results as JSON")
	root.AddCommand(searchCmd)

	// ---- sift watch <dir> --------------------------------------------------
	root.AddCommand(&cobra.Command{
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

	// ---- sift tui ----------------------------------------------------------
	root.AddCommand(&cobra.Command{
		Use:   "tui",
		Short: "Launch interactive BubbleTea search interface",
		RunE: func(cmd *cobra.Command, args []string) error {
			idx, err := openIndex(ortLib)
			if err != nil {
				return err
			}
			defer idx.Close()

			m := tui.New(idx)
			p := tea.NewProgram(m, tea.WithAltScreen())
			_, err = p.Run()
			return err
		},
	})

	// ---- sift stats --------------------------------------------------------
	root.AddCommand(&cobra.Command{
		Use:   "stats",
		Short: "Show index statistics",
		RunE: func(cmd *cobra.Command, args []string) error {
			idx, err := openIndex(ortLib)
			if err != nil {
				return err
			}
			defer idx.Close()

			s := idx.Stats()
			fmt.Printf("chunks:    %d\n", s.NumChunks)
			fmt.Printf("files:     %d\n", s.NumFiles)
			fmt.Printf("size:      %d KB\n", s.IndexSizeKB)
			if !s.LastUpdated.IsZero() {
				fmt.Printf("updated:   %s\n", s.LastUpdated.Format("2006-01-02 15:04:05"))
			}
			return nil
		},
	})

	// ---- sift clear --------------------------------------------------------
	var forceFlag bool
	clearCmd := &cobra.Command{
		Use:   "clear",
		Short: "Remove the sift index (.sift/ directory)",
		RunE: func(cmd *cobra.Command, args []string) error {
			if _, err := os.Stat(defaultSiftDir); os.IsNotExist(err) {
				fmt.Println("No index found — nothing to clear.")
				return nil
			}
			if !forceFlag {
				fmt.Printf("Remove %s? This cannot be undone. [y/N] ", defaultSiftDir)
				var ans string
				fmt.Scanln(&ans)
				if ans != "y" && ans != "Y" {
					fmt.Println("Aborted.")
					return nil
				}
			}
			if err := os.RemoveAll(defaultSiftDir); err != nil {
				return fmt.Errorf("clear: %w", err)
			}
			fmt.Println("Index cleared.")
			return nil
		},
	}
	clearCmd.Flags().BoolVar(&forceFlag, "force", false, "skip confirmation prompt")
	root.AddCommand(clearCmd)

	// ---- sift rebuild -------------------------------------------------------
	root.AddCommand(&cobra.Command{
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

	// ---- sift bench --------------------------------------------------------
	root.AddCommand(&cobra.Command{
		Use:   "bench",
		Short: "Benchmark tokenizer and ONNX inference speed on this machine",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Fprint(os.Stderr, "Loading model… ")
			e, err := embed.New(modelDir, resolveOrtLib(ortLib), numThreads)
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

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

// isInterrupted returns true if err indicates a context cancellation or deadline.
func isInterrupted(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// makeProgressPrinter returns a ProgressFunc that prints a compact progress line.
// Skipped files (mtime cache hit) are shown with · instead of a percentage.
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
