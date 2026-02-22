// Package watcher watches a directory for file changes and triggers incremental
// re-indexing using fsnotify.
package watcher

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/tejas242/sift/internal/chunker"
	"github.com/tejas242/sift/internal/index"
)

// Watcher watches a directory tree for changes and updates the index.
type Watcher struct {
	fw  *fsnotify.Watcher
	idx *index.Index
}

// New creates a Watcher backed by the given index.
func New(idx *index.Index) (*Watcher, error) {
	fw, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("fsnotify: %w", err)
	}
	return &Watcher{fw: fw, idx: idx}, nil
}

// Watch adds rootDir (and all subdirectories) to the watch list and begins
// processing events. It blocks until ctx is cancelled or an unrecoverable
// error occurs. Call this in a goroutine.
func (w *Watcher) Watch(rootDir string, done <-chan struct{}) error {
	// Add all existing subdirectories.
	if err := w.addDirRecursive(rootDir); err != nil {
		return err
	}

	// Debounce map: path→timer
	pending := make(map[string]*time.Timer)

	for {
		select {
		case <-done:
			return w.fw.Close()

		case event, ok := <-w.fw.Events:
			if !ok {
				return nil
			}
			path := event.Name

			// Add new directories to the watch list.
			if event.Has(fsnotify.Create) {
				if fi, err := os.Stat(path); err == nil && fi.IsDir() {
					_ = w.addDirRecursive(path)
				}
			}

			if !chunker.IsSupportedFile(path) {
				continue
			}

			if event.Has(fsnotify.Write) || event.Has(fsnotify.Create) {
				// Debounce: reset timer on rapid saves.
				if t, ok := pending[path]; ok {
					t.Stop()
				}
				pending[path] = time.AfterFunc(500*time.Millisecond, func() {
					fmt.Fprintf(os.Stderr, "[watch] re-indexing %s\n", path)
					if _, err := w.idx.AddFile(path); err != nil {
						fmt.Fprintf(os.Stderr, "[watch] error: %v\n", err)
						return
					}
					if err := w.idx.Flush(); err != nil {
						fmt.Fprintf(os.Stderr, "[watch] flush error: %v\n", err)
					}
				})
			}

		case err, ok := <-w.fw.Errors:
			if !ok {
				return nil
			}
			fmt.Fprintf(os.Stderr, "[watch] error: %v\n", err)
		}
	}
}

// addDirRecursive adds dir and all non-hidden subdirectories to the watcher.
func (w *Watcher) addDirRecursive(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	if err := w.fw.Add(dir); err != nil {
		return fmt.Errorf("watch %s: %w", dir, err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		if e.IsDir() {
			if err := w.addDirRecursive(filepath.Join(dir, e.Name())); err != nil {
				// Non-fatal: log and continue.
				fmt.Fprintf(os.Stderr, "[watch] skip dir: %v\n", err)
			}
		}
	}
	return nil
}
