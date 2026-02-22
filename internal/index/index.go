// Package index manages the sift vector index: chunk metadata, vector storage,
// and the HNSW graph. Vectors are stored as flat float32 blocks in a binary file.
package index

import (
	"context"
	"encoding/json"
	"fmt"

	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/screenager/sift/internal/chunker"
	"github.com/screenager/sift/internal/embed"
	"github.com/screenager/sift/internal/hnsw"
)

const (
	hnswFile    = "hnsw.bin"
	vectorsFile = "vectors.bin"
	metaFile    = "meta.json"
)

// ChunkMeta stores provenance for each indexed chunk.
type ChunkMeta struct {
	Path       string    `json:"path"`
	LineNum    int       `json:"line_num"`
	StartByte  int64     `json:"start_byte"`
	EndByte    int64     `json:"end_byte"`
	ChunkIndex int       `json:"chunk_index"`
	Text       string    `json:"text"` // preview (first 200 chars)
	Mtime      time.Time `json:"mtime"`
}

// Stats holds summary information about the current index.
type Stats struct {
	NumChunks   int
	NumFiles    int
	IndexSizeKB int64
	LastUpdated time.Time
}

// SearchResult is a single result returned from Search.
type SearchResult struct {
	Meta  ChunkMeta
	Score float32
}

// Index is the main index state.
type Index struct {
	mu               sync.RWMutex
	dir              string
	graph            *hnsw.Graph
	chunks           []ChunkMeta          // indexed by chunk ID (== HNSW node ID)
	fileCache        map[string]time.Time // path → mtime of last indexed version
	embedder         *embed.Embedder
	maxFileSizeBytes int64
	dirty            bool
	lastUpdated      time.Time
}

// Open loads (or creates) an index stored in dir.
// modelDir is the path to the BGE-small model directory.
// ortLibPath is the path to onnxruntime.so; pass "" to use the system default.
// numThreads controls ONNX intra-op parallelism; 0 = auto (min(NumCPU, 4)).
// maxFileKB skips files larger than this limit.
func Open(dir, modelDir, ortLibPath string, numThreads, maxFileKB int) (*Index, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("mkdir %s: %w", dir, err)
	}

	e, err := embed.New(modelDir, ortLibPath, numThreads)
	if err != nil {
		return nil, fmt.Errorf("embedder: %w", err)
	}

	idx := &Index{
		dir:              dir,
		embedder:         e,
		maxFileSizeBytes: int64(maxFileKB) * 1024,
		graph:            hnsw.New(hnsw.DefaultM, hnsw.DefaultEfConstruction, hnsw.DefaultEfSearch),
	}

	// Load existing index if present.
	metaPath := filepath.Join(dir, metaFile)
	if data, err := os.ReadFile(metaPath); err == nil {
		if err := json.Unmarshal(data, &idx.chunks); err != nil {
			return nil, fmt.Errorf("corrupt meta.json — run `sift index` to rebuild: %w", err)
		}
	}

	hnswPath := filepath.Join(dir, hnswFile)
	if _, err := os.Stat(hnswPath); err == nil {
		g, err := hnsw.Load(hnswPath)
		if err != nil {
			return nil, fmt.Errorf("corrupt hnsw.bin — run `sift index` to rebuild: %w", err)
		}
		idx.graph = g
	}

	// Build mtime skip-cache from loaded chunks.
	idx.fileCache = make(map[string]time.Time, len(idx.chunks))
	for _, c := range idx.chunks {
		if existing, ok := idx.fileCache[c.Path]; !ok || c.Mtime.After(existing) {
			idx.fileCache[c.Path] = c.Mtime
		}
	}

	return idx, nil
}

// Close flushes dirty state and releases the embedder.
func (idx *Index) Close() error {
	if err := idx.Flush(); err != nil {
		return err
	}
	idx.embedder.Close()
	return nil
}

// AddFile chunks, embeds, and indexes all chunks from a single file.
// If the file's mtime matches the cached value it is skipped (already up to date).
// ctx is checked between embedding batches: cancelling it stops mid-file.
func (idx *Index) AddFile(path string) (skipped bool, err error) {
	return idx.AddFileCtx(context.Background(), path)
}

// AddFileCtx is like AddFile but respects ctx cancellation between embed batches.
func (idx *Index) AddFileCtx(ctx context.Context, path string) (skipped bool, err error) {
	if !chunker.IsSupportedFile(path) {
		return false, nil
	}

	info, statErr := os.Stat(path)
	if statErr != nil {
		fmt.Fprintf(os.Stderr, "skip %s: %v\n", path, statErr)
		return false, nil
	}

	// Skip very large files — they're almost certainly generated data, not
	// source code or documentation worth indexing chunk by chunk.
	if info.Size() > idx.maxFileSizeBytes {
		fmt.Fprintf(os.Stderr, "skip %s: file too large (%d KB > %d KB limit)\n",
			path, info.Size()/1024, idx.maxFileSizeBytes/1024)
		return false, nil
	}

	mtime := info.ModTime()

	// Skip-cache: file at this mtime is already indexed.
	idx.mu.RLock()
	cachedMtime, inCache := idx.fileCache[path]
	idx.mu.RUnlock()
	if inCache && cachedMtime.Equal(mtime) {
		return true, nil
	}

	chunks, err := chunker.ChunkFile(path, chunker.DefaultOptions())
	if err != nil {
		fmt.Fprintf(os.Stderr, "skip %s: chunk error: %v\n", path, err)
		return false, nil
	}
	if len(chunks) == 0 {
		return false, nil
	}

	base := filepath.Base(path)
	nChunks := len(chunks)
	verbose := nChunks > 4 // show chunk progress for files with many chunks

	// Embed batch-by-batch so we can: (a) show live progress and (b) check ctx.
	const batchSize = 4
	vecs := make([][]float32, 0, nChunks)
	for start := 0; start < nChunks; start += batchSize {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return false, ctxErr
		}
		end := start + batchSize
		if end > nChunks {
			end = nChunks
		}
		batch := make([]string, end-start)
		for i, c := range chunks[start:end] {
			batch[i] = c.Text
		}
		if verbose {
			fmt.Fprintf(os.Stderr, "\r    embedding chunk %d–%d / %d  %s ",
				start+1, end, nChunks, base)
		}
		batchVecs, embedErr := idx.embedder.Embed(batch)
		if embedErr != nil {
			if verbose {
				fmt.Fprintln(os.Stderr, "")
			}
			fmt.Fprintf(os.Stderr, "skip %s: embed error: %v\n", path, embedErr)
			return false, nil
		}
		vecs = append(vecs, batchVecs...)
	}
	if verbose {
		fmt.Fprintf(os.Stderr, "\r    %-60s\r", "") // clear the chunk line
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	for i, vec := range vecs {
		preview := chunks[i].Text
		if len(preview) > 200 {
			preview = preview[:197] + "..."
		}
		idx.chunks = append(idx.chunks, ChunkMeta{
			Path:       path,
			LineNum:    chunks[i].LineNum,
			StartByte:  chunks[i].StartByte,
			EndByte:    chunks[i].EndByte,
			ChunkIndex: chunks[i].Index,
			Text:       preview,
			Mtime:      mtime,
		})
		idx.graph.Insert(vec)
	}

	idx.fileCache[path] = mtime
	idx.dirty = true
	idx.lastUpdated = time.Now()
	return false, nil
}

// Search embeds query with the BGE instruction prefix and returns the top-k most similar chunks.
// It performs cross-chunk deduplication: it will not return two chunks from the same file.
func (idx *Index) Search(query string, k int) ([]SearchResult, error) {
	queryVec, err := idx.embedder.EmbedQuery(query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Fetch more hits to allow filtering out duplicates from the same file.
	fetchK := k * 5
	if fetchK > len(idx.chunks) {
		fetchK = len(idx.chunks)
	}
	if fetchK == 0 {
		return nil, nil
	}

	hits := idx.graph.Search(queryVec, fetchK)

	queryWords := strings.Fields(strings.ToLower(query))

	type scoredHit struct {
		meta  ChunkMeta
		score float32
		text  string
	}
	var reranked []scoredHit

	for _, h := range hits {
		if int(h.ID) >= len(idx.chunks) {
			continue
		}
		meta := idx.chunks[h.ID]
		score := h.Score

		// Read chunk text to allow both keyword boosting and cross-encoder reranking
		var chunkText string
		f, err := os.Open(meta.Path)
		if err == nil {
			buf := make([]byte, meta.EndByte-meta.StartByte)
			if _, err := f.ReadAt(buf, meta.StartByte); err == nil {
				chunkText = string(buf)
				lowerText := strings.ToLower(chunkText)
				var matches int
				for _, w := range queryWords {
					if len(w) > 2 && strings.Contains(lowerText, w) {
						matches++
					}
				}
				score += float32(matches) * 0.05
			}
			f.Close()
		}

		reranked = append(reranked, scoredHit{meta: meta, score: score, text: chunkText})
	}

	// Sort by hybrid bi-encoder + keyword score
	sort.Slice(reranked, func(i, j int) bool {
		return reranked[i].score > reranked[j].score
	})

	results := make([]SearchResult, 0, k)
	seen := make(map[string]bool)

	for _, h := range reranked {
		if len(results) >= k {
			break
		}
		if seen[h.meta.Path] {
			continue
		}
		seen[h.meta.Path] = true

		results = append(results, SearchResult{
			Meta:  h.meta,
			Score: h.score,
		})
	}
	return results, nil
}

// Flush writes the HNSW graph and metadata to disk if dirty.
func (idx *Index) Flush() error {
	idx.mu.RLock()
	dirty := idx.dirty
	idx.mu.RUnlock()

	if !dirty {
		return nil
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Save HNSW graph.
	hnswPath := filepath.Join(idx.dir, hnswFile)
	if err := idx.graph.Save(hnswPath); err != nil {
		return fmt.Errorf("save hnsw: %w", err)
	}

	// Save chunk metadata.
	metaPath := filepath.Join(idx.dir, metaFile)
	data, err := json.MarshalIndent(idx.chunks, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal meta: %w", err)
	}
	if err := os.WriteFile(metaPath, data, 0o644); err != nil {
		return fmt.Errorf("write meta: %w", err)
	}

	idx.dirty = false
	return nil
}

// Stats returns summary statistics about the index.
func (idx *Index) Stats() Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	fileSet := make(map[string]struct{})
	for _, c := range idx.chunks {
		fileSet[c.Path] = struct{}{}
	}

	// Measure disk usage.
	var sizeBytes int64
	for _, fname := range []string{hnswFile, metaFile} {
		if fi, err := os.Stat(filepath.Join(idx.dir, fname)); err == nil {
			sizeBytes += fi.Size()
		}
	}

	return Stats{
		NumChunks:   len(idx.chunks),
		NumFiles:    len(fileSet),
		IndexSizeKB: sizeBytes / 1024,
		LastUpdated: idx.lastUpdated,
	}
}

// RebuildFromDir reindexes everything in rootDir from scratch.
func (idx *Index) RebuildFromDir(ctx context.Context, rootDir string) error {
	idx.mu.Lock()
	idx.chunks = idx.chunks[:0]
	idx.graph = hnsw.New(hnsw.DefaultM, hnsw.DefaultEfConstruction, hnsw.DefaultEfSearch)
	idx.fileCache = make(map[string]time.Time) // clear skip-cache
	idx.mu.Unlock()

	return idx.IndexDirWithProgress(ctx, rootDir, nil)
}

// ProgressFunc is called after each file is processed during indexing.
// done and total are file counts; skipped=true means mtime cache hit (no re-embed).
type ProgressFunc func(done, total int, path string, skipped bool)

// IndexDir walks rootDir and indexes all supported files.
// ctx is checked between files; cancel it to interrupt indexing gracefully.
func (idx *Index) IndexDir(ctx context.Context, rootDir string) error {
	return idx.IndexDirWithProgress(ctx, rootDir, nil)
}

// IndexDirWithProgress walks rootDir, indexes all supported files, and calls
// progress after each file (may be nil). ctx is checked between each file;
// cancel it to stop indexing after the current file finishes embedding.
func (idx *Index) IndexDirWithProgress(ctx context.Context, rootDir string, progress ProgressFunc) error {
	// First pass: collect all eligible file paths so we know the total.
	var paths []string
	err := walkDir(rootDir, func(path string) error {
		if chunker.IsSupportedFile(path) {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return err
	}

	total := len(paths)
	for i, path := range paths {
		// Check for cancellation before each file (embedding can be slow).
		if err := ctx.Err(); err != nil {
			return err
		}
		skipped, err := idx.AddFileCtx(ctx, path)
		if err != nil {
			return err
		}
		if progress != nil {
			progress(i+1, total, path, skipped)
		}
	}
	return nil
}

// walkDir walks rootDir recursively, calling fn for each file.
// Skips hidden directories.
func walkDir(rootDir string, fn func(string) error) error {
	entries, err := os.ReadDir(rootDir)
	if err != nil {
		return fmt.Errorf("readdir %s: %w", rootDir, err)
	}
	for _, entry := range entries {
		name := entry.Name()
		// Skip hidden.
		if strings.HasPrefix(name, ".") {
			continue
		}
		full := filepath.Join(rootDir, name)
		if entry.IsDir() {
			if err := walkDir(full, fn); err != nil {
				return err
			}
		} else {
			if err := fn(full); err != nil {
				return err
			}
		}
	}
	return nil
}
