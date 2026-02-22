// Package chunker splits text files into overlapping token-window chunks
// suitable for embedding. It streams file content to avoid loading large
// files fully into memory.
package chunker

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// SupportedExtensions is the set of file extensions sift will index.
var SupportedExtensions = map[string]bool{
	".md": true, ".txt": true, ".go": true, ".py": true,
	".js": true, ".ts": true, ".rs": true, ".c": true,
	".cpp": true, ".h": true, ".json": true, ".yaml": true,
	".yml": true, ".toml": true, ".kdl": true, ".conf": true,
}

// Chunk represents a slice of a source file.
type Chunk struct {
	Path      string
	Text      string
	LineNum   int // 1-indexed line number of the start of the chunk
	StartByte int64
	EndByte   int64
	Index     int // chunk index within the file
}

// Options controls chunking behaviour.
type Options struct {
	// MaxBytes is the maximum size of a single chunk.
	// BGE-small supports 512 tokens (~2000 bytes), but 1200 bytes is safer
	// and preserves strong semantic density.
	MaxBytes int
	// OverlapBytes is how many bytes of the previous chunk to include in the next.
	OverlapBytes int
}

// DefaultOptions returns the recommended chunking parameters for BGE-small.
func DefaultOptions() Options {
	return Options{
		MaxBytes:     1200, // ~250-300 tokens
		OverlapBytes: 250,  // ~50-60 tokens overlap
	}
}

// IsSupportedFile returns true if the file extension is supported and the
// file does not appear to be binary (checked via a short header sniff).
func IsSupportedFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	if !SupportedExtensions[ext] {
		return false
	}
	return !isBinary(path)
}

// isBinary sniffs the first 512 bytes to detect binary content.
func isBinary(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return true
	}
	defer f.Close()

	buf := make([]byte, 512)
	n, err := f.Read(buf)
	if err != nil && err != io.EOF {
		return true
	}
	buf = buf[:n]

	// Null bytes strongly indicate binary data.
	return bytes.IndexByte(buf, 0) != -1
}

// ChunkFile reads the file at path and returns overlapping semantic chunks.
// It splits on \n\n, \n, or space to keep paragraphs and code blocks intact.
func ChunkFile(path string, opts Options) ([]Chunk, error) {
	if opts.MaxBytes <= 0 {
		opts = DefaultOptions()
	}

	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("%s is a directory", path)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	return chunkBytes(data, path, opts)
}

// chunkBytes performs semantic text splitting.
func chunkBytes(data []byte, path string, opts Options) ([]Chunk, error) {
	text := string(data)
	if len(strings.TrimSpace(text)) == 0 {
		return nil, nil
	}

	var chunks []Chunk
	var chunkIdx int
	start := 0

	for start < len(text) {
		end := start + opts.MaxBytes
		if end >= len(text) {
			leadingSpaces := len(text[start:]) - len(strings.TrimLeft(text[start:], " \t\n\r"))
			chunks = append(chunks, Chunk{
				Path:      path,
				Text:      strings.TrimSpace(text[start:]),
				LineNum:   1 + bytes.Count(data[:start+leadingSpaces], []byte{'\n'}),
				StartByte: int64(start),
				EndByte:   int64(len(text)),
				Index:     chunkIdx,
			})
			break
		}

		// Find best semantic split point looking backwards from 'end'
		bestSplit := -1

		// 1. Try paragraph break (\n\n)
		bestSplit = strings.LastIndex(text[start:end], "\n\n")
		if bestSplit != -1 {
			bestSplit += start + 2
		} else {
			// 2. Try line break (\n)
			bestSplit = strings.LastIndex(text[start:end], "\n")
			if bestSplit != -1 {
				bestSplit += start + 1
			} else {
				// 3. Try space word break
				bestSplit = strings.LastIndexByte(text[start:end], ' ')
				if bestSplit != -1 {
					bestSplit += start + 1
				} else {
					// 4. Force split mid-word
					bestSplit = end
				}
			}
		}

		leadingSpaces := len(text[start:bestSplit]) - len(strings.TrimLeft(text[start:bestSplit], " \t\n\r"))
		chunks = append(chunks, Chunk{
			Path:      path,
			Text:      strings.TrimSpace(text[start:bestSplit]),
			LineNum:   1 + bytes.Count(data[:start+leadingSpaces], []byte{'\n'}),
			StartByte: int64(start),
			EndByte:   int64(bestSplit),
			Index:     chunkIdx,
		})
		chunkIdx++

		// Calculate overlap context for the next chunk
		overlapStart := bestSplit - opts.OverlapBytes
		if overlapStart <= start {
			// Ensure we always advance at least 1 character to avoid infinite loops
			overlapStart = start + 1
		} else {
			// Snap overlap start forward to the next semantic boundary
			// so the overlap starts cleanly at a line or word
			nextNL := strings.IndexByte(text[overlapStart:bestSplit], '\n')
			if nextNL != -1 {
				overlapStart += nextNL + 1
			} else {
				nextSp := strings.IndexByte(text[overlapStart:bestSplit], ' ')
				if nextSp != -1 {
					overlapStart += nextSp + 1
				}
			}
		}

		start = overlapStart
	}

	// Filter out empty chunks resulting from pure whitespace text regions
	var filtered []Chunk
	for _, c := range chunks {
		if c.Text != "" {
			filtered = append(filtered, c)
		}
	}

	return filtered, nil
}
