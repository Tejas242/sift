// Package embed provides BGE-small-en-v1.5 text embedding via ONNX Runtime.
// Vectors are L2-normalized so dot product == cosine similarity.
package embed

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	// maxSeqLen is the effective maximum token length per input.
	// BGE-small supports up to 512 tokens, but capping at 256 halves the
	// attention matrix (O(seqLen²)) and is sufficient for 200-word chunks.
	// Most English text at 200 words ≈ 250 tokens; some unicode-heavy text
	// may get truncated but embedding quality is negligibly affected.
	maxSeqLen = 256
	// EmbeddingDim is the output dimension of BGE-small-en-v1.5.
	EmbeddingDim = 384
	// defaultBatchSize keeps memory + inference latency bounded on low-end CPUs.
	defaultBatchSize = 4

	// BGEQueryPrefix is prepended to queries (not documents) for asymmetric
	// retrieval per the BGE-small-en-v1.5 paper recommendation.
	// Docs: https://huggingface.co/BAAI/bge-small-en-v1.5
	BGEQueryPrefix = "Represent this sentence for searching relevant passages: "
)

// Embedder wraps an ONNX session and a HuggingFace tokenizer.
type Embedder struct {
	session   *ort.DynamicAdvancedSession
	tokenizer *tokenizers.Tokenizer
	batchSize int
}

// New loads the ONNX model and tokenizer from modelDir.
// ortLibPath is the path to onnxruntime.so; pass "" to use the system default.
// numThreads controls intra-op parallelism; 0 = use min(4, NumCPU).
// modelDir must contain: model.onnx, tokenizer.json
func New(modelDir, ortLibPath string, numThreads int) (*Embedder, error) {
	modelPath := filepath.Join(modelDir, "model.onnx")
	tokenPath := filepath.Join(modelDir, "tokenizer.json")

	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model not found at %s — run `make download-model` first", modelPath)
	}
	if _, err := os.Stat(tokenPath); err != nil {
		return nil, fmt.Errorf("tokenizer not found at %s — run `make download-model` first", tokenPath)
	}

	// Point ORT at the bundled shared library if specified.
	if ortLibPath != "" {
		ort.SetSharedLibraryPath(ortLibPath)
	}

	// Initialize ONNX Runtime (no-op if already initialized).
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("init ort: %w", err)
	}

	// Determine thread count. More threads rarely help on ≤4-core machines
	// and cause severe contention when both IntraOp and InterOp spawn threads.
	if numThreads <= 0 {
		numThreads = runtime.NumCPU()
		if numThreads > 4 {
			numThreads = 4
		}
	}

	// Build session options (CPU only, conservatively threaded).
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()

	// IntraOpNumThreads: parallelism WITHIN a single op (e.g. MatMul).
	if err := opts.SetIntraOpNumThreads(numThreads); err != nil {
		return nil, fmt.Errorf("set intra threads: %w", err)
	}
	// InterOpNumThreads: parallelism BETWEEN ops in the graph.
	// Keep this at 1 to avoid excessive goroutine/thread spawning overhead.
	if err := opts.SetInterOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("set inter threads: %w", err)
	}

	// Input/output names for BGE-small-en-v1.5 ONNX.
	inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
	outputNames := []string{"last_hidden_state"}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, opts)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	tk, err := tokenizers.FromFile(tokenPath)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	return &Embedder{
		session:   session,
		tokenizer: tk,
		batchSize: defaultBatchSize,
	}, nil
}

// Close releases the ONNX session and tokenizer.
func (e *Embedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
	if e.tokenizer != nil {
		e.tokenizer.Close()
	}
}

// Embed embeds a batch of document texts (no instruction prefix).
// Use this for indexing document chunks.
func (e *Embedder) Embed(texts []string) ([][]float32, error) {
	results := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += e.batchSize {
		end := i + e.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch, err := e.embedBatch(texts[i:end])
		if err != nil {
			return nil, fmt.Errorf("batch [%d:%d]: %w", i, end, err)
		}
		results = append(results, batch...)
	}
	return results, nil
}

// EmbedQuery embeds a single query string with the BGE instruction prefix.
// Always use this for search queries — never for document chunks.
// The prefix "Represent this sentence for searching relevant passages: "
// is recommended by the BGE authors for asymmetric retrieval tasks.
func (e *Embedder) EmbedQuery(query string) ([]float32, error) {
	vecs, err := e.Embed([]string{BGEQueryPrefix + query})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("empty result for query")
	}
	return vecs[0], nil
}

// encoded holds tokenization results for a single text.
type encoded struct {
	ids  []int64
	mask []int64
}

// embedBatch runs a single ONNX inference call for up to batchSize texts.
// Set SIFT_DEBUG=1 to print per-phase timing to stderr.
func (e *Embedder) embedBatch(texts []string) ([][]float32, error) {
	debug := os.Getenv("SIFT_DEBUG") == "1"
	batchSize := len(texts)
	t0 := time.Now()

	// ── Phase 1: Tokenize ───────────────────────────────────────────────────
	all := make([]encoded, batchSize)
	maxLen := 0
	for i, text := range texts {
		enc := e.tokenizer.EncodeWithOptions(
			text,
			true, // add special tokens (CLS, SEP)
			tokenizers.WithReturnAttentionMask(),
		)
		ids := enc.IDs
		if len(ids) > maxSeqLen {
			ids = ids[:maxSeqLen]
		}
		ids64 := make([]int64, len(ids))
		mask64 := make([]int64, len(ids))
		for j, v := range ids {
			ids64[j] = int64(v)
			mask64[j] = 1
		}
		if len(enc.AttentionMask) >= len(ids) {
			for j := range ids64 {
				mask64[j] = int64(enc.AttentionMask[j])
			}
		}
		all[i] = encoded{ids: ids64, mask: mask64}
		if len(ids64) > maxLen {
			maxLen = len(ids64)
		}
	}
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] tokenize(%d texts, maxLen=%d):   %v\n", batchSize, maxLen, time.Since(t0))
	}

	if maxLen == 0 {
		return nil, fmt.Errorf("all texts tokenized to zero length")
	}

	// ── Phase 2: Build tensors ──────────────────────────────────────────────
	t1 := time.Now()
	flatIDs := make([]int64, batchSize*maxLen)
	flatMask := make([]int64, batchSize*maxLen)
	flatType := make([]int64, batchSize*maxLen) // all zeros (token_type_ids)
	for i, enc := range all {
		copy(flatIDs[i*maxLen:], enc.ids)
		copy(flatMask[i*maxLen:], enc.mask)
	}
	shape := ort.NewShape(int64(batchSize), int64(maxLen))

	inputIDs, err := ort.NewTensor(shape, flatIDs)
	if err != nil {
		return nil, fmt.Errorf("input_ids tensor: %w", err)
	}
	defer inputIDs.Destroy()

	attnMask, err := ort.NewTensor(shape, flatMask)
	if err != nil {
		return nil, fmt.Errorf("attention_mask tensor: %w", err)
	}
	defer attnMask.Destroy()

	typeIDs, err := ort.NewTensor(shape, flatType)
	if err != nil {
		return nil, fmt.Errorf("token_type_ids tensor: %w", err)
	}
	defer typeIDs.Destroy()
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] build tensors:                   %v\n", time.Since(t1))
	}

	// ── Phase 3: ONNX inference ─────────────────────────────────────────────
	t2 := time.Now()
	inputs := []ort.Value{inputIDs, attnMask, typeIDs}
	outputs := []ort.Value{nil}
	if err := e.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("ort run: %w", err)
	}
	defer func() {
		if outputs[0] != nil {
			outputs[0].Destroy()
		}
	}()
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] session.Run (batch=%d, seq=%d): %v\n", batchSize, maxLen, time.Since(t2))
	}

	// ── Phase 4: CLS pool + L2 normalize ────────────────────────────────────
	t3 := time.Now()
	hiddenTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output type (want *Tensor[float32])")
	}
	hidden := hiddenTensor.GetData()
	seqLen := int(hiddenTensor.GetShape()[1])

	embeddings := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		vec := make([]float32, EmbeddingDim)
		// BGE-small uses the [CLS] token (the first token at t=0) as the sentence embedding.
		base := i * seqLen * EmbeddingDim
		for d := 0; d < EmbeddingDim; d++ {
			vec[d] = hidden[base+d]
		}

		l2Normalize(vec)
		embeddings[i] = vec
	}
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] CLS pool + normalize:            %v  (total: %v)\n",
			time.Since(t3), time.Since(t0))
	}

	return embeddings, nil
}

// BenchmarkSingle embeds a single short text and returns phase timings for
// the sift bench command. Returns (tokenizeMs, inferenceMs, totalMs, error).
func (e *Embedder) BenchmarkSingle(text string) (tokenize, inference, total time.Duration, err error) {
	t0 := time.Now()
	enc := e.tokenizer.EncodeWithOptions(text, true, tokenizers.WithReturnAttentionMask())
	ids := enc.IDs
	if len(ids) > maxSeqLen {
		ids = ids[:maxSeqLen]
	}
	tokenize = time.Since(t0)

	ids64 := make([]int64, len(ids))
	mask64 := make([]int64, len(ids))
	flatType := make([]int64, len(ids))
	for j, v := range ids {
		ids64[j] = int64(v)
		mask64[j] = 1
	}
	shape := ort.NewShape(1, int64(len(ids)))
	idsT, e2 := ort.NewTensor(shape, ids64)
	if e2 != nil {
		return 0, 0, 0, e2
	}
	defer idsT.Destroy()
	maskT, e2 := ort.NewTensor(shape, mask64)
	if e2 != nil {
		return 0, 0, 0, e2
	}
	defer maskT.Destroy()
	typT, e2 := ort.NewTensor(shape, flatType)
	if e2 != nil {
		return 0, 0, 0, e2
	}
	defer typT.Destroy()

	t1 := time.Now()
	outputs := []ort.Value{nil}
	if e2 := e.session.Run([]ort.Value{idsT, maskT, typT}, outputs); e2 != nil {
		return 0, 0, 0, e2
	}
	if outputs[0] != nil {
		outputs[0].Destroy()
	}
	inference = time.Since(t1)
	total = time.Since(t0)
	return tokenize, inference, total, nil
}

// l2Normalize normalizes v in-place to unit length.
func l2Normalize(v []float32) {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if norm < 1e-10 {
		return
	}
	inv := float32(1.0 / norm)
	for i := range v {
		v[i] *= inv
	}
}
