package hnsw

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
)

// magic is the file header for sift HNSW binary files.
var magic = [4]byte{'S', 'H', 'N', 'W'}

const formatVersion = uint16(1)

// Save serializes the graph to a binary file.
// Format:
//
//	[4]byte  magic
//	uint16   version
//	uint32   nodeCount
//	uint32   entryPoint
//	uint8    maxLayer
//	uint16   m
//	uint16   efConstruction
//	uint16   efSearch
//	--- per node ---
//	uint8    layerCount (= maxLayer for this node + 1)
//	uint16   vecLen
//	float32  vec[vecLen]
//	--- per layer in node ---
//	uint16   neighborCount
//	uint32   neighbor[neighborCount]
func (g *Graph) Save(path string) error {
	g.mu.RLock()
	defer g.mu.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	w := &binaryWriter{w: f}

	w.write(magic)
	w.writeU16(formatVersion)
	w.writeU32(uint32(len(g.nodes)))
	w.writeU32(g.entryPoint)
	w.writeU8(uint8(g.maxLayer))
	w.writeU16(uint16(g.m))
	w.writeU16(uint16(g.efConstruction))
	w.writeU16(uint16(g.efSearch))

	for _, n := range g.nodes {
		w.writeU8(uint8(len(n.neighbors)))
		w.writeU16(uint16(len(n.vec)))
		for _, v := range n.vec {
			w.writeF32(v)
		}
		for _, layer := range n.neighbors {
			w.writeU16(uint16(len(layer)))
			for _, nb := range layer {
				w.writeU32(nb)
			}
		}
	}

	return w.err
}

// Load deserializes a graph from a binary file previously written by Save.
func Load(path string) (*Graph, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	r := &binaryReader{r: f}

	var gotMagic [4]byte
	r.read(&gotMagic)
	if gotMagic != magic {
		return nil, fmt.Errorf("invalid magic bytes in %s — graph may be corrupted", path)
	}

	version := r.readU16()
	if version != formatVersion {
		return nil, fmt.Errorf("unsupported version %d (expected %d)", version, formatVersion)
	}

	nodeCount := r.readU32()
	entryPoint := r.readU32()
	maxLayer := int(r.readU8())
	m := int(r.readU16())
	efConstruction := int(r.readU16())
	efSearch := int(r.readU16())

	if r.err != nil {
		return nil, fmt.Errorf("read header: %w", r.err)
	}

	nodes := make([]node, nodeCount)
	for i := range nodes {
		layerCount := int(r.readU8())
		vecLen := int(r.readU16())
		vec := make([]float32, vecLen)
		for j := range vec {
			vec[j] = r.readF32()
		}
		neighbors := make([][]uint32, layerCount)
		for l := range neighbors {
			nbCount := int(r.readU16())
			neighbors[l] = make([]uint32, nbCount)
			for j := range neighbors[l] {
				neighbors[l][j] = r.readU32()
			}
		}
		nodes[i] = node{vec: vec, neighbors: neighbors}
	}

	if r.err != nil {
		return nil, fmt.Errorf("read nodes: %w", r.err)
	}

	g := &Graph{
		nodes:          nodes,
		entryPoint:     entryPoint,
		maxLayer:       maxLayer,
		m:              m,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		rng:            rand.New(rand.NewSource(42)),
	}
	import_ml(g)
	return g, nil
}

// import_ml recalculates the level factor from m (needed after deserialization).
func import_ml(g *Graph) {
	if g.m > 0 {
		g.ml = 1.0 / math.Log(float64(g.m))
	}
}

// binaryWriter wraps an io.Writer and accumulates the first error.
type binaryWriter struct {
	w   io.Writer
	err error
}

func (bw *binaryWriter) write(v interface{}) {
	if bw.err != nil {
		return
	}
	bw.err = binary.Write(bw.w, binary.LittleEndian, v)
}
func (bw *binaryWriter) writeU8(v uint8)    { bw.write(v) }
func (bw *binaryWriter) writeU16(v uint16)  { bw.write(v) }
func (bw *binaryWriter) writeU32(v uint32)  { bw.write(v) }
func (bw *binaryWriter) writeF32(v float32) { bw.write(v) }

// binaryReader wraps an io.Reader and accumulates the first error.
type binaryReader struct {
	r   io.Reader
	err error
}

func (br *binaryReader) read(v interface{}) {
	if br.err != nil {
		return
	}
	br.err = binary.Read(br.r, binary.LittleEndian, v)
}
func (br *binaryReader) readU8() uint8 {
	var v uint8
	br.read(&v)
	return v
}
func (br *binaryReader) readU16() uint16 {
	var v uint16
	br.read(&v)
	return v
}
func (br *binaryReader) readU32() uint32 {
	var v uint32
	br.read(&v)
	return v
}
func (br *binaryReader) readF32() float32 {
	var v float32
	br.read(&v)
	return v
}
