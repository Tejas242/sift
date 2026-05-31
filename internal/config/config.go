// Package config manages application configuration loaded from file and environment.
package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/pelletier/go-toml/v2"
)

// Config represents the application configuration.
type Config struct {
	ModelDir  string `toml:"model-dir"`
	OrtLib    string `toml:"ort-lib"`
	Threads   int    `toml:"threads"`
	MaxFileKB int    `toml:"max-file-kb"`
}

const (
	// DefaultModelDir is the default directory where ONNX models are stored.
	DefaultModelDir = "./models"
	// DefaultSiftDir is the default directory where index data is persisted.
	DefaultSiftDir  = ".sift"
	// DefaultOrtLib is the default fallback path to onnxruntime.so.
	DefaultOrtLib   = "./lib/onnxruntime.so"
	// DefaultThreads is the default intra-op thread count for ONNX.
	DefaultThreads  = 0
	// DefaultMaxFile is the default file size skip limit in KB.
	DefaultMaxFile  = 512
)

// Load parses .sift.toml if it exists and returns a Config with merged defaults.
func Load() (*Config, error) {
	cfg := &Config{
		ModelDir:  DefaultModelDir,
		OrtLib:    DefaultOrtLib,
		Threads:   DefaultThreads,
		MaxFileKB: DefaultMaxFile,
	}

	b, err := os.ReadFile(".sift.toml")
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return nil, fmt.Errorf("read .sift.toml: %w", err)
	}

	var fileCfg Config
	if err := toml.Unmarshal(b, &fileCfg); err != nil {
		return nil, fmt.Errorf("parse .sift.toml: %w", err)
	}

	if fileCfg.ModelDir != "" {
		cfg.ModelDir = fileCfg.ModelDir
	}
	if fileCfg.OrtLib != "" {
		cfg.OrtLib = fileCfg.OrtLib
	}
	if fileCfg.Threads > 0 {
		cfg.Threads = fileCfg.Threads
	}
	if fileCfg.MaxFileKB > 0 {
		cfg.MaxFileKB = fileCfg.MaxFileKB
	}

	return cfg, nil
}

// ResolveOrtLib resolves the absolute path of onnxruntime.so.
func ResolveOrtLib(flagPath string) string {
	if flagPath != "" {
		return flagPath
	}
	if exe, err := os.Executable(); err == nil {
		candidate := filepath.Join(filepath.Dir(exe), "lib", "onnxruntime.so")
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	if _, err := os.Stat(DefaultOrtLib); err == nil {
		absPath, _ := filepath.Abs(DefaultOrtLib)
		return absPath
	}
	return ""
}
