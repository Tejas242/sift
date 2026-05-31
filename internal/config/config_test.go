package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoad_NoFile(t *testing.T) {
	// Temporarily run in an empty directory
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	if err := os.Chdir(tmp); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = os.Chdir(origDir)
	}()

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if cfg.ModelDir != DefaultModelDir {
		t.Errorf("expected ModelDir %q, got %q", DefaultModelDir, cfg.ModelDir)
	}
	if cfg.OrtLib != DefaultOrtLib {
		t.Errorf("expected OrtLib %q, got %q", DefaultOrtLib, cfg.OrtLib)
	}
	if cfg.Threads != DefaultThreads {
		t.Errorf("expected Threads %d, got %d", DefaultThreads, cfg.Threads)
	}
	if cfg.MaxFileKB != DefaultMaxFile {
		t.Errorf("expected MaxFileKB %d, got %d", DefaultMaxFile, cfg.MaxFileKB)
	}
}

func TestLoad_WithFile(t *testing.T) {
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	if err := os.Chdir(tmp); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = os.Chdir(origDir)
	}()

	tomlContent := `
model-dir = "./custom-models"
ort-lib = "./custom-lib/onnxruntime.so"
threads = 4
max-file-kb = 1024
`
	if err := os.WriteFile(".sift.toml", []byte(tomlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if cfg.ModelDir != "./custom-models" {
		t.Errorf("expected ModelDir %q, got %q", "./custom-models", cfg.ModelDir)
	}
	if cfg.OrtLib != "./custom-lib/onnxruntime.so" {
		t.Errorf("expected OrtLib %q, got %q", "./custom-lib/onnxruntime.so", cfg.OrtLib)
	}
	if cfg.Threads != 4 {
		t.Errorf("expected Threads %d, got %d", 4, cfg.Threads)
	}
	if cfg.MaxFileKB != 1024 {
		t.Errorf("expected MaxFileKB %d, got %d", 1024, cfg.MaxFileKB)
	}
}

func TestLoad_CorruptFile(t *testing.T) {
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	if err := os.Chdir(tmp); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = os.Chdir(origDir)
	}()

	tomlContent := `
model-dir = "unclosed-string
`
	if err := os.WriteFile(".sift.toml", []byte(tomlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err = Load()
	if err == nil {
		t.Error("expected error loading corrupt TOML file, got nil")
	}
}

func TestResolveOrtLib(t *testing.T) {
	// Custom flagPath provided
	res := ResolveOrtLib("/path/to/custom.so")
	if res != "/path/to/custom.so" {
		t.Errorf("expected ResolveOrtLib to return custom path, got %q", res)
	}

	// FlagPath empty, default file does not exist
	res2 := ResolveOrtLib("")
	if res2 != "" {
		if !filepath.IsAbs(res2) {
			t.Errorf("expected returned library path to be absolute, got %q", res2)
		}
	}
}
