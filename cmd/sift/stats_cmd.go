package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

func init() {
	rootCmd.AddCommand(&cobra.Command{
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
}
