package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/spf13/cobra"
)

var jsonExport bool

func init() {
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
	rootCmd.AddCommand(searchCmd)
}
