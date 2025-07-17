package main

import (
	"fmt"
	"log"
	"os"

	workersai "github.com/ashishdatta/workers-ai-golang/workers-ai"
)

func main() {
	fmt.Println("Cloudflare Workers AI Go Client Example")
	
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	apiToken := os.Getenv("CLOUDFLARE_AUTH_TOKEN")
	
	if accountID == "" || apiToken == "" {
		log.Fatal("Please set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN environment variables")
	}
	
	client := workersai.NewClient(accountID, apiToken)
	
	// Get model information
	/* modelInfo, err := client.GetModelInfo(workersai.ModelLlama4Scout17B)
	if err != nil {
		log.Printf("Error getting model info: %v", err)
	} else {
		fmt.Printf("Model: %s\n", modelInfo.Name)
		fmt.Printf("Description: %s\n", modelInfo.Description)
		fmt.Printf("Task: %s\n", modelInfo.Task.Name)
		fmt.Printf("Max Tokens: %d\n", modelInfo.Properties.MaxTotalTokens)
	} */
	
	// Chat with the model
	messages := []workersai.Message{
		{Role: "system", Content: "You are a friendly assistant"},
		{Role: "user", Content: "Why is pizza so good?"},
	}
	
	response, err := client.Chat(workersai.ModelLlama4Scout17B, messages)
	if err != nil {
		log.Printf("Error sending chat request: %v", err)
	} else {
		fmt.Printf("AI Response: %s\n", response.Result.Response)
	}
	
	// List available models
	// models, err := client.ListModels()
	// if err != nil {
	// 	log.Printf("Error listing models: %v", err)
	// } else {
	// 	fmt.Printf("\nAvailable models (%d):\n", len(models))
	// 	for i, model := range models {
	// 		fmt.Printf("%d. %s - %s\n", i+1, model.Name, model.Description)
	// 		if i >= 4 { // Show only first 5 models
	// 			fmt.Printf("... and %d more\n", len(models)-5)
	// 			break
	// 		}
	// 	}
}
