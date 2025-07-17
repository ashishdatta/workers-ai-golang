package workersai

const (
	// Chat models
	ModelLlama4Scout17B      = "@cf/meta/llama-4-scout-17b-16e-instruct"
	ModelLlama38B           = "@cf/meta/llama-3-8b-instruct"
	ModelLlama370B          = "@cf/meta/llama-3-70b-instruct"
	ModelMistral7B          = "@cf/mistral/mistral-7b-instruct-v0.1"
	ModelCodeLlama7B        = "@cf/meta/code-llama-7b-instruct"
	ModelQwen330ba3b        = "@cf/qwen/qwen3-30b-a3b-fp8"
	
	// Image generation models
	ModelStableDiffusion    = "@cf/stabilityai/stable-diffusion-xl-base-1.0"
	ModelDreamshaper        = "@cf/lykon/dreamshaper-8-lcm"
	
	// Text-to-speech models
	ModelSpeechT5          = "@cf/microsoft/speecht5-tts"
	
	// Embedding models
	ModelBAAI              = "@cf/baai/bge-base-en-v1.5"
	ModelBAAILarge         = "@cf/baai/bge-large-en-v1.5"
	
	// Translation models
	ModelM2M100            = "@cf/meta/m2m100-1.2b"
)
