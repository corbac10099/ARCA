import argparse
import sys
from .arca_core import PyArcaModel

def main():
    parser = argparse.ArgumentParser(description="ARCA LLM Inference CLI")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the .sovereign model file")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("-n", "--max-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature for sampling")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading model from {args.model}...", file=sys.stderr)
        model = PyArcaModel(args.model)
        
        print(f"Generating {args.max_tokens} tokens...", file=sys.stderr)
        tokens = model.generate(args.prompt, args.max_tokens, args.temp)
        text = model.decode(tokens)
        print("Generated Text:", text)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
