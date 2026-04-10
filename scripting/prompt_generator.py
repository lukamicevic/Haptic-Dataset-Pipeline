#!/usr/bin/env python3
"""
Generate prompt variations for haptic signal operations using Qwen via llama.cpp
"""

import json
from pathlib import Path
from llama_cpp import Llama

# Operations and their descriptions
OPERATIONS = {
    "add_noise": {
        "description": "Add white noise to make the haptic signal feel rougher/grainier",
        "seed_prompts": ["make it rougher", "add texture", "add grit"]
    },
    "smooth_signal": {
        "description": "Apply moving average smoothing to reduce harshness",
        "seed_prompts": ["smooth it out", "make it softer", "reduce harshness"]
    },
    "butterworth_lowpass": {
        "description": "Apply lowpass filter to remove harsh high frequencies",
        "seed_prompts": ["remove harsh frequencies", "make it warmer", "less sharp"]
    },
    "insert_signal": {
        "description": "Insert a haptic pulse or tap at a position",
        "seed_prompts": ["add a pulse", "insert a tap", "add a click"]
    },
    "remove_signal": {
        "description": "Remove a section of the haptic signal",
        "seed_prompts": ["remove the section", "delete this part", "cut out the middle"]
    },
    "mix_signals": {
        "description": "Blend/overlay another haptic signal",
        "seed_prompts": ["blend with", "overlay", "combine with"]
    },
    "mask_signal": {
        "description": "Mute/silence a portion of the signal",
        "seed_prompts": ["mute this section", "silence the middle", "blank out"]
    },
    "amplitude_increase": {
        "description": "Make the haptic vibration stronger/more intense",
        "seed_prompts": ["make it stronger", "increase intensity", "more powerful"]
    },
    "amplitude_decrease": {
        "description": "Make the haptic vibration weaker/more subtle",
        "seed_prompts": ["make it subtle", "reduce intensity", "softer vibration"]
    },
    "downsample": {
        "description": "Reduce the resolution/quality of the signal",
        "seed_prompts": ["lower quality", "reduce resolution", "coarser samples"]
    }
}


def load_model(model_path: str = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"):
    """Load the Qwen model"""
    print(f"Loading model from {model_path}...")
    return Llama(model_path=model_path, n_ctx=512, verbose=False)


def generate_prompts(llm, operation: str, description: str, num_variations: int = 10) -> list[str]:
    """Generate prompt variations for an operation using Qwen"""

    system_msg = """You are helping create training data for a haptic signal editing AI.
Generate short, natural command prompts (3-8 words each) that users might say to request a specific haptic editing operation.
Keep them varied and natural sounding. Output ONLY the prompts, one per line, no numbering."""

    user_msg = f"""Generate {num_variations} different ways to say a command for this haptic editing operation:

Operation: {operation}
What it does: {description}

Output {num_variations} short command phrases (3-8 words each), one per line:"""

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=256,
        temperature=0.8
    )

    response = output['choices'][0]['message']['content']

    # Parse response into individual prompts
    prompts = []
    for line in response.strip().split('\n'):
        line = line.strip()
        # Remove numbering if present
        if line and line[0].isdigit():
            line = line.lstrip('0123456789.-) ').strip()
        # Remove quotes if present
        line = line.strip('"\'')
        if line and len(line) > 2:
            prompts.append(line.lower())

    return prompts


def main():
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    llm = load_model(str(model_path))

    # Generate prompts for each operation
    all_prompts = {}

    for op_name, op_info in OPERATIONS.items():
        print(f"\nGenerating prompts for: {op_name}")

        # Generate variations
        generated = generate_prompts(llm, op_name, op_info["description"], num_variations=10)

        # Combine with seed prompts and deduplicate
        all_prompts[op_name] = {
            "description": op_info["description"],
            "prompts": list(set(op_info["seed_prompts"] + generated))
        }

        print(f"  Generated {len(all_prompts[op_name]['prompts'])} unique prompts")
        for p in all_prompts[op_name]['prompts'][:5]:
            print(f"    - {p}")

    # Save to JSON
    output_path = Path(__file__).parent / "generated_prompts.json"
    with open(output_path, 'w') as f:
        json.dump(all_prompts, f, indent=2)

    print(f"\n✓ Saved prompts to {output_path}")

    # Print summary
    total = sum(len(v['prompts']) for v in all_prompts.values())
    print(f"\nTotal: {total} prompts across {len(all_prompts)} operations")


if __name__ == "__main__":
    main()
