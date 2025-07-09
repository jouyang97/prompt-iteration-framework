import prompt
from openai import OpenAI
import os
import json
import csv
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fire
from pydantic import BaseModel # Only use for structured outputs

api_key = os.getenv("OPENAI_API_KEY")

def read_inputs_from_directory(input_dir: str) -> list[str]:
    """Read inputs from various file formats in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    inputs = []
    
    for file_path in input_path.rglob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    inputs.extend([line.strip() for line in f if line.strip()])
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        inputs.extend([str(item) for item in data])
                    elif isinstance(data, dict):
                        # If it's a dict, try to extract values
                        for value in data.values():
                            if isinstance(value, str):
                                inputs.append(value)
                            elif isinstance(value, list):
                                inputs.extend([str(item) for item in value])
            elif file_path.suffix.lower() == '.csv':
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:  # Skip empty rows
                            inputs.append(row[0] if len(row) == 1 else ', '.join(row))
    
    return inputs

def call_llm(prompt: str, input: str) -> str:
    """Call the LLM with a given prompt and input."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input}
        ]
    )
    return response.choices[0].message.content

def parallel_calls(inputs: list[str], prompt: str) -> list[str]:
    """Make parallel calls to the LLM."""
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(call_llm, prompt, input) for input in inputs]
        return [future.result() for future in futures]

def write_results_to_directory(results: list[str], output_dir: str, inputs: list[str] = None):
    """Write results to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write individual JSON files for each input-response pair
    for i, result in enumerate(results):
        result_item = {"response": result}
        if inputs and i < len(inputs):
            result_item["input"] = inputs[i]
        
        with open(output_path / f"result_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(result_item, f, indent=2, ensure_ascii=False)

def main(prompt_ver: str, input_dir: str, output_dir: str):
    """
    Main function to process inputs through LLM and save results.
    
    Args:
        prompt_ver: Name of the prompt to use (e.g., 'prompt1', 'prompt2')
        input_dir: Directory containing input files (txt, json, csv)
        output_dir: Directory to save results (will be created if doesn't exist)
    """
    if not hasattr(prompt, prompt_ver):
        raise ValueError(f"Prompt '{prompt_ver}' not found in prompt.py")
    
    selected_prompt = getattr(prompt, prompt_ver)
    
    # Read inputs from the input directory
    print(f"Reading inputs from {input_dir}...")
    inputs = read_inputs_from_directory(input_dir)
    
    if not inputs:
        print("No inputs found in the specified directory")
        return
    
    print(f"Found {len(inputs)} inputs")
    
    print(f"Processing {len(inputs)} inputs through LLM...")
    results = parallel_calls(inputs, selected_prompt)
    
    print(f"Writing {len(results)} results to {output_dir}...")
    write_results_to_directory(results, output_dir, inputs)
    
    print(f"Processed {len(inputs)} inputs and saved results to {output_dir}")

if __name__ == "__main__":
    fire.Fire(main)