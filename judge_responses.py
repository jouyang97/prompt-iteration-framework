from openai import OpenAI
import os
import json
from pathlib import Path
from textwrap import dedent
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import fire

api_key = os.getenv("OPENAI_API_KEY")

class JudgeResponse(BaseModel):
    q1: str
    q1_score: int
    q2: str
    q2_score: int
    q3: str
    q3_score: int
    # You can add more questions here if you want, but remember to add the questions to the system prompt.
    total_score: int

def read_input_response_pairs(input_dir: str) -> list[tuple[str, str]]:
    """Read input-response pairs from JSON files in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    pairs = []
    
    for file_path in input_path.glob("*.json"):
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "input" in data and "response" in data:
                        pairs.append((data["input"], data["response"]))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue
    
    return pairs

def judge_llm(input: str, llm_response: str) -> JudgeResponse:
    """Judge a single LLM response."""
    client = OpenAI(api_key=api_key)
    system_prompt = dedent("""Your task is to evaluate the quality of a response from an LLM.
                           You will be given a user input and the LLM's response to that input.
                           To accomplish your task, ask and answer the following questions about the response to the input.
                           Then score the response on a scale from 0 to 5 where 0 is the worst and 5 is the best.
                           1. INSERT QUESTION 1 HERE
                           2. INSERT QUESTION 2 HERE
                           3. INSERT QUESTION 3 HERE
                           Respond in JSON.
                           """)
    response = client.responses.parse(
        model="gpt-4.1",
        temperature=0.0,
        response_format=JudgeResponse,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": dedent(f"""Here is the user input:
                                               <user_input>
                                               {input}
                                               </user_input>
                                               
                                               Here is the LLM's response:
                                               <llm_response>
                                               {llm_response}
                                               </llm_response>
                                               
                                               How well did the did the LLM respond to the input?""")},
        ]
    )
    
    # Calculate total score from individual scores
    parsed_response = response.parsed
    total_score = parsed_response.q1_score + parsed_response.q2_score + parsed_response.q3_score
    
    # Create new JudgeResponse with total_score
    return JudgeResponse(
        q1=parsed_response.q1,
        q1_score=parsed_response.q1_score,
        q2=parsed_response.q2,
        q2_score=parsed_response.q2_score,
        q3=parsed_response.q3,
        q3_score=parsed_response.q3_score,
        total_score=total_score
    )

def parallel_judging(input_response_pairs: list[tuple[str, str]]) -> list[JudgeResponse]:
    """Judge multiple input-response pairs in parallel."""
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(judge_llm, input_text, response) 
                  for input_text, response in input_response_pairs]
        return [future.result() for future in futures]

def write_judgments_to_directory(judgments: list[JudgeResponse], output_dir: str, 
                                input_response_pairs: list[tuple[str, str]] = None):
    """Write judgment results to the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write individual JSON files for each judgment
    for i, judgment in enumerate(judgments):
        judgment_data = judgment.model_dump()
        if input_response_pairs and i < len(input_response_pairs):
            input_text, response = input_response_pairs[i]
            judgment_data["input"] = input_text
            judgment_data["response"] = response
        
        with open(output_path / f"judgment_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(judgment_data, f, indent=2, ensure_ascii=False)

def main(input_dir: str, output_dir: str):
    """
    Main function to judge LLM responses and save results.
    
    Args:
        input_dir: Directory containing JSON files with input-response pairs
        output_dir: Directory to save judgment results (will be created if doesn't exist)
    """
    print(f"Reading input-response pairs from {input_dir}...")
    input_response_pairs = read_input_response_pairs(input_dir)
    
    if not input_response_pairs:
        print("No input-response pairs found in the specified directory")
        return
    
    print(f"Found {len(input_response_pairs)} input-response pairs")
    
    print(f"Judging {len(input_response_pairs)} responses...")
    judgments = parallel_judging(input_response_pairs)
    
    print(f"Writing {len(judgments)} judgments to {output_dir}...")
    write_judgments_to_directory(judgments, output_dir, input_response_pairs)
    
    print(f"Judged {len(input_response_pairs)} responses and saved results to {output_dir}")

if __name__ == "__main__":
    fire.Fire(main)