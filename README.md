# Prompt Iteration Framework

A comprehensive framework for iterating on LLM prompts, generating responses, evaluating them, and analyzing the results statistically.

## Overview

This framework provides a complete pipeline for prompt engineering:

1. **Generate LLM responses** from inputs using different prompts
2. **Evaluate response quality** using automated judging
3. **Analyze results** with statistical comparisons

## Components

### 1. `call_llm.py` - Generate LLM Responses
Generates responses from inputs using specified prompts.

**Usage:**
```bash
python call_llm.py main <prompt_name> <input_dir> <output_dir>
```

**Arguments:**
- `prompt_name`: Name of the prompt to use (e.g., 'prompt1', 'prompt2')
- `input_dir`: Directory containing input files (supports .txt, .json, .csv)
- `output_dir`: Directory to save results (created if doesn't exist)

**Example:**
```bash
python call_llm.py main prompt1 ./inputs ./llm_results
```

**Input Formats:**
- **TXT files**: Each line becomes an input
- **JSON files**: Supports arrays and objects
- **CSV files**: Each row becomes an input

**Output:**
- Individual JSON files: `result_1.json`, `result_2.json`, etc.
- Each file contains: `{"input": "...", "response": "..."}`

### 2. `judge_responses.py` - Evaluate Response Quality
Evaluates LLM responses using automated judging criteria.

**Usage:**
```bash
python judge_responses.py main <input_dir> <output_dir>
```

**Arguments:**
- `input_dir`: Directory containing JSON files with input-response pairs
- `output_dir`: Directory to save judgment results

**Example:**
```bash
python judge_responses.py main ./llm_results ./judgments
```

**Input:**
- JSON files from `call_llm.py` output
- Each file must contain `input` and `response` keys

**Output:**
- Individual JSON files: `judgment_1.json`, `judgment_2.json`, etc.
- Each file contains:
  ```json
  {
    "input": "original input",
    "response": "LLM response",
    "q1": "question 1 answer",
    "q1_score": 4,
    "q2": "question 2 answer", 
    "q2_score": 3,
    "q3": "question 3 answer",
    "q3_score": 5,
    "total_score": 12
  }
  ```

### 3. `calc_stats.py` - Statistical Analysis
Analyzes judgment results with two modes.

#### Stats Mode - Basic Statistics
**Usage:**
```bash
python calc_stats.py main stats --input_dir <judgment_dir>
```

**Example:**
```bash
python calc_stats.py main stats --input_dir ./judgments
```

**Output:**
- Mean, median, standard deviation for each question score
- Total score statistics
- Sample counts

#### Compare Mode - T-Test Comparison
**Usage:**
```bash
python calc_stats.py main compare --dir1 <dir1> --dir2 <dir2>
```

**Example:**
```bash
python calc_stats.py main compare --dir1 ./judgments_prompt1 --dir2 ./judgments_prompt2
```

**Output:**
- T-test results between two directories
- P-values and significance testing
- Group means and sample sizes

### 4. `prompt.py` - Prompt Definitions
Define your prompts here.

**Example:**
```python
prompt1 = dedent("""
    You are a helpful assistant. Answer the following question:
""")

prompt2 = dedent("""
    You are an expert consultant. Provide detailed analysis for:
""")
```

## Complete Workflow Example

```bash
# 1. Generate responses with different prompts
python call_llm.py main prompt1 ./inputs ./results_prompt1
python call_llm.py main prompt2 ./inputs ./results_prompt2

# 2. Evaluate responses
python judge_responses.py main ./results_prompt1 ./judgments_prompt1
python judge_responses.py main ./results_prompt2 ./judgments_prompt2

# 3. Analyze results
python calc_stats.py main stats --input_dir ./judgments_prompt1
python calc_stats.py main compare --dir1 ./judgments_prompt1 --dir2 ./judgments_prompt2
```

## Setup

### Prerequisites
```bash
pip install openai fire pydantic scipy numpy
```

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Customizing Judging Criteria

Edit the system prompt in `judge_responses.py`:

```python
system_prompt = dedent("""Your task is to evaluate the quality of a response from an LLM.
                           You will be given a user input and the LLM's response to that input.
                           To accomplish your task, ask and answer the following questions about the response with a comprehensive analysis, then score the response on a scale from 0 to 5 where 0 is the worst and 5 is the best.
                           1. INSERT QUESTION 1 HERE
                           2. INSERT QUESTION 2 HERE
                           3. INSERT QUESTION 3 HERE
                           Respond in JSON.
                       """)
```

## File Structure

```
prompt-iteration-framework/
├── call_llm.py          # Generate LLM responses
├── judge_responses.py   # Evaluate response quality
├── calc_stats.py        # Statistical analysis
├── prompt.py           # Prompt definitions
└── README.md           # This file
```

## Features

- **Parallel Processing**: All scripts use ThreadPoolExecutor for efficient processing
- **Flexible Input Formats**: Support for TXT, JSON, and CSV files
- **Error Handling**: Graceful handling of malformed files and API errors
- **Statistical Rigor**: Proper T-tests and significance testing
- **CLI Interface**: Easy-to-use command-line interface with Fire
- **Modular Design**: Each component can be used independently