from statistics import mean, median, stdev 
import fire
import json
from pathlib import Path
from scipy import stats
import numpy as np

def read_judgment_files(input_dir: str) -> list[dict]:
    """Read all judge results JSON files from a directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    judgments = []
    
    for file_path in input_path.glob("*.json"):
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    judgments.append(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue
    
    return judgments

def extract_scores(judgments: list[dict]) -> dict:
    """Extract all scores from judgments into separate lists."""
    scores = {
        'q1_scores': [],
        'q2_scores': [],
        'q3_scores': [],
        'total_scores': []
    }
    
    for judgment in judgments:
        if 'q1_score' in judgment:
            scores['q1_scores'].append(judgment['q1_score'])
        if 'q2_score' in judgment:
            scores['q2_scores'].append(judgment['q2_score'])
        if 'q3_score' in judgment:
            scores['q3_scores'].append(judgment['q3_score'])
        if 'total_score' in judgment:
            scores['total_scores'].append(judgment['total_score'])
    
    return scores

def calculate_basic_stats(scores: dict) -> dict:
    """Calculate mean, median, and standard deviation for each score type."""
    stats_dict = {}
    
    for score_type, score_list in scores.items():
        if score_list:
            stats_dict[score_type] = {
                'mean': mean(score_list),
                'median': median(score_list),
                'stdev': stdev(score_list) if len(score_list) > 1 else 0,
                'count': len(score_list)
            }
    
    return stats_dict

def print_stats(stats_dict: dict):
    """Print statistics in a formatted way."""
    print("\n" + "="*50)
    print("STATISTICS SUMMARY")
    print("="*50)
    
    for score_type, stats in stats_dict.items():
        print(f"\n{score_type.upper().replace('_', ' ')}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Standard Deviation: {stats['stdev']:.2f}")

def perform_t_test(scores1: dict, scores2: dict) -> dict:
    """Perform T-tests between two sets of scores."""
    t_test_results = {}
    
    for score_type in scores1.keys():
        if score_type in scores2 and scores1[score_type] and scores2[score_type]:
            # Perform independent t-test
            t_stat, p_value = stats.ttest_ind(scores1[score_type], scores2[score_type])
            
            t_test_results[score_type] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'group1_count': len(scores1[score_type]),
                'group2_count': len(scores2[score_type]),
                'group1_mean': mean(scores1[score_type]),
                'group2_mean': mean(scores2[score_type])
            }
    
    return t_test_results

def print_comparison(t_test_results: dict):
    """Print T-test results in a formatted way."""
    print("\n" + "="*60)
    print("T-TEST COMPARISON RESULTS")
    print("="*60)
    
    for score_type, results in t_test_results.items():
        print(f"\n{score_type.upper().replace('_', ' ')}:")
        print(f"  Group 1 Mean: {results['group1_mean']:.2f} (n={results['group1_count']})")
        print(f"  Group 2 Mean: {results['group2_mean']:.2f} (n={results['group2_count']})")
        print(f"  T-statistic: {results['t_statistic']:.4f}")
        print(f"  P-value: {results['p_value']:.4f}")
        print(f"  Significant (p<0.05): {'YES' if results['significant'] else 'NO'}")

def stats_mode(input_dir: str):
    """
    Calculate and print basic statistics for judgment scores.
    
    Args:
        input_dir: Directory containing judgment JSON files
    """
    print(f"Reading judgment files from {input_dir}...")
    judgments = read_judgment_files(input_dir)
    
    if not judgments:
        print("No judgment files found in the specified directory")
        return
    
    print(f"Found {len(judgments)} judgment files")
    
    # Extract scores
    scores = extract_scores(judgments)
    
    # Calculate statistics
    stats_dict = calculate_basic_stats(scores)
    
    # Print results
    print_stats(stats_dict)

def compare_mode(dir1: str, dir2: str):
    """
    Compare judgment scores between two directories using T-tests.
    
    Args:
        dir1: First directory containing judgment JSON files
        dir2: Second directory containing judgment JSON files
    """
    print(f"Reading judgment files from {dir1}...")
    judgments1 = read_judgment_files(dir1)
    
    print(f"Reading judgment files from {dir2}...")
    judgments2 = read_judgment_files(dir2)
    
    if not judgments1:
        print(f"No judgment files found in {dir1}")
        return
    
    if not judgments2:
        print(f"No judgment files found in {dir2}")
        return
    
    print(f"Found {len(judgments1)} judgment files in {dir1}")
    print(f"Found {len(judgments2)} judgment files in {dir2}")
    
    # Extract scores from both directories
    scores1 = extract_scores(judgments1)
    scores2 = extract_scores(judgments2)
    
    # Perform T-tests
    t_test_results = perform_t_test(scores1, scores2)
    
    # Print results
    print_comparison(t_test_results)

def main(mode: str, **kwargs):
    """
    Main function for calculating statistics on judgment results.
    
    Args:
        mode: Either 'stats' or 'compare'
        **kwargs: Additional arguments depending on mode
    """
    if mode == "stats":
        if "input_dir" not in kwargs:
            raise ValueError("stats mode requires 'input_dir' argument")
        stats_mode(kwargs["input_dir"])
    
    elif mode == "compare":
        if "dir1" not in kwargs or "dir2" not in kwargs:
            raise ValueError("compare mode requires both 'dir1' and 'dir2' arguments")
        compare_mode(kwargs["dir1"], kwargs["dir2"])
    
    else:
        raise ValueError("Mode must be either 'stats' or 'compare'")

if __name__ == "__main__":
    fire.Fire(main)