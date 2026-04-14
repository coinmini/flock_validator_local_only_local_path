"""
Parse validation log and reorder evaluation entries by Conv index.

Usage:
    python parse_log.py task21.log
    python parse_log.py task21.log --conv 5        # show only Conv 5
    python parse_log.py task21.log --summary-only   # show only summaries
"""

import re
import sys
import click
from collections import defaultdict


def parse_log(log_path: str):
    """Parse log file and extract generation and evaluation entries."""
    generations = {}  # conv_idx -> list of log lines
    evaluations = defaultdict(list)  # conv_idx -> list of (timestamp, line)
    summaries = {}  # conv_idx -> line
    other_lines = []  # non-conv lines (header, stage info, etc.)
    final_lines = []  # final score lines

    # Patterns
    gen_pattern = re.compile(r'\[Generation (\d+)/\d+\]')
    eval_pattern = re.compile(r'\[Conv (\d+)\] Model:')
    summary_pattern = re.compile(r'\[Conv (\d+) Summary\]')
    final_pattern = re.compile(r'(Raw weighted avg score|Overall normalized|Validation complete|=== .* Reasoning ===)')

    current_gen_idx = None
    current_gen_lines = []

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Generation entry (may span multiple lines)
        gen_match = gen_pattern.search(line)
        if gen_match:
            # Save previous generation if exists
            if current_gen_idx is not None:
                generations[current_gen_idx] = current_gen_lines

            current_gen_idx = int(gen_match.group(1)) - 1  # 0-based
            current_gen_lines = [line]
            i += 1
            # Collect continuation lines (model output) until next log entry
            while i < len(lines):
                next_line = lines[i].rstrip('\n')
                # Check if this is a new log entry (starts with timestamp)
                if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+', next_line):
                    break
                current_gen_lines.append(next_line)
                i += 1
            continue

        # Save any pending generation
        if current_gen_idx is not None and not gen_pattern.search(line):
            # Check if line is a new log entry - save pending gen
            if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+', line) and not gen_pattern.search(line):
                if current_gen_idx is not None:
                    generations[current_gen_idx] = current_gen_lines
                    current_gen_idx = None
                    current_gen_lines = []

        # Evaluation entry (may span multiple lines due to reasoning with newlines)
        eval_match = eval_pattern.search(line)
        if eval_match:
            conv_idx = int(eval_match.group(1))
            eval_line = line
            i += 1
            # Collect continuation lines until next log entry
            while i < len(lines):
                next_line = lines[i].rstrip('\n')
                if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+', next_line):
                    break
                eval_line += ' ' + next_line.strip() if next_line.strip() else ''
                i += 1
            evaluations[conv_idx].append(eval_line)
            continue

        # Summary entry
        summary_match = summary_pattern.search(line)
        if summary_match:
            conv_idx = int(summary_match.group(1))
            summaries[conv_idx] = line
            i += 1
            continue

        # Final score lines
        if final_pattern.search(line):
            final_lines.append(line)
            i += 1
            continue

        # Everything else
        other_lines.append(line)
        i += 1

    # Save last generation if pending
    if current_gen_idx is not None:
        generations[current_gen_idx] = current_gen_lines

    return generations, evaluations, summaries, other_lines, final_lines


def print_section(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


@click.command()
@click.argument("log_path", type=click.Path(exists=True))
@click.option("--conv", type=int, default=None, help="Show only specific Conv index")
@click.option("--summary-only", is_flag=True, help="Show only summaries")
@click.option("--no-generation", is_flag=True, help="Hide generation output")
@click.option("--no-header", is_flag=True, help="Hide header/setup lines")
def main(log_path: str, conv: int | None, summary_only: bool, no_generation: bool, no_header: bool):
    """Parse and reorder validation log by Conv index."""

    generations, evaluations, summaries, other_lines, final_lines = parse_log(log_path)

    total_convs = max(
        max(generations.keys(), default=-1),
        max(evaluations.keys(), default=-1),
        max(summaries.keys(), default=-1),
    ) + 1

    # Header
    if not no_header and not summary_only and conv is None:
        print_section("Setup & Configuration")
        for line in other_lines:
            print(line)

    # Determine which convs to show
    conv_indices = [conv] if conv is not None else range(total_convs)

    for idx in conv_indices:
        if summary_only:
            if idx in summaries:
                print(summaries[idx])
            continue

        print_section(f"Conv {idx}")

        # Generation
        if not no_generation and idx in generations:
            print("--- Generation ---")
            for line in generations[idx]:
                print(line)
            print()

        # Evaluations
        if idx in evaluations:
            print("--- Evaluations ---")
            for line in evaluations[idx]:
                print(line)

        # Summary
        if idx in summaries:
            print()
            print(summaries[idx])

    # Final score
    if final_lines and conv is None:
        print_section("Final Results")
        for line in final_lines:
            print(line)


if __name__ == "__main__":
    main()
