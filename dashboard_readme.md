
# Political Even-handedness Evaluation Dashboard

## Overview

This interactive dashboard provides a comprehensive interface for exploring and testing Anthropic's political even-handedness evaluation framework for Large Language Models (LLMs).

## Purpose

The dashboard enables researchers, developers, and evaluators to:

1. **Explore the evaluation dataset** - Browse 150 political topics across 60 categories with paired prompts representing opposing political stances
2. **Visualize data distributions** - Analyze prompt patterns, topic coverage, and stance distributions through interactive charts
3. **Understand evaluation methodology** - Review the three core evaluation dimensions and their grading criteria
4. **Test evaluation prompts** - Generate formatted evaluation prompts using custom inputs to understand the grading system

## Core Features

### 1. Dataset Overview
- Statistical summary of prompts, topics, and categories
- Visual distributions by main category, template type, and topic
- Prompt group analysis showing stance balance

### 2. Prompt Explorer
- Interactive browsing of prompt pairs by topic and template
- Side-by-side comparison of opposing stances
- Metadata display including partisan classification

### 3. Evaluation Templates
- **Even-handedness**: Measures equal helpfulness across opposing political positions
- **Refusal**: Assesses engagement vs. declining to respond to political content
- **Opposing Perspectives**: Evaluates acknowledgment of counterarguments

### 4. Test Evaluator
- Interactive prompt generation tool
- Supports all three evaluation dimensions
- Helps users understand how responses are graded

## Use Cases

- **Research**: Analyze political bias patterns in LLM evaluation datasets
- **Development**: Test and validate even-handedness evaluation approaches
- **Education**: Learn about systematic political bias assessment in AI systems
- **Quality Assurance**: Verify evaluation prompt formatting and structure

## Technical Stack

- **Framework**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas
- **Dataset**: eval_set.csv (1,200 prompt pairs)

## Getting Started

```bash
# Install dependencies
pip install streamlit pandas plotly

# Run the dashboard
streamlit run app.py
