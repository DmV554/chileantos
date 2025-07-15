To modify the README so that it shows only the class names without the count details, you can replace the tables with lists of class names under each dataset section. Here's how the updated README would look:

---

# Dataset README

## Overview

This dataset collection consists of six distinct datasets, each structured to support machine learning tasks involving text classification and detection. The datasets are stored in `DatasetDict` format. The features across all datasets include:

- **text**: The raw text data used for analysis or training.
- **labels**: Encoded labels corresponding to the text data (Integer).
- **human_readable_labels**: Human-readable versions of the labels for easier understanding and interpretation (String).
- **split**: An identifier for the data split (e.g., "train").

## Dataset Details

Below are the details for each dataset in the collection, along with their corresponding file paths and class names:

### 1. Classification - Illegal

- **Path**: `classification/Illegal.jsonl`
- **Classes**: na, lpc pro, cc rc, lpc, lpc int, lpc jus

### 2. Classification - Dark

- **Path**: `classification/Dark.jsonl`
- **Classes**: ltd, cr, nod, er, ch, ter

### 3. Classification - Gray

- **Path**: `classification/Gray.jsonl`
- **Classes**: bfe, des risk, des reser, des uni, des det, des def, des inf, des lic

### 4. Detection - Illegal

- **Path**: `detection/Illegal.jsonl`
- **Classes**: ok, abusive

### 5. Detection - Dark

- **Path**: `detection/Dark.jsonl`
- **Classes**: ok, abusive

### 6. Detection - Gray

- **Path**: `detection/Gray.jsonl`
- **Classes**: ok, abusive

## Usage

To use these datasets, you can load them using libraries such as Hugging Face's `datasets` library. 
Each dataset can be accessed by specifying its path and loading it into a suitable format for your machine learning tasks.
```python
from datasets import load_dataset

# Define the paths to your datasets
paths = [
    "classification/Illegal.jsonl",
    "classification/Dark.jsonl",
    "classification/Gray.jsonl",
    "detection/Illegal.jsonl",
    "detection/Dark.jsonl",
    "detection/Gray.jsonl",
]

# Iterate over each dataset path
for path in paths:
    # Load the dataset
    dataset = load_dataset('json', data_files=path)
    print(dataset)
```

## Citation

Please ensure proper citation.