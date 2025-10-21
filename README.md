# DX 703 Final Project

## Overview

This repository contains our team's final project for DX 703, focusing on **multi-class classification** using deep learning. The project involves exploratory data analysis, model development, and performance evaluation on one of two datasets:

- **Food-101**: Image classification across 101 food categories (~101,000 images)
- **HuffPost News Category**: Text classification across 41 news topics (~200,000 articles)

## Team Members

- [Team Member 1 - August Siu]
- [Team Member 2 - Peter Shmorhun]
- [Team Member 3 - Emma Stiller]

## Project Structure

```
dx_703_final_project/
├── Milestone_01.ipynb          # Initial EDA and dataset selection
├── Milestone_02.ipynb          # Model development (coming soon)
├── Milestone_03.ipynb          # Final evaluation (coming soon)
├── data/                       # Dataset storage (gitignored)
├── models/                     # Trained model checkpoints (gitignored)
├── README.md                   # This file
└── requirements.txt            # Python dependencies (to be added)
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- GPU recommended (for training deep learning models)

### Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd dx_703_final_project
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: If `requirements.txt` doesn't exist yet, you'll need to install the following core packages:
```bash
pip install tensorflow numpy pandas matplotlib datasets pillow spacy tqdm
python -m spacy download en_core_web_sm  # For text processing
```

### Running the Notebooks

Launch Jupyter:
```bash
jupyter notebook
```

Open [Milestone_01.ipynb](Milestone_01.ipynb) to begin with exploratory data analysis.

## Dataset Information

### Food-101 (Image Classification)
- **Size**: ~101,000 color images
- **Classes**: 101 food categories
- **Split**: ~800 train / 100 validation / 100 test per class
- **Challenges**: Varying lighting, composition, color balance, image sizes
- **Source**: Hugging Face `food101` dataset

### HuffPost News Category (Text Classification)
- **Size**: ~200,000 news articles
- **Classes**: 41 topical categories
- **Format**: Headline + short description (concatenated with `[SEP]` token)
- **Challenges**: Class imbalance, overlapping categories, varying text lengths
- **Source**: Hugging Face JSON dataset

## Milestones

### Milestone 1: EDA and Problem Framing (Due: October 26)
- Form team and complete team contract
- Select dataset (Food-101 or HuffPost)
- Perform exploratory data analysis
- Identify challenges and propose solutions
- Define evaluation metrics

### Milestone 2: Model Development (TBD)
- Build baseline model
- Implement data preprocessing pipeline
- Train and tune deep learning model
- Evaluate performance

### Milestone 3: Final Evaluation (TBD)
- Final model evaluation
- Performance analysis
- Results presentation
- Project documentation

## Workflow Guidelines

### Branch Strategy
- `main`: Stable, working code only
- `dev`: Development branch for integration
- Feature branches: `feature/[feature-name]` for individual work

### Collaboration Best Practices

1. **Before starting work**:
   ```bash
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Committing changes**:
   ```bash
   git add [files]
   git commit -m "Clear, descriptive message"
   git push origin feature/your-feature-name
   ```

3. **Merging changes**:
   - Create a Pull Request on GitHub
   - Have at least one team member review
   - Merge after approval

### Working with Notebooks

- **Clear outputs before committing** to reduce merge conflicts:
  ```bash
  jupyter nbconvert --clear-output --inplace Milestone_01.ipynb
  ```
- Communicate with team before working on the same notebook
- Consider splitting work into separate notebooks when possible

## Data Management

Large files (datasets, model checkpoints) are excluded from version control via [.gitignore](.gitignore).

To share data among team members:
1. Store datasets in a shared cloud location (Google Drive, Dropbox, etc.)
2. Document download instructions in this README
3. Use the same folder structure: `data/` for datasets, `models/` for checkpoints

## Resources

- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- Course materials and coding notebooks

## License

This project is for educational purposes as part of DX 703 coursework.
