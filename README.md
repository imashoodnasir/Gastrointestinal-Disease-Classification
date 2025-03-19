# Gastrointestinal Disease Classification (GID-Xpert)

## Overview
GID-Xpert is a **Hierarchical Multi-Stage Attention-Driven Mixture of Experts Model with Dynamic Routing** for Gastrointestinal Disease Classification. This deep learning framework enhances feature extraction, incorporates expert blocks, and utilizes dynamic routing to improve classification accuracy.

## Features
- **Hierarchical Multi-Stage Architecture**: Progressive feature refinement for robust classification.
- **Mixture of Experts (MoE)**: Distributed learning across specialized expert blocks.
- **Spatial-Channel Attention Mechanism**: Enhances diagnostically meaningful areas while reducing irrelevant information.
- **Squeeze-and-Excitation (SE) Blocks**: Recalibrates feature representations to emphasize key diagnostic patterns.
- **Dynamic Routing Mechanism**: Allocates weights to expert blocks dynamically based on input complexity.
- **Intermediate Auxiliary Classifiers**: Improves feature learning and reduces overfitting.
- **Tree-Structured Parzen Estimator (TPE) Optimization**: Automated hyperparameter tuning for efficiency.

## Datasets
GID-Xpert is trained and evaluated on three benchmark datasets:
- **KAUHC Dataset**: 3301 WCE images categorized into Normal, Arteriovenous Malformations, and Ulcer.
- **WCEBleedGen Dataset**: 1309 bleeding and 1309 non-bleeding frames for automated bleeding detection.
- **GastroEndoNet Dataset**: 4604 images focusing on GERD and gastrointestinal polyps.

## Installation
\`\`\`bash
# Clone the repository
git clone https://github.com/imashoodnasir/Gastrointestinal-Disease-Classification.git
cd Gastrointestinal-Disease-Classification

# Install required dependencies
pip install -r requirements.txt
\`\`\`

## File Structure
\`\`\`
├── data_preprocessing.py      # Handles dataset loading and augmentation
├── feature_extraction.py      # Implements initial feature extraction
├── expert_blocks.py          # Defines expert blocks with attention mechanisms
├── dynamic_routing.py        # Implements the dynamic routing mechanism
├── transition_block.py       # Downsamples feature maps for better abstraction
├── classification_head.py    # Final and auxiliary classification layers
├── gid_xpert.py              # Integrates all components into the main model
├── hyperparameter_optimization.py  # Hyperparameter tuning using TPE
├── train.py                  # Trains the model
├── evaluate.py               # Evaluates model performance
├── README.md                 # Project documentation
\`\`\`

## Training the Model
\`\`\`bash
python train.py
\`\`\`

## Evaluating the Model
\`\`\`bash
python evaluate.py
\`\`\`

## Model Performance
| Dataset         | Accuracy |
|----------------|----------|
| KAUHC         | 99.98%   |
| WCEBleedGen   | 100%     |
| GastroEndoNet | 75.32%   |

## License
This project is licensed under the MIT License.

## Acknowledgments
This work is based on the research paper **"Hierarchical Multi-Stage Attention-Driven Mixture of Experts with Dynamic Routing for Gastrointestinal Disease Classification"**. Special thanks to the authors and contributors of this study.
