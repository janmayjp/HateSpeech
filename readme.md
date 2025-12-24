# Evaluating the Impact of Explainability on Bias and Misclassification in Hate Speech Detection

## Overview
This repository contains the implementation of a research project that evaluates transformer-based hate speech detection models through the lens of explainability and fairness. The project investigates how explainable AI (XAI) techniques can be used to uncover demographic bias and misclassification patterns in automated content moderation systems, while maintaining strong classification performance.

## Objectives
- Evaluate BERT, RoBERTa, and DistilBERT for hate speech classification  
- Address severe class imbalance using SMOTE  
- Apply multiple explainability methods to interpret model decisions  
- Quantify demographic bias across protected attribute groups  
- Analyze misclassification patterns, especially between hate speech and offensive language  

## Dataset
- Hate Speech and Offensive Language Dataset  
- 24,783 tweets labeled as:
  - Hate Speech  
  - Offensive Language  
  - Neutral  
- Highly imbalanced (~5.8% hate speech); SMOTE applied only to the training set  

## Models
The following transformer architectures are fine-tuned for 3-class classification:
- BERT-base-uncased  
- RoBERTa-base  
- DistilBERT-base-uncased  

All models achieve approximately 90% accuracy, with RoBERTa achieving the highest weighted F1-score.

## Explainability Framework
A unified explainability pipeline is implemented using:
- LIME – local, model-agnostic word-level explanations  
- SHAP – game-theoretic feature attribution  
- Integrated Gradients – gradient-based attribution  
- Attention Visualization – transformer attention analysis  

Cross-method agreement is used to improve explanation reliability and identify spurious feature associations.

## Bias and Fairness Analysis
Bias is evaluated across five demographic subgroups:
- Race  
- Religion  
- Gender  
- Sexuality  
- Disability  

Key fairness metrics include:
- Subgroup F1-score  
- Fairness gap (overall F1 − subgroup F1)  
- False positive rate for hate speech  

Results show systematic over-classification of identity-related content, particularly for race and sexuality, driven by learned associations between demographic terms and hate labels.

## Key Findings
- SMOTE improves hate speech F1-score from 0.72 to ~0.88  
- High overall accuracy does not imply fairness  
- Identity terms receive disproportionate attention, leading to false positives  
- Bias patterns are consistent across all tested architectures  
- Explainability methods are essential for diagnosing unfair behavior  

## Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- imbalanced-learn
- LIME, SHAP, Captum
- pandas, numpy, matplotlib, seaborn

## Reproducibility
- Fixed random seeds  
- Stratified data splits  
- No test-set leakage  
- Saved checkpoints and logged hyperparameters  

## Disclaimer
This project is intended for research and academic purposes only. It highlights limitations of current hate speech detection systems and demonstrates how explainable AI can be used to audit and understand model behavior.

## Citation
If you use this work, please cite:

Evaluating the Impact of Explainability on Bias and Misclassification in Hate Speech Detection  
Janmay Panchal, Haard Patel  
Pandit Deendayal Energy University, 2025
