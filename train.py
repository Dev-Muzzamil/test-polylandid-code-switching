#!/usr/bin/env python3
"""
Training script for the code-switching language detection system.
"""
import os
import json
import argparse
import logging
from typing import List, Dict
import random
import numpy as np

from src.language_detector import CodeSwitchingLanguageDetector
from src.evaluation import LanguageDetectionEvaluator
from src.utils import load_dataset, save_results, get_dataset_statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass


def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple:
    """Split dataset into train, validation, and test sets."""
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    return train_data, val_data, test_data


def download_fasttext_model(model_dir: str):
    """Download FastText language identification model if not present."""
    model_path = os.path.join(model_dir, 'lid.176.bin')
    
    if not os.path.exists(model_path):
        logger.info("Downloading FastText language identification model...")
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Downloaded FastText model to {model_path}")
        except Exception as e:
            logger.warning(f"Failed to download FastText model: {e}")
            logger.info("Continuing without FastText model - will use fallback methods")
    
    return model_path if os.path.exists(model_path) else None


def train_model(args):
    """Main training function."""
    logger.info("Starting training process...")
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    data = load_dataset(args.data_path)
    logger.info(f"Loaded {len(data)} samples")
    
    # Print dataset statistics
    stats = get_dataset_statistics(data)
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total sentences: {stats['total_sentences']}")
    logger.info(f"  Total tokens: {stats['total_tokens']}")
    logger.info(f"  Average tokens per sentence: {stats['avg_tokens_per_sentence']:.2f}")
    logger.info(f"  Average languages per sentence: {stats['avg_languages_per_sentence']:.2f}")
    logger.info(f"  Languages: {list(stats['language_counts'].keys())}")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        data, args.train_ratio, args.val_ratio)
    
    logger.info(f"Dataset split:")
    logger.info(f"  Training: {len(train_data)} samples")
    logger.info(f"  Validation: {len(val_data)} samples") 
    logger.info(f"  Test: {len(test_data)} samples")
    
    # Download FastText model if needed
    fasttext_path = download_fasttext_model(args.model_dir)
    
    # Initialize detector
    detector = CodeSwitchingLanguageDetector(
        fasttext_model_path=fasttext_path,
        use_sequence_model=args.use_sequence_model,
        min_confidence_threshold=args.min_confidence
    )
    
    # Load pre-trained models if available
    if os.path.exists(args.model_dir) and any(os.listdir(args.model_dir)):
        try:
            detector.load_models(args.model_dir)
            logger.info("Loaded existing models")
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
    
    # Train the model
    logger.info("Starting model training...")
    detector.train(train_data, val_data)
    
    # Save trained models
    detector.save_models(args.model_dir)
    logger.info(f"Models saved to {args.model_dir}")
    
    # Evaluate on validation set if available
    if val_data:
        logger.info("Evaluating on validation set...")
        val_results = evaluate_model(detector, val_data, "Validation")
        
        # Save validation results
        val_results_path = os.path.join(args.output_dir, 'validation_results.json')
        save_results(val_results, val_results_path)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = evaluate_model(detector, test_data, "Test")
    
    # Save test results
    test_results_path = os.path.join(args.output_dir, 'test_results.json')
    save_results(test_results, test_results_path)
    
    logger.info("Training completed successfully!")
    
    return detector, test_results


def evaluate_model(detector: CodeSwitchingLanguageDetector, 
                  data: List[Dict], 
                  dataset_name: str = "Test") -> Dict:
    """Evaluate model on given dataset."""
    logger.info(f"Evaluating on {len(data)} {dataset_name.lower()} samples...")
    
    # Generate predictions
    predictions = []
    for i, sample in enumerate(data):
        if i % 100 == 0:
            logger.info(f"Processing sample {i+1}/{len(data)}")
        
        text = sample['text']
        try:
            pred = detector.predict(text, return_spans=True, apply_postprocessing=True)
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Failed to predict sample {i}: {e}")
            # Create dummy prediction
            tokens = text.split()
            dummy_pred = {
                'text': text,
                'tokens': tokens,
                'predictions': [{'text': token, 'language': 'en', 'confidence': 0.5} 
                               for token in tokens]
            }
            predictions.append(dummy_pred)
    
    # Evaluate predictions
    evaluator = LanguageDetectionEvaluator()
    results = evaluator.evaluate_predictions(predictions, data)
    
    # Print evaluation report
    logger.info(f"\n{dataset_name} Evaluation Results:")
    all_requirements_met = evaluator.print_evaluation_report(results)
    
    # Add metadata
    results['dataset_name'] = dataset_name
    results['all_requirements_met'] = all_requirements_met
    results['predictions'] = predictions[:10]  # Save first 10 for inspection
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train code-switching language detection model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='multilingual_dataset_10k.json',
                       help='Path to the multilingual dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of data to use for validation')
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save/load models')
    parser.add_argument('--use_sequence_model', action='store_true', default=True,
                       help='Whether to use sequence modeling (BiLSTM-CRF)')
    parser.add_argument('--min_confidence', type=float, default=0.3,
                       help='Minimum confidence threshold for predictions')
    
    # Training arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    detector, results = train_model(args)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    overall = results['overall']
    sentence = results['sentence_level']
    
    logger.info(f"Final Test Results:")
    logger.info(f"  Overall F1: {overall['macro_f1']:.4f} (target: > 0.91)")
    logger.info(f"  Sentence Accuracy: {sentence['sentence_detection_rate']:.4f} (target: > 0.90)")
    
    # Check language-specific F1
    per_lang = results['per_language']
    min_lang_f1 = min(metrics['f1_score'] for metrics in per_lang.values())
    logger.info(f"  Min Language F1: {min_lang_f1:.4f} (target: > 0.85)")
    
    if results['all_requirements_met']:
        logger.info("🎉 ALL REQUIREMENTS MET! 🎉")
    else:
        logger.info("⚠️  Some requirements not met. Check detailed results.")
    
    logger.info("="*60)


if __name__ == '__main__':
    main()