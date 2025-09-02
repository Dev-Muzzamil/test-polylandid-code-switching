#!/usr/bin/env python3
"""
Evaluation script for the code-switching language detection system.
"""
import os
import json
import argparse
import logging
from typing import List, Dict

from src.language_detector import CodeSwitchingLanguageDetector
from src.evaluation import LanguageDetectionEvaluator
from src.utils import load_dataset, save_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(args):
    """Main evaluation function."""
    logger.info("Starting evaluation process...")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    data = load_dataset(args.data_path)
    logger.info(f"Loaded {len(data)} samples")
    
    # Initialize detector
    detector = CodeSwitchingLanguageDetector()
    
    # Load trained models
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    detector.load_models(args.model_dir)
    logger.info("Models loaded successfully")
    
    # Generate predictions
    logger.info("Generating predictions...")
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
    logger.info("Evaluating predictions...")
    evaluator = LanguageDetectionEvaluator()
    results = evaluator.evaluate_predictions(predictions, data)
    
    # Print evaluation report
    logger.info("Evaluation Results:")
    all_requirements_met = evaluator.print_evaluation_report(results)
    
    # Add metadata
    results['dataset_size'] = len(data)
    results['all_requirements_met'] = all_requirements_met
    
    # Save detailed results
    if args.save_predictions:
        results['detailed_predictions'] = predictions
    
    # Save results
    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    save_results(results, output_path)
    logger.info(f"Results saved to {output_path}")
    
    # Save prediction examples
    if args.save_examples:
        examples_path = os.path.join(args.output_dir, 'prediction_examples.json')
        examples = []
        
        for i, (pred, gt) in enumerate(zip(predictions[:args.num_examples], data[:args.num_examples])):
            example = {
                'sample_id': i,
                'input_text': gt['text'],
                'ground_truth': gt['spans'],
                'prediction': pred,
                'evaluation': detector.evaluate_sample(pred, gt)
            }
            examples.append(example)
        
        save_results(examples, examples_path)
        logger.info(f"Examples saved to {examples_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate code-switching language detection model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='multilingual_dataset_10k.json',
                       help='Path to the evaluation dataset')
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing trained models')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save all detailed predictions')
    parser.add_argument('--save_examples', action='store_true', default=True,
                       help='Save prediction examples for inspection')
    parser.add_argument('--num_examples', type=int, default=50,
                       help='Number of examples to save')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    results = evaluate_model(args)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    overall = results['overall']
    sentence = results['sentence_level']
    per_lang = results['per_language']
    
    logger.info(f"Overall Results:")
    logger.info(f"  Token Accuracy: {overall['accuracy']:.4f}")
    logger.info(f"  Macro F1: {overall['macro_f1']:.4f} (target: > 0.91)")
    logger.info(f"  Sentence Detection Rate: {sentence['sentence_detection_rate']:.4f} (target: > 0.90)")
    
    # Language-specific results
    min_lang_f1 = min(metrics['f1_score'] for metrics in per_lang.values())
    max_lang_f1 = max(metrics['f1_score'] for metrics in per_lang.values())
    logger.info(f"  Language F1 Range: {min_lang_f1:.4f} - {max_lang_f1:.4f} (target: all > 0.85)")
    
    # Requirements check
    req_overall = overall['macro_f1'] > 0.91
    req_sentence = sentence['sentence_detection_rate'] > 0.90
    req_languages = min_lang_f1 > 0.85
    
    logger.info(f"\nRequirement Status:")
    logger.info(f"  Overall F1 > 0.91: {'✓ PASS' if req_overall else '✗ FAIL'}")
    logger.info(f"  Sentence Acc > 0.90: {'✓ PASS' if req_sentence else '✗ FAIL'}")
    logger.info(f"  All Language F1 > 0.85: {'✓ PASS' if req_languages else '✗ FAIL'}")
    
    if results['all_requirements_met']:
        logger.info("\n🎉 ALL REQUIREMENTS MET! 🎉")
    else:
        logger.info("\n⚠️  Some requirements not met.")
    
    logger.info("="*60)


if __name__ == '__main__':
    main()