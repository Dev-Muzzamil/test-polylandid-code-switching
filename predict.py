#!/usr/bin/env python3
"""
Prediction script for the code-switching language detection system.
Interactive interface for testing the model on new text.
"""
import os
import argparse
import json
import logging
from typing import List, Dict

from src.language_detector import CodeSwitchingLanguageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_prediction_output(prediction: Dict, verbose: bool = False) -> str:
    """Format prediction output for display."""
    text = prediction['text']
    predictions = prediction['predictions']
    summary = prediction['summary']
    
    output = []
    output.append(f"Input Text: {text}")
    output.append(f"Languages Detected: {', '.join(summary['languages_detected'])}")
    output.append(f"Is Code-Switched: {summary['is_code_switched']}")
    output.append(f"Average Confidence: {summary['avg_confidence']:.3f}")
    output.append("")
    
    # Token-level predictions
    output.append("Token-Level Predictions:")
    for i, pred in enumerate(predictions):
        confidence_bar = "█" * int(pred['confidence'] * 10) + "░" * (10 - int(pred['confidence'] * 10))
        output.append(f"  {i+1:2d}. '{pred['text']:15}' → {pred['language'].upper():2} ({pred['confidence']:.3f}) {confidence_bar}")
    
    # Spans if available
    if 'spans' in prediction:
        output.append("")
        output.append("Language Spans:")
        for i, span in enumerate(prediction['spans']):
            output.append(f"  {i+1}. [{span['language'].upper()}] {span['text']} (conf: {span['confidence']:.3f})")
    
    if verbose and 'preprocessing_info' in prediction:
        output.append("")
        output.append("Preprocessing Info:")
        info = prediction['preprocessing_info']
        if info.get('special_tokens'):
            special = info['special_tokens']
            for token_type, tokens in special.items():
                if tokens:
                    output.append(f"  {token_type}: {len(tokens)} found")
    
    return "\n".join(output)


def interactive_mode(detector: CodeSwitchingLanguageDetector):
    """Interactive mode for testing the model."""
    print("=" * 60)
    print("Code-Switching Language Detection - Interactive Mode")
    print("=" * 60)
    print("Enter text to analyze (or 'quit' to exit)")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'help' - Show this help message")
    print("  'verbose on/off' - Toggle verbose output")
    print("=" * 60)
    
    verbose = False
    
    while True:
        try:
            # Get user input
            text = input("\n> ").strip()
            
            if not text:
                continue
            
            # Handle commands
            if text.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            elif text.lower() == 'help':
                print("\nCommands:")
                print("  'quit' or 'exit' - Exit the program")
                print("  'help' - Show this help message")
                print("  'verbose on/off' - Toggle verbose output")
                continue
            elif text.lower().startswith('verbose'):
                if 'on' in text.lower():
                    verbose = True
                    print("Verbose mode: ON")
                elif 'off' in text.lower():
                    verbose = False
                    print("Verbose mode: OFF")
                else:
                    print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            
            # Make prediction
            print("\nAnalyzing...")
            prediction = detector.predict(text, return_spans=True, apply_postprocessing=True)
            
            # Display results
            print("\n" + "="*60)
            print(format_prediction_output(prediction, verbose))
            print("="*60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(detector: CodeSwitchingLanguageDetector, 
               input_file: str, 
               output_file: str = None,
               verbose: bool = False):
    """Batch mode for processing multiple texts."""
    logger.info(f"Processing batch file: {input_file}")
    
    # Load input texts
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = [item if isinstance(item, str) else item.get('text', str(item)) for item in data]
        else:
            texts = [data.get('text', str(data))]
    else:
        # Assume plain text file with one text per line
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(texts)} texts for processing")
    
    # Process texts
    results = []
    for i, text in enumerate(texts):
        logger.info(f"Processing text {i+1}/{len(texts)}")
        try:
            prediction = detector.predict(text, return_spans=True, apply_postprocessing=True)
            results.append({
                'id': i,
                'input_text': text,
                'prediction': prediction
            })
        except Exception as e:
            logger.error(f"Failed to process text {i}: {e}")
            results.append({
                'id': i,
                'input_text': text,
                'error': str(e)
            })
    
    # Save results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if 'error' not in r)
    logger.info(f"Successfully processed {successful}/{len(texts)} texts")
    
    # Print examples if verbose
    if verbose:
        for result in results[:3]:  # Show first 3 examples
            if 'error' not in result:
                print("\n" + "="*60)
                print(f"Example {result['id'] + 1}:")
                print(format_prediction_output(result['prediction'], verbose))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict languages for code-switched text')
    
    # Model arguments
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing trained models')
    
    # Input arguments
    parser.add_argument('--text', type=str,
                       help='Single text to analyze')
    parser.add_argument('--input_file', type=str,
                       help='File containing texts to analyze (JSON or plain text)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    # Output arguments
    parser.add_argument('--output_file', type=str,
                       help='Output file for batch results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Load model
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        print("Please train the model first using train.py")
        return
    
    print("Loading models...")
    detector = CodeSwitchingLanguageDetector()
    detector.load_models(args.model_dir)
    print("Models loaded successfully!")
    
    # Determine mode
    if args.interactive:
        interactive_mode(detector)
    elif args.input_file:
        batch_mode(detector, args.input_file, args.output_file, args.verbose)
    elif args.text:
        # Single text prediction
        prediction = detector.predict(args.text, return_spans=True, apply_postprocessing=True)
        print("\n" + "="*60)
        print(format_prediction_output(prediction, args.verbose))
        print("="*60)
    else:
        # Default to interactive mode
        interactive_mode(detector)


if __name__ == '__main__':
    main()