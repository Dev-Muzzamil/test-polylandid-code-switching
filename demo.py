#!/usr/bin/env python3
"""
Demo script showing the language detection system components.
This script demonstrates the pipeline without requiring full model training.
"""
import sys
sys.path.append('.')

from src.preprocessing import TextPreprocessor
from src.tokenization import ScriptAwareTokenizer
from src.models import TokenLevelDetector
from src.utils import LANGUAGE_CODES, get_dataset_statistics, load_dataset

def demo_preprocessing():
    """Demonstrate text preprocessing."""
    print("="*60)
    print("PREPROCESSING DEMO")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "Hello 世界 #codeswitching @user",
        "helloworld concatenated text",
        "مرحبا world こんにちは",
        "Visit https://example.com for more info"
    ]
    
    for text in test_texts:
        processed, info = preprocessor.preprocess(text)
        print(f"Original:  {text}")
        print(f"Processed: {processed}")
        print()

def demo_tokenization():
    """Demonstrate script-aware tokenization."""
    print("="*60)
    print("TOKENIZATION DEMO")
    print("="*60)
    
    tokenizer = ScriptAwareTokenizer()
    
    test_texts = [
        "Hello こんにちは world مرحبا",
        "Mixed 中文 русский français",
        "한국어 English ไทย বাংলা"
    ]
    
    for text in test_texts:
        print(f"Text: {text}")
        tokens = tokenizer.tokenize(text)
        
        for token in tokens:
            script = token['script']
            hints = token['language_hints'][:3]  # Show first 3 hints
            print(f"  '{token['text']:10}' -> {script.name:12} -> {hints}")
        print()

def demo_dataset_analysis():
    """Demonstrate dataset analysis."""
    print("="*60)
    print("DATASET ANALYSIS DEMO")
    print("="*60)
    
    try:
        data = load_dataset('multilingual_dataset_10k.json')
        stats = get_dataset_statistics(data)
        
        print(f"Dataset Statistics:")
        print(f"  Total sentences: {stats['total_sentences']:,}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Average tokens per sentence: {stats['avg_tokens_per_sentence']:.2f}")
        print(f"  Average languages per sentence: {stats['avg_languages_per_sentence']:.2f}")
        print()
        
        print("Language distribution:")
        for lang, count in stats['language_counts'].most_common():
            percentage = count / stats['total_tokens'] * 100
            print(f"  {lang.upper()}: {count:5,} tokens ({percentage:5.1f}%)")
        print()
        
        print("Sentence complexity:")
        for num_langs, count in sorted(stats['sentence_language_counts'].items()):
            percentage = count / stats['total_sentences'] * 100
            lang_desc = "monolingual" if num_langs == 1 else f"{num_langs} languages"
            print(f"  {lang_desc:15}: {count:4,} sentences ({percentage:5.1f}%)")
        print()
        
        print("Top language pairs in code-switched sentences:")
        for pair, count in stats['language_pairs'].most_common(10):
            print(f"  {pair}: {count:3,} sentences")
        
    except FileNotFoundError:
        print("Dataset file not found. Please make sure multilingual_dataset_10k.json exists.")
    except Exception as e:
        print(f"Error loading dataset: {e}")

def demo_simple_prediction():
    """Demonstrate simple language prediction without full training."""
    print("="*60)
    print("SIMPLE PREDICTION DEMO")
    print("="*60)
    
    # Initialize components
    preprocessor = TextPreprocessor()
    tokenizer = ScriptAwareTokenizer()
    token_detector = TokenLevelDetector()
    
    # Train basic character n-gram models on sample data
    print("Training basic character n-gram models...")
    try:
        data = load_dataset('multilingual_dataset_10k.json')
        # Use a small subset for quick demo
        sample_data = data[:100]  
        token_detector.train_char_ngram_models(sample_data)
        print("Training completed!")
    except:
        print("Could not load dataset for training. Using script-based predictions only.")
    
    # Test sentences
    test_sentences = [
        "colectivo fascinating мудрость harmonioso",
        "Hello こんにちは world مرحبا",
        "Bonjour I am speaking français and English",
        "这是中文 and this is English text",
        "Mixed भाषा language मिश्रण example"
    ]
    
    for sentence in test_sentences:
        print(f"\nAnalyzing: {sentence}")
        print("-" * 50)
        
        # Preprocess
        processed, _ = preprocessor.preprocess(sentence)
        
        # Tokenize
        tokens = tokenizer.tokenize(processed)
        
        # Predict each token
        predictions = []
        for token_info in tokens:
            token = token_info['text']
            script_hints = [str(token_info['script'])] if token_info['script'] else None
            language_hints = token_info.get('language_hints', [])
            
            # Simple prediction based on script + character n-grams
            pred = token_detector.predict_token(token, language_hints, script_hints)
            
            if pred:
                best_lang = max(pred, key=pred.get)
                confidence = pred[best_lang]
            else:
                # Fallback to script hints
                best_lang = language_hints[0] if language_hints else 'en'
                confidence = 0.5
            
            predictions.append({
                'text': token,
                'language': best_lang,
                'confidence': confidence,
                'script': token_info['script'].name if token_info['script'] else 'UNKNOWN'
            })
        
        # Display results
        for pred in predictions:
            confidence_bar = "█" * int(pred['confidence'] * 10) + "░" * (10 - int(pred['confidence'] * 10))
            print(f"  '{pred['text']:12}' -> {pred['language'].upper():2} ({pred['confidence']:.3f}) {confidence_bar} [{pred['script']}]")

def main():
    """Run all demos."""
    print("Code-Switching Language Detection System Demo")
    print("This demonstrates the core components without full model training.")
    print()
    
    try:
        demo_preprocessing()
        print()
        
        demo_tokenization()
        print()
        
        demo_dataset_analysis()
        print()
        
        demo_simple_prediction()
        print()
        
        print("="*60)
        print("DEMO COMPLETED")
        print("="*60)
        print("To train the full model with all components, run:")
        print("  python train.py")
        print()
        print("To evaluate a trained model, run:")
        print("  python evaluate.py")
        print()
        print("To use the model for prediction, run:")
        print("  python predict.py --interactive")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()