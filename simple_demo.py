#!/usr/bin/env python3
"""
Simple demo script showing the language detection system components.
This script demonstrates basic functionality without requiring heavy dependencies.
"""
import sys
sys.path.append('.')

from src.preprocessing import TextPreprocessor
from src.tokenization import ScriptAwareTokenizer
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

def demo_script_based_prediction():
    """Demonstrate simple script-based language prediction."""
    print("="*60)
    print("SCRIPT-BASED PREDICTION DEMO")
    print("="*60)
    
    # Initialize components
    preprocessor = TextPreprocessor()
    tokenizer = ScriptAwareTokenizer()
    
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
        
        # Predict based on script and language hints
        for token_info in tokens:
            token = token_info['text']
            script = token_info['script']
            language_hints = token_info.get('language_hints', [])
            
            # Simple prediction based on script hints
            if language_hints:
                # Use first hint as primary prediction
                predicted_lang = language_hints[0]
                confidence = 0.8 if len(language_hints) == 1 else 0.6
            else:
                # Fallback to English
                predicted_lang = 'en'
                confidence = 0.3
            
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            script_name = script.name if script else 'UNKNOWN'
            print(f"  '{token:12}' -> {predicted_lang.upper():2} ({confidence:.3f}) {confidence_bar} [{script_name}]")

def show_language_families():
    """Show the supported languages and their families."""
    print("="*60)
    print("SUPPORTED LANGUAGES")
    print("="*60)
    
    from src.utils import LANGUAGE_FAMILIES, SCRIPT_GROUPS
    
    print(f"Total languages supported: {len(LANGUAGE_CODES)}")
    print()
    
    print("Languages by family:")
    for family, langs in LANGUAGE_FAMILIES.items():
        print(f"  {family.replace('_', ' ').title()}:")
        for lang in langs:
            if lang in LANGUAGE_CODES:
                print(f"    {lang} - {LANGUAGE_CODES[lang]}")
    print()
    
    print("Languages by script:")
    for script, langs in SCRIPT_GROUPS.items():
        print(f"  {script.replace('_', ' ').title()}:")
        for lang in langs:
            if lang in LANGUAGE_CODES:
                print(f"    {lang} - {LANGUAGE_CODES[lang]}")
    print()

def main():
    """Run all demos."""
    print("Code-Switching Language Detection System Demo")
    print("This demonstrates the core components without requiring heavy dependencies.")
    print()
    
    try:
        show_language_families()
        print()
        
        demo_preprocessing()
        print()
        
        demo_tokenization()
        print()
        
        demo_dataset_analysis()
        print()
        
        demo_script_based_prediction()
        print()
        
        print("="*60)
        print("DEMO COMPLETED")
        print("="*60)
        print("Next steps:")
        print("1. Install all dependencies: pip install -r requirements.txt")
        print("2. Train the full model: python train.py")
        print("3. Evaluate the model: python evaluate.py")
        print("4. Use for prediction: python predict.py --interactive")
        print()
        print("The full system includes:")
        print("- FastText language identification")
        print("- Character n-gram models") 
        print("- BiLSTM-CRF sequence modeling")
        print("- Confidence calibration")
        print("- Advanced post-processing")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()