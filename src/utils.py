"""
Utility functions for the language detection pipeline.
"""
import json
import os
from typing import List, Dict, Tuple, Any
import numpy as np
from collections import Counter, defaultdict


# Language codes and their full names
LANGUAGE_CODES = {
    'ar': 'Arabic',
    'bn': 'Bengali', 
    'de': 'German',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'hi': 'Hindi',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'nl': 'Dutch',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'th': 'Thai',
    'tr': 'Turkish',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese'
}

# Reverse mapping
LANGUAGE_NAMES = {v: k for k, v in LANGUAGE_CODES.items()}

# Language families for similarity metrics
LANGUAGE_FAMILIES = {
    'indo_european': ['en', 'de', 'es', 'fr', 'it', 'nl', 'pl', 'pt', 'ru', 'hi', 'ur', 'bn'],
    'sino_tibetan': ['zh'],
    'japonic': ['ja'],
    'koreanic': ['ko'],
    'afroasiatic': ['ar'],
    'austronesian': ['id'],
    'tai_kadai': ['th'],
    'turkic': ['tr'],
    'austroasiatic': ['vi']
}

# Script-based language groupings
SCRIPT_GROUPS = {
    'latin': ['en', 'de', 'es', 'fr', 'it', 'nl', 'pl', 'pt', 'tr', 'id', 'vi'],
    'cyrillic': ['ru'],
    'arabic': ['ar', 'ur'],
    'devanagari': ['hi'],
    'bengali': ['bn'],
    'han': ['zh', 'ja'],
    'hangul': ['ko'],
    'thai': ['th'],
    'japanese_mixed': ['ja']  # Uses multiple scripts
}


def load_dataset(filepath: str) -> List[Dict]:
    """Load the multilingual dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(results: Dict, filepath: str):
    """Save results to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def calculate_confidence_scores(predictions: Dict[str, float]) -> Dict[str, float]:
    """Calculate normalized confidence scores from raw predictions."""
    if not predictions:
        return {}
    
    # Softmax normalization
    max_score = max(predictions.values())
    exp_scores = {lang: np.exp(score - max_score) for lang, score in predictions.items()}
    sum_exp = sum(exp_scores.values())
    
    return {lang: score / sum_exp for lang, score in exp_scores.items()}


def entropy(probabilities: Dict[str, float]) -> float:
    """Calculate entropy of probability distribution."""
    probs = list(probabilities.values())
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    return -sum(p * np.log2(p) for p in probs)


def merge_predictions(predictions_list: List[Dict[str, float]], weights: List[float] = None) -> Dict[str, float]:
    """Merge multiple prediction dictionaries with optional weights."""
    if not predictions_list:
        return {}
    
    if weights is None:
        weights = [1.0] * len(predictions_list)
    
    if len(weights) != len(predictions_list):
        raise ValueError("Number of weights must match number of predictions")
    
    # Get all languages
    all_langs = set()
    for pred in predictions_list:
        all_langs.update(pred.keys())
    
    # Weighted average
    merged = {}
    total_weight = sum(weights)
    
    for lang in all_langs:
        weighted_sum = sum(pred.get(lang, 0.0) * weight 
                          for pred, weight in zip(predictions_list, weights))
        merged[lang] = weighted_sum / total_weight
    
    return merged


def get_language_similarity(lang1: str, lang2: str) -> float:
    """Get similarity score between two languages based on family and script."""
    if lang1 == lang2:
        return 1.0
    
    similarity = 0.0
    
    # Same family bonus
    for family, langs in LANGUAGE_FAMILIES.items():
        if lang1 in langs and lang2 in langs:
            similarity += 0.5
            break
    
    # Same script bonus
    for script, langs in SCRIPT_GROUPS.items():
        if lang1 in langs and lang2 in langs:
            similarity += 0.3
            break
    
    return min(similarity, 1.0)


def apply_language_constraints(predictions: Dict[str, float], 
                             candidate_langs: List[str] = None,
                             script_hints: List[str] = None) -> Dict[str, float]:
    """Apply constraints based on candidate languages and script hints."""
    if not predictions:
        return predictions
    
    constrained = predictions.copy()
    
    # Apply candidate language filter
    if candidate_langs:
        candidate_set = set(candidate_langs)
        for lang in list(constrained.keys()):
            if lang not in candidate_set:
                constrained[lang] *= 0.1  # Heavy penalty, don't completely remove
    
    # Apply script constraints
    if script_hints:
        script_langs = set()
        for script in script_hints:
            script_langs.update(SCRIPT_GROUPS.get(script, []))
        
        if script_langs:
            for lang in list(constrained.keys()):
                if lang not in script_langs:
                    constrained[lang] *= 0.2  # Penalty for script mismatch
    
    # Renormalize
    total = sum(constrained.values())
    if total > 0:
        constrained = {lang: score / total for lang, score in constrained.items()}
    
    return constrained


def smooth_sequence_predictions(token_predictions: List[Dict[str, float]], 
                              smoothing_factor: float = 0.1) -> List[Dict[str, float]]:
    """Apply simple smoothing to sequence of token predictions."""
    if len(token_predictions) <= 1:
        return token_predictions
    
    smoothed = []
    
    for i, pred in enumerate(token_predictions):
        smoothed_pred = pred.copy()
        
        # Get neighboring predictions
        neighbors = []
        if i > 0:
            neighbors.append(token_predictions[i-1])
        if i < len(token_predictions) - 1:
            neighbors.append(token_predictions[i+1])
        
        # Apply smoothing
        if neighbors:
            neighbor_avg = {}
            for neighbor in neighbors:
                for lang, score in neighbor.items():
                    neighbor_avg[lang] = neighbor_avg.get(lang, 0) + score / len(neighbors)
            
            # Blend with neighbors
            for lang in set(list(pred.keys()) + list(neighbor_avg.keys())):
                current_score = pred.get(lang, 0.0)
                neighbor_score = neighbor_avg.get(lang, 0.0)
                smoothed_pred[lang] = (1 - smoothing_factor) * current_score + smoothing_factor * neighbor_score
        
        smoothed.append(smoothed_pred)
    
    return smoothed


def consolidate_spans(tokens: List[str], 
                     predictions: List[str], 
                     confidences: List[float] = None,
                     min_span_length: int = 2) -> List[Dict]:
    """Consolidate adjacent tokens with same language into spans."""
    if len(tokens) != len(predictions):
        raise ValueError("Tokens and predictions must have same length")
    
    if confidences is None:
        confidences = [1.0] * len(tokens)
    
    spans = []
    current_span = {
        'tokens': [],
        'language': None,
        'start_idx': 0,
        'end_idx': 0,
        'confidence': 0.0
    }
    
    for i, (token, pred, conf) in enumerate(zip(tokens, predictions, confidences)):
        if current_span['language'] is None:
            # Start new span
            current_span = {
                'tokens': [token],
                'language': pred,
                'start_idx': i,
                'end_idx': i,
                'confidence': conf
            }
        elif current_span['language'] == pred:
            # Extend current span
            current_span['tokens'].append(token)
            current_span['end_idx'] = i
            current_span['confidence'] = (current_span['confidence'] + conf) / 2
        else:
            # Language change - close current span and start new one
            if len(current_span['tokens']) >= min_span_length or current_span['confidence'] > 0.8:
                current_span['text'] = ' '.join(current_span['tokens'])
                spans.append(current_span.copy())
            
            current_span = {
                'tokens': [token],
                'language': pred,
                'start_idx': i,
                'end_idx': i,
                'confidence': conf
            }
    
    # Add final span
    if current_span['tokens']:
        if len(current_span['tokens']) >= min_span_length or current_span['confidence'] > 0.8:
            current_span['text'] = ' '.join(current_span['tokens'])
            spans.append(current_span)
    
    return spans


def get_dataset_statistics(data: List[Dict]) -> Dict:
    """Calculate statistics about the dataset."""
    stats = {
        'total_sentences': len(data),
        'total_tokens': 0,
        'language_counts': Counter(),
        'sentence_language_counts': Counter(),
        'avg_tokens_per_sentence': 0.0,
        'avg_languages_per_sentence': 0.0,
        'language_pairs': Counter(),
        'token_length_distribution': Counter()
    }
    
    for sample in data:
        spans = sample['spans']
        sentence_langs = set()
        
        stats['total_tokens'] += len(spans)
        
        for span in spans:
            lang = span['lang']
            text = span['text']
            
            stats['language_counts'][lang] += 1
            sentence_langs.add(lang)
            stats['token_length_distribution'][len(text)] += 1
        
        # Track languages in this sentence
        sentence_langs = sorted(sentence_langs)
        stats['sentence_language_counts'][len(sentence_langs)] += 1
        
        # Track language pairs in code-switched sentences
        if len(sentence_langs) > 1:
            for i in range(len(sentence_langs)):
                for j in range(i+1, len(sentence_langs)):
                    pair = f"{sentence_langs[i]}-{sentence_langs[j]}"
                    stats['language_pairs'][pair] += 1
    
    stats['avg_tokens_per_sentence'] = stats['total_tokens'] / stats['total_sentences']
    stats['avg_languages_per_sentence'] = sum(count * num_langs for num_langs, count in stats['sentence_language_counts'].items()) / stats['total_sentences']
    
    return stats


def analyze_errors(true_labels: List[str], 
                  predicted_labels: List[str], 
                  tokens: List[str] = None) -> Dict:
    """Analyze prediction errors in detail."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("True and predicted labels must have same length")
    
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    error_examples = defaultdict(list)
    
    for i, (true_lang, pred_lang) in enumerate(zip(true_labels, predicted_labels)):
        confusion_matrix[true_lang][pred_lang] += 1
        
        if true_lang != pred_lang:
            error_info = {
                'true_lang': true_lang,
                'pred_lang': pred_lang,
                'token_idx': i
            }
            if tokens:
                error_info['token'] = tokens[i]
            
            error_examples[f"{true_lang}->{pred_lang}"].append(error_info)
    
    return {
        'confusion_matrix': dict(confusion_matrix),
        'error_examples': dict(error_examples),
        'total_errors': sum(1 for t, p in zip(true_labels, predicted_labels) if t != p),
        'total_tokens': len(true_labels)
    }