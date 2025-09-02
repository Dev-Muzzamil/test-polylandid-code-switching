"""
Evaluation framework for language detection models.
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import logging

from .utils import LANGUAGE_CODES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDetectionEvaluator:
    """Comprehensive evaluation for language detection systems."""
    
    def __init__(self, target_languages: List[str] = None):
        self.target_languages = target_languages or list(LANGUAGE_CODES.keys())
    
    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of prediction dictionaries from detector
            ground_truth: List of ground truth samples
            
        Returns:
            Comprehensive evaluation metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Number of predictions must match ground truth")
        
        # Collect all token-level predictions and labels
        all_predicted_labels = []
        all_true_labels = []
        all_confidences = []
        
        sentence_level_results = []
        boundary_errors = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Align tokens with ground truth
            try:
                aligned_labels = self._align_prediction_with_gt(pred, gt)
                
                pred_labels = [p['language'] for p in pred['predictions']]
                confidences = [p['confidence'] for p in pred['predictions']]
                
                if len(pred_labels) == len(aligned_labels):
                    all_predicted_labels.extend(pred_labels)
                    all_true_labels.extend(aligned_labels)
                    all_confidences.extend(confidences)
                    
                    # Sentence-level evaluation
                    sentence_result = self._evaluate_sentence_level(pred, gt)
                    sentence_level_results.append(sentence_result)
                    
                    # Boundary error analysis
                    boundary_result = self._analyze_boundary_errors(pred_labels, aligned_labels)
                    boundary_errors.append(boundary_result)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample: {e}")
                continue
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            all_predicted_labels, all_true_labels, all_confidences)
        
        # Calculate per-language metrics
        language_metrics = self._calculate_per_language_metrics(
            all_predicted_labels, all_true_labels)
        
        # Sentence-level metrics
        sentence_metrics = self._calculate_sentence_metrics(sentence_level_results)
        
        # Boundary error metrics
        boundary_metrics = self._calculate_boundary_metrics(boundary_errors)
        
        # Confusion matrix
        confusion_matrix = self._calculate_confusion_matrix(
            all_predicted_labels, all_true_labels)
        
        return {
            'overall': overall_metrics,
            'per_language': language_metrics,
            'sentence_level': sentence_metrics,
            'boundary_errors': boundary_metrics,
            'confusion_matrix': confusion_matrix,
            'total_samples': len(predictions),
            'total_tokens': len(all_predicted_labels)
        }
    
    def _align_prediction_with_gt(self, prediction: Dict, ground_truth: Dict) -> List[str]:
        """Align prediction tokens with ground truth labels."""
        pred_tokens = prediction['tokens']
        gt_spans = ground_truth['spans']
        gt_text = ground_truth['text']
        
        # Create character-to-label mapping from ground truth
        char_labels = ['UNK'] * len(gt_text)
        
        for span in gt_spans:
            span_text = span['text']
            span_lang = span['lang']
            
            # Find all occurrences of span text
            start_pos = 0
            while True:
                pos = gt_text.find(span_text, start_pos)
                if pos == -1:
                    break
                
                # Mark characters with language label
                for i in range(len(span_text)):
                    if pos + i < len(char_labels):
                        char_labels[pos + i] = span_lang
                
                start_pos = pos + 1
        
        # Reconstruct token positions and map to labels
        token_labels = []
        current_pos = 0
        
        for token in pred_tokens:
            # Find token in text
            token_start = gt_text.find(token, current_pos)
            if token_start == -1:
                # Fallback: use current position
                token_start = current_pos
            
            token_end = token_start + len(token)
            
            # Get labels for token span
            token_span_labels = char_labels[token_start:token_end]
            
            # Find most common non-UNK label
            label_counts = Counter(label for label in token_span_labels if label != 'UNK')
            
            if label_counts:
                best_label = label_counts.most_common(1)[0][0]
            else:
                # Fallback to first valid label in span or default
                valid_labels = [label for label in token_span_labels if label in LANGUAGE_CODES]
                best_label = valid_labels[0] if valid_labels else 'en'
            
            token_labels.append(best_label)
            current_pos = token_end
        
        return token_labels
    
    def _calculate_overall_metrics(self, predicted: List[str], true: List[str], confidences: List[float]) -> Dict:
        """Calculate overall accuracy and F1 metrics."""
        if len(predicted) != len(true):
            return {}
        
        # Token-level accuracy
        correct = sum(1 for p, t in zip(predicted, true) if p == t)
        total = len(predicted)
        accuracy = correct / total if total > 0 else 0.0
        
        # Macro F1 (average across all languages)
        lang_f1_scores = []
        for lang in self.target_languages:
            tp = sum(1 for p, t in zip(predicted, true) if p == lang and t == lang)
            fp = sum(1 for p, t in zip(predicted, true) if p == lang and t != lang)
            fn = sum(1 for p, t in zip(predicted, true) if p != lang and t == lang)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            lang_f1_scores.append(f1)
        
        macro_f1 = np.mean(lang_f1_scores)
        
        # Weighted F1 (weighted by support)
        lang_weights = []
        weighted_f1_scores = []
        
        for lang in self.target_languages:
            support = sum(1 for t in true if t == lang)
            lang_weights.append(support)
            
            if support > 0:
                tp = sum(1 for p, t in zip(predicted, true) if p == lang and t == lang)
                fp = sum(1 for p, t in zip(predicted, true) if p == lang and t != lang)
                fn = sum(1 for p, t in zip(predicted, true) if p != lang and t == lang)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                weighted_f1_scores.append(f1 * support)
            else:
                weighted_f1_scores.append(0.0)
        
        total_support = sum(lang_weights)
        weighted_f1 = sum(weighted_f1_scores) / total_support if total_support > 0 else 0.0
        
        # Confidence statistics
        avg_confidence = np.mean(confidences) if confidences else 0.0
        confidence_correct = [conf for conf, p, t in zip(confidences, predicted, true) if p == t]
        confidence_incorrect = [conf for conf, p, t in zip(confidences, predicted, true) if p != t]
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'correct_tokens': correct,
            'total_tokens': total,
            'avg_confidence': avg_confidence,
            'avg_confidence_correct': np.mean(confidence_correct) if confidence_correct else 0.0,
            'avg_confidence_incorrect': np.mean(confidence_incorrect) if confidence_incorrect else 0.0
        }
    
    def _calculate_per_language_metrics(self, predicted: List[str], true: List[str]) -> Dict:
        """Calculate precision, recall, F1 for each language."""
        language_metrics = {}
        
        for lang in self.target_languages:
            tp = sum(1 for p, t in zip(predicted, true) if p == lang and t == lang)
            fp = sum(1 for p, t in zip(predicted, true) if p == lang and t != lang)
            fn = sum(1 for p, t in zip(predicted, true) if p != lang and t == lang)
            tn = sum(1 for p, t in zip(predicted, true) if p != lang and t != lang)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            support = tp + fn  # Number of true instances
            
            language_metrics[lang] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
        
        return language_metrics
    
    def _evaluate_sentence_level(self, prediction: Dict, ground_truth: Dict) -> Dict:
        """Evaluate sentence-level detection accuracy."""
        # Extract languages from prediction
        pred_languages = set(pred['language'] for pred in prediction['predictions'])
        
        # Extract languages from ground truth
        gt_languages = set(span['lang'] for span in ground_truth['spans'])
        
        # Check if all languages were detected
        all_detected = gt_languages.issubset(pred_languages)
        exact_match = pred_languages == gt_languages
        
        # Count over-detection (extra languages)
        extra_languages = pred_languages - gt_languages
        missed_languages = gt_languages - pred_languages
        
        return {
            'all_languages_detected': all_detected,
            'exact_language_match': exact_match,
            'num_predicted_languages': len(pred_languages),
            'num_true_languages': len(gt_languages),
            'extra_languages': list(extra_languages),
            'missed_languages': list(missed_languages),
            'is_code_switched': len(gt_languages) > 1
        }
    
    def _analyze_boundary_errors(self, predicted: List[str], true: List[str]) -> Dict:
        """Analyze boundary detection errors."""
        if len(predicted) != len(true):
            return {}
        
        # Find language switches
        pred_boundaries = []
        true_boundaries = []
        
        for i in range(1, len(predicted)):
            if predicted[i] != predicted[i-1]:
                pred_boundaries.append(i)
            if true[i] != true[i-1]:
                true_boundaries.append(i)
        
        # Calculate boundary precision and recall
        correct_boundaries = 0
        for boundary in pred_boundaries:
            # Check if there's a true boundary within ±1 position
            if any(abs(boundary - tb) <= 1 for tb in true_boundaries):
                correct_boundaries += 1
        
        boundary_precision = correct_boundaries / len(pred_boundaries) if pred_boundaries else 1.0
        boundary_recall = correct_boundaries / len(true_boundaries) if true_boundaries else 1.0
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0.0
        
        return {
            'predicted_boundaries': len(pred_boundaries),
            'true_boundaries': len(true_boundaries),
            'correct_boundaries': correct_boundaries,
            'boundary_precision': boundary_precision,
            'boundary_recall': boundary_recall,
            'boundary_f1': boundary_f1
        }
    
    def _calculate_sentence_metrics(self, sentence_results: List[Dict]) -> Dict:
        """Calculate aggregate sentence-level metrics."""
        if not sentence_results:
            return {}
        
        all_detected = sum(1 for r in sentence_results if r['all_languages_detected'])
        exact_match = sum(1 for r in sentence_results if r['exact_language_match'])
        total_sentences = len(sentence_results)
        
        # Code-switching specific metrics
        cs_sentences = [r for r in sentence_results if r['is_code_switched']]
        cs_all_detected = sum(1 for r in cs_sentences if r['all_languages_detected'])
        cs_exact_match = sum(1 for r in cs_sentences if r['exact_language_match'])
        
        return {
            'sentence_detection_rate': all_detected / total_sentences,
            'sentence_exact_match_rate': exact_match / total_sentences,
            'total_sentences': total_sentences,
            'code_switched_sentences': len(cs_sentences),
            'cs_detection_rate': cs_all_detected / len(cs_sentences) if cs_sentences else 0.0,
            'cs_exact_match_rate': cs_exact_match / len(cs_sentences) if cs_sentences else 0.0
        }
    
    def _calculate_boundary_metrics(self, boundary_results: List[Dict]) -> Dict:
        """Calculate aggregate boundary detection metrics."""
        if not boundary_results:
            return {}
        
        total_pred_boundaries = sum(r['predicted_boundaries'] for r in boundary_results)
        total_true_boundaries = sum(r['true_boundaries'] for r in boundary_results)
        total_correct_boundaries = sum(r['correct_boundaries'] for r in boundary_results)
        
        overall_boundary_precision = total_correct_boundaries / total_pred_boundaries if total_pred_boundaries > 0 else 1.0
        overall_boundary_recall = total_correct_boundaries / total_true_boundaries if total_true_boundaries > 0 else 1.0
        overall_boundary_f1 = 2 * overall_boundary_precision * overall_boundary_recall / (overall_boundary_precision + overall_boundary_recall) if (overall_boundary_precision + overall_boundary_recall) > 0 else 0.0
        
        return {
            'boundary_precision': overall_boundary_precision,
            'boundary_recall': overall_boundary_recall,
            'boundary_f1': overall_boundary_f1,
            'total_predicted_boundaries': total_pred_boundaries,
            'total_true_boundaries': total_true_boundaries,
            'total_correct_boundaries': total_correct_boundaries
        }
    
    def _calculate_confusion_matrix(self, predicted: List[str], true: List[str]) -> Dict:
        """Calculate confusion matrix."""
        confusion = defaultdict(lambda: defaultdict(int))
        
        for p, t in zip(predicted, true):
            confusion[t][p] += 1
        
        # Convert to regular dict
        confusion_dict = {}
        for true_lang in self.target_languages:
            confusion_dict[true_lang] = {}
            for pred_lang in self.target_languages:
                confusion_dict[true_lang][pred_lang] = confusion[true_lang][pred_lang]
        
        return confusion_dict
    
    def print_evaluation_report(self, evaluation_results: Dict):
        """Print a comprehensive evaluation report."""
        print("=" * 60)
        print("LANGUAGE DETECTION EVALUATION REPORT")
        print("=" * 60)
        
        # Overall metrics
        overall = evaluation_results['overall']
        print(f"\nOVERALL METRICS:")
        print(f"  Token Accuracy:     {overall['accuracy']:.4f}")
        print(f"  Macro F1:          {overall['macro_f1']:.4f}")
        print(f"  Weighted F1:       {overall['weighted_f1']:.4f}")
        print(f"  Total Tokens:      {overall['total_tokens']}")
        print(f"  Avg Confidence:    {overall['avg_confidence']:.4f}")
        
        # Check requirements
        print(f"\nREQUIREMENT CHECKS:")
        print(f"  Overall F1 > 0.91:     {'✓' if overall['macro_f1'] > 0.91 else '✗'} ({overall['macro_f1']:.4f})")
        
        # Per-language metrics
        per_lang = evaluation_results['per_language']
        print(f"\nPER-LANGUAGE F1 SCORES:")
        lang_f1_values = []
        for lang in sorted(per_lang.keys()):
            metrics = per_lang[lang]
            f1 = metrics['f1_score']
            lang_f1_values.append(f1)
            status = '✓' if f1 > 0.85 else '✗'
            print(f"  {lang.upper()}: {f1:.4f} {status} (support: {metrics['support']})")
        
        min_lang_f1 = min(lang_f1_values) if lang_f1_values else 0.0
        print(f"  Min Language F1 > 0.85: {'✓' if min_lang_f1 > 0.85 else '✗'} ({min_lang_f1:.4f})")
        
        # Sentence-level metrics
        sentence = evaluation_results['sentence_level']
        print(f"\nSENTENCE-LEVEL METRICS:")
        print(f"  Detection Rate:     {sentence['sentence_detection_rate']:.4f}")
        print(f"  Exact Match Rate:   {sentence['sentence_exact_match_rate']:.4f}")
        print(f"  CS Detection Rate:  {sentence['cs_detection_rate']:.4f}")
        print(f"  Total Sentences:    {sentence['total_sentences']}")
        print(f"  CS Sentences:       {sentence['code_switched_sentences']}")
        
        sentence_acc = sentence['sentence_detection_rate']
        print(f"  Sentence Acc > 0.90:   {'✓' if sentence_acc > 0.90 else '✗'} ({sentence_acc:.4f})")
        
        # Boundary metrics
        boundary = evaluation_results['boundary_errors']
        print(f"\nBOUNDARY DETECTION:")
        print(f"  Boundary Precision: {boundary['boundary_precision']:.4f}")
        print(f"  Boundary Recall:    {boundary['boundary_recall']:.4f}")
        print(f"  Boundary F1:        {boundary['boundary_f1']:.4f}")
        
        print("\n" + "=" * 60)
        
        # Summary
        overall_pass = overall['macro_f1'] > 0.91
        lang_pass = min_lang_f1 > 0.85
        sentence_pass = sentence_acc > 0.90
        
        all_pass = overall_pass and lang_pass and sentence_pass
        
        print("REQUIREMENT SUMMARY:")
        print(f"  Overall F1 > 0.91:      {'PASS' if overall_pass else 'FAIL'}")
        print(f"  Language F1 > 0.85:     {'PASS' if lang_pass else 'FAIL'}")
        print(f"  Sentence Acc > 0.90:    {'PASS' if sentence_pass else 'FAIL'}")
        print(f"  ALL REQUIREMENTS:       {'PASS' if all_pass else 'FAIL'}")
        print("=" * 60)
        
        return all_pass