"""
Main language detection pipeline integrating all components.
"""
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from .preprocessing import TextPreprocessor
from .tokenization import ScriptAwareTokenizer, ScriptType
from .models import HybridLanguageDetector
from .utils import consolidate_spans, LANGUAGE_CODES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeSwitchingLanguageDetector:
    """
    Complete pipeline for code-switching language identification.
    
    Supports 20 languages with high accuracy requirements:
    - Overall F1 > 0.91
    - Language-specific F1 > 0.85  
    - Sentence accuracy > 90%
    """
    
    def __init__(self, 
                 fasttext_model_path: Optional[str] = None,
                 use_sequence_model: bool = True,
                 min_confidence_threshold: float = 0.3):
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.tokenizer = ScriptAwareTokenizer()
        self.detector = HybridLanguageDetector(
            fasttext_model_path=fasttext_model_path,
            use_sequence_model=use_sequence_model
        )
        
        self.min_confidence_threshold = min_confidence_threshold
        self.is_trained = False
        
        logger.info("Code-switching language detector initialized")
    
    def load_models(self, model_dir: str):
        """Load pre-trained models."""
        self.detector.load_models(model_dir)
        self.is_trained = True
        logger.info("Models loaded successfully")
    
    def save_models(self, model_dir: str):
        """Save trained models."""
        self.detector.save_models(model_dir)
        logger.info(f"Models saved to {model_dir}")
    
    def train(self, training_data: List[Dict], validation_data: List[Dict] = None):
        """Train the language detection models."""
        logger.info(f"Training on {len(training_data)} samples...")
        
        # Train character n-gram models
        self.detector.token_detector.train_char_ngram_models(training_data)
        
        # Extract features for sequence model training
        if self.detector.use_sequence_model:
            self._train_sequence_model(training_data, validation_data)
        
        self.is_trained = True
        logger.info("Training completed")
    
    def _train_sequence_model(self, training_data: List[Dict], validation_data: List[Dict] = None):
        """Train the sequence model component."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training sequence model...")
        
        # Prepare training data
        features_list = []
        labels_list = []
        
        for sample in training_data:
            text = sample['text']
            spans = sample['spans']
            
            # Preprocess and tokenize
            processed_text, _ = self.preprocessor.preprocess(text)
            tokens_info = self.tokenizer.tokenize(processed_text)
            
            # Align tokens with ground truth labels
            aligned_labels = self._align_tokens_with_labels(tokens_info, spans, text)
            
            if len(aligned_labels) != len(tokens_info):
                continue  # Skip misaligned samples
            
            # Get token-level predictions for features
            token_predictions = []
            for token_info in tokens_info:
                pred = self.detector.token_detector.predict_token(
                    token_info['text'],
                    script_hints=[str(token_info['script'])] if token_info['script'] else None
                )
                token_predictions.append(pred)
            
            # Extract features
            sequence_features = []
            sequence_labels = []
            
            for token_info, pred, label in zip(tokens_info, token_predictions, aligned_labels):
                feat = self.detector.extract_features(
                    token_info['text'], pred, str(token_info['script'])
                )
                sequence_features.append(feat)
                
                label_idx = self.detector.language_to_idx.get(label, 0)
                sequence_labels.append(label_idx)
            
            if len(sequence_features) > 0:
                features_list.append(np.array(sequence_features))
                labels_list.append(np.array(sequence_labels))
        
        if not features_list:
            logger.warning("No training data for sequence model")
            return
        
        # Fit feature scaler
        all_features = np.concatenate(features_list, axis=0)
        self.detector.feature_scaler.fit(all_features)
        
        # Normalize features
        normalized_features = []
        for features in features_list:
            normalized = self.detector.feature_scaler.transform(features)
            normalized_features.append(normalized)
        
        # Create PyTorch dataset
        class SequenceDataset(Dataset):
            def __init__(self, features_list, labels_list):
                self.features_list = features_list
                self.labels_list = labels_list
            
            def __len__(self):
                return len(self.features_list)
            
            def __getitem__(self, idx):
                return torch.FloatTensor(self.features_list[idx]), torch.LongTensor(self.labels_list[idx])
        
        def collate_fn(batch):
            features, labels = zip(*batch)
            lengths = [len(f) for f in features]
            max_len = max(lengths)
            
            # Pad sequences
            padded_features = torch.zeros(len(batch), max_len, features[0].shape[1])
            padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
            
            for i, (feat, label) in enumerate(zip(features, labels)):
                seq_len = len(feat)
                padded_features[i, :seq_len] = feat
                padded_labels[i, :seq_len] = label
            
            return padded_features, padded_labels, torch.LongTensor(lengths)
        
        # Create data loaders
        train_dataset = SequenceDataset(normalized_features, labels_list)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        
        # Initialize model
        self.detector.sequence_model = self.detector.SequenceModel(len(LANGUAGE_CODES))
        optimizer = torch.optim.Adam(self.detector.sequence_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training loop
        num_epochs = 10
        self.detector.sequence_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_features, batch_labels, lengths in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.detector.sequence_model(batch_features, lengths)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                targets = batch_labels.view(-1)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        self.detector.sequence_model.eval()
        logger.info("Sequence model training completed")
    
    def _align_tokens_with_labels(self, tokens_info: List[Dict], spans: List[Dict], original_text: str) -> List[str]:
        """Align tokenized tokens with ground truth span labels."""
        # Create character-to-label mapping
        char_labels = ['UNK'] * len(original_text)
        
        for span in spans:
            span_text = span['text']
            span_lang = span['lang']
            
            # Find span position in original text
            start_pos = original_text.find(span_text)
            if start_pos != -1:
                end_pos = start_pos + len(span_text)
                for i in range(start_pos, end_pos):
                    if i < len(char_labels):
                        char_labels[i] = span_lang
        
        # Map tokens to labels
        token_labels = []
        for token_info in tokens_info:
            start = token_info.get('start', 0)
            end = token_info.get('end', start + len(token_info['text']))
            
            # Get most common label in token span
            token_span_labels = char_labels[start:end]
            if token_span_labels:
                # Find most frequent non-UNK label
                label_counts = {}
                for label in token_span_labels:
                    if label != 'UNK':
                        label_counts[label] = label_counts.get(label, 0) + 1
                
                if label_counts:
                    best_label = max(label_counts, key=label_counts.get)
                    token_labels.append(best_label)
                else:
                    token_labels.append('en')  # Default fallback
            else:
                token_labels.append('en')  # Default fallback
        
        return token_labels
    
    def predict(self, text: str, 
                return_spans: bool = True,
                apply_postprocessing: bool = True) -> Dict:
        """
        Predict languages for input text.
        
        Args:
            text: Input text to analyze
            return_spans: Whether to return consolidated spans
            apply_postprocessing: Whether to apply post-processing steps
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.is_trained:
            logger.warning("Model not trained. Using basic predictions.")
        
        # Step 1: Preprocessing
        processed_text, preprocessing_info = self.preprocessor.preprocess(text)
        
        # Step 2: Tokenization
        tokens_info = self.tokenizer.tokenize(processed_text)
        
        # Step 3: Language detection
        predictions = self.detector.predict(processed_text, tokens_info)
        
        # Step 4: Post-processing
        if apply_postprocessing:
            predictions = self._apply_postprocessing(predictions, tokens_info)
        
        # Step 5: Create result format
        result = {
            'text': text,
            'processed_text': processed_text,
            'tokens': [pred['text'] for pred in predictions],
            'predictions': predictions,
            'preprocessing_info': preprocessing_info
        }
        
        # Step 6: Consolidate spans if requested
        if return_spans:
            tokens = [pred['text'] for pred in predictions]
            languages = [pred['language'] for pred in predictions]
            confidences = [pred['confidence'] for pred in predictions]
            
            spans = consolidate_spans(tokens, languages, confidences, min_span_length=1)
            result['spans'] = spans
        
        # Add summary statistics
        result['summary'] = self._generate_summary(predictions)
        
        return result
    
    def _apply_postprocessing(self, predictions: List[Dict], tokens_info: List[Dict]) -> List[Dict]:
        """Apply post-processing to improve predictions."""
        if len(predictions) <= 1:
            return predictions
        
        processed = predictions.copy()
        
        # 1. Handle very short tokens with low confidence
        for i, pred in enumerate(processed):
            if len(pred['text'].strip()) <= 2 and pred['confidence'] < self.min_confidence_threshold:
                # Look at neighboring tokens
                neighbors = []
                if i > 0:
                    neighbors.append(processed[i-1])
                if i < len(processed) - 1:
                    neighbors.append(processed[i+1])
                
                if neighbors:
                    # Use most confident neighbor's language
                    best_neighbor = max(neighbors, key=lambda x: x['confidence'])
                    if best_neighbor['confidence'] > pred['confidence']:
                        processed[i]['language'] = best_neighbor['language']
                        processed[i]['confidence'] = min(best_neighbor['confidence'] * 0.8, 0.9)
        
        # 2. Smooth very short language switches
        for i in range(1, len(processed) - 1):
            prev_lang = processed[i-1]['language']
            curr_lang = processed[i]['language']
            next_lang = processed[i+1]['language']
            
            # If current token is different from both neighbors and has low confidence
            if (curr_lang != prev_lang and curr_lang != next_lang and 
                prev_lang == next_lang and 
                processed[i]['confidence'] < 0.6):
                
                processed[i]['language'] = prev_lang
                processed[i]['confidence'] = min(processed[i]['confidence'] * 1.2, 0.9)
        
        # 3. Handle high entropy tokens
        for i, pred in enumerate(processed):
            if 'all_predictions' in pred:
                from .utils import entropy
                pred_entropy = entropy(pred['all_predictions'])
                
                if pred_entropy > 2.5:  # High uncertainty
                    # Mark as uncertain or use script hints
                    token_info = tokens_info[i] if i < len(tokens_info) else {}
                    script_hints = token_info.get('language_hints', [])
                    
                    if script_hints and pred['language'] not in script_hints:
                        # Use best script hint
                        best_hint = script_hints[0]
                        if best_hint in pred['all_predictions']:
                            processed[i]['language'] = best_hint
                            processed[i]['confidence'] = pred['all_predictions'][best_hint]
        
        return processed
    
    def _generate_summary(self, predictions: List[Dict]) -> Dict:
        """Generate summary statistics for predictions."""
        if not predictions:
            return {}
        
        languages = [pred['language'] for pred in predictions]
        confidences = [pred['confidence'] for pred in predictions]
        
        # Language distribution
        from collections import Counter
        lang_counts = Counter(languages)
        total_tokens = len(predictions)
        
        # Calculate statistics
        summary = {
            'total_tokens': total_tokens,
            'num_languages': len(lang_counts),
            'languages_detected': list(lang_counts.keys()),
            'language_distribution': {lang: count/total_tokens 
                                   for lang, count in lang_counts.items()},
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'is_code_switched': len(lang_counts) > 1,
            'dominant_language': lang_counts.most_common(1)[0][0] if lang_counts else None
        }
        
        # Boundary analysis
        boundaries = 0
        for i in range(1, len(languages)):
            if languages[i] != languages[i-1]:
                boundaries += 1
        
        summary['language_boundaries'] = boundaries
        summary['avg_span_length'] = total_tokens / (boundaries + 1) if boundaries > 0 else total_tokens
        
        return summary
    
    def evaluate_sample(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """Evaluate a single sample prediction against ground truth."""
        pred_tokens = predicted['tokens']
        pred_langs = [pred['language'] for pred in predicted['predictions']]
        
        # Extract ground truth
        gt_spans = ground_truth['spans']
        gt_text = ground_truth['text']
        
        # Align ground truth with predictions
        gt_labels = self._align_tokens_with_ground_truth(pred_tokens, gt_spans, gt_text)
        
        if len(pred_langs) != len(gt_labels):
            return {'error': 'Token alignment failed'}
        
        # Calculate metrics
        correct = sum(1 for p, g in zip(pred_langs, gt_labels) if p == g)
        total = len(pred_langs)
        
        # Per-language metrics
        from collections import defaultdict
        lang_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for pred_lang, gt_lang in zip(pred_langs, gt_labels):
            if pred_lang == gt_lang:
                lang_stats[gt_lang]['tp'] += 1
            else:
                lang_stats[pred_lang]['fp'] += 1
                lang_stats[gt_lang]['fn'] += 1
        
        # Calculate F1 scores
        lang_f1_scores = {}
        for lang, stats in lang_stats.items():
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            lang_f1_scores[lang] = f1
        
        # Sentence-level accuracy (did we detect all languages?)
        gt_languages = set(span['lang'] for span in gt_spans)
        pred_languages = set(pred_langs)
        sentence_correct = gt_languages == pred_languages
        
        return {
            'token_accuracy': correct / total,
            'correct_tokens': correct,
            'total_tokens': total,
            'language_f1_scores': lang_f1_scores,
            'macro_f1': np.mean(list(lang_f1_scores.values())) if lang_f1_scores else 0.0,
            'sentence_level_correct': sentence_correct,
            'predicted_languages': list(pred_languages),
            'ground_truth_languages': list(gt_languages)
        }
    
    def _align_tokens_with_ground_truth(self, tokens: List[str], gt_spans: List[Dict], gt_text: str) -> List[str]:
        """Align prediction tokens with ground truth spans."""
        # Simple approach: reconstruct text and find character positions
        reconstructed_text = ' '.join(tokens)
        
        # Create character-to-language mapping for ground truth
        char_to_lang = {}
        char_offset = 0
        
        for span in gt_spans:
            span_text = span['text']
            span_lang = span['lang']
            
            # Find span in ground truth text
            span_start = gt_text.find(span_text, char_offset)
            if span_start != -1:
                for i in range(len(span_text)):
                    char_to_lang[span_start + i] = span_lang
                char_offset = span_start + len(span_text)
        
        # Map tokens to languages
        token_labels = []
        char_pos = 0
        
        for token in tokens:
            # Find most common language in token span
            token_langs = []
            for i in range(len(token)):
                if char_pos + i in char_to_lang:
                    token_langs.append(char_to_lang[char_pos + i])
            
            if token_langs:
                # Use most frequent language
                from collections import Counter
                lang_counts = Counter(token_langs)
                best_lang = lang_counts.most_common(1)[0][0]
                token_labels.append(best_lang)
            else:
                token_labels.append('en')  # Default
            
            char_pos += len(token) + 1  # +1 for space
        
        return token_labels