"""
Core language detection models and components.
"""
import os
import re
import fasttext
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import logging

from .utils import (
    LANGUAGE_CODES, calculate_confidence_scores, entropy, 
    merge_predictions, apply_language_constraints
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceLevelDetector:
    """Sentence-level language detection to get candidate languages."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        
        # Fallback to langdetect if fasttext model not available
        try:
            from langdetect import detect_langs
            self.fallback_detect = detect_langs
        except ImportError:
            self.fallback_detect = None
            logger.warning("langdetect not available for fallback")
    
    def load_model(self, model_path: str = None):
        """Load FastText language identification model."""
        if model_path is None:
            model_path = self.model_path
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = fasttext.load_model(model_path)
                logger.info(f"Loaded FastText model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load FastText model: {e}")
                self.model = None
        else:
            logger.warning("No FastText model path provided or file not found")
    
    def detect_languages(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Detect top-k candidate languages for a sentence."""
        candidates = []
        
        if self.model:
            try:
                # Clean text for FastText
                clean_text = re.sub(r'\n', ' ', text.strip())
                if clean_text:
                    predictions = self.model.predict(clean_text, k=top_k)
                    labels, scores = predictions
                    
                    for label, score in zip(labels, scores):
                        # Extract language code from FastText label (usually __label__xx)
                        lang_code = label.replace('__label__', '')
                        
                        # Map to our supported languages
                        if lang_code in LANGUAGE_CODES:
                            candidates.append((lang_code, float(score)))
            except Exception as e:
                logger.warning(f"FastText prediction failed: {e}")
        
        # Fallback to langdetect if available and no candidates found
        if not candidates and self.fallback_detect:
            try:
                detections = self.fallback_detect(text)
                for detection in detections[:top_k]:
                    if detection.lang in LANGUAGE_CODES:
                        candidates.append((detection.lang, detection.prob))
            except Exception as e:
                logger.warning(f"Langdetect fallback failed: {e}")
        
        # If still no candidates, return uniform distribution over common languages
        if not candidates:
            common_langs = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'ar']
            prob = 1.0 / len(common_langs)
            candidates = [(lang, prob) for lang in common_langs]
        
        return candidates


class TokenLevelDetector:
    """Token-level language detection using multiple approaches."""
    
    def __init__(self, fasttext_model_path: Optional[str] = None):
        self.fasttext_model = None
        self.fasttext_model_path = fasttext_model_path
        
        # Character n-gram features for short tokens
        self.char_ngram_models = {}
        
    def load_fasttext_model(self, model_path: str = None):
        """Load FastText model for token-level detection."""
        if model_path is None:
            model_path = self.fasttext_model_path
            
        if model_path and os.path.exists(model_path):
            try:
                self.fasttext_model = fasttext.load_model(model_path)
                logger.info(f"Loaded FastText model for token detection from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load FastText model: {e}")
    
    def train_char_ngram_models(self, training_data: List[Dict]):
        """Train character n-gram models for each language."""
        logger.info("Training character n-gram models...")
        
        # Collect training data by language
        lang_texts = defaultdict(list)
        for sample in training_data:
            for span in sample['spans']:
                lang_texts[span['lang']].append(span['text'])
        
        # Train model for each language
        for lang, texts in lang_texts.items():
            # Create character n-gram features (2-4 grams)
            char_counts = defaultdict(int)
            total_chars = 0
            
            for text in texts:
                text = text.lower()
                # Add padding
                padded_text = f"^{text}$"
                
                # Extract n-grams
                for n in range(2, 5):
                    for i in range(len(padded_text) - n + 1):
                        ngram = padded_text[i:i+n]
                        char_counts[ngram] += 1
                        total_chars += 1
            
            # Convert to probabilities
            char_probs = {ngram: count / total_chars 
                         for ngram, count in char_counts.items()}
            
            self.char_ngram_models[lang] = char_probs
        
        logger.info(f"Trained character n-gram models for {len(self.char_ngram_models)} languages")
    
    def predict_with_fasttext(self, token: str, top_k: int = 5) -> Dict[str, float]:
        """Predict language using FastText."""
        if not self.fasttext_model or len(token.strip()) < 2:
            return {}
        
        try:
            clean_token = re.sub(r'[^\w]', '', token.lower())
            if not clean_token:
                return {}
            
            predictions = self.fasttext_model.predict(clean_token, k=top_k)
            labels, scores = predictions
            
            result = {}
            for label, score in zip(labels, scores):
                lang_code = label.replace('__label__', '')
                if lang_code in LANGUAGE_CODES:
                    result[lang_code] = float(score)
            
            return result
        except Exception as e:
            logger.warning(f"FastText token prediction failed for '{token}': {e}")
            return {}
    
    def predict_with_char_ngrams(self, token: str) -> Dict[str, float]:
        """Predict language using character n-grams."""
        if not self.char_ngram_models or len(token.strip()) < 2:
            return {}
        
        token = token.lower()
        padded_token = f"^{token}$"
        
        lang_scores = {}
        
        for lang, ngram_probs in self.char_ngram_models.items():
            score = 0.0
            ngram_count = 0
            
            # Score based on character n-grams
            for n in range(2, min(5, len(padded_token) + 1)):
                for i in range(len(padded_token) - n + 1):
                    ngram = padded_token[i:i+n]
                    if ngram in ngram_probs:
                        score += np.log(ngram_probs[ngram] + 1e-10)
                        ngram_count += 1
            
            if ngram_count > 0:
                lang_scores[lang] = score / ngram_count
        
        if not lang_scores:
            return {}
        
        # Convert log scores to probabilities
        max_score = max(lang_scores.values())
        exp_scores = {lang: np.exp(score - max_score) 
                     for lang, score in lang_scores.items()}
        total = sum(exp_scores.values())
        
        return {lang: score / total for lang, score in exp_scores.items()}
    
    def predict_token(self, token: str, 
                     candidate_langs: List[str] = None,
                     script_hints: List[str] = None) -> Dict[str, float]:
        """Predict language for a single token using multiple methods."""
        # Get predictions from different methods
        fasttext_pred = self.predict_with_fasttext(token)
        ngram_pred = self.predict_with_char_ngrams(token)
        
        # Merge predictions
        predictions = []
        weights = []
        
        if fasttext_pred:
            predictions.append(fasttext_pred)
            weights.append(0.7)  # Higher weight for FastText
        
        if ngram_pred:
            predictions.append(ngram_pred)
            weights.append(0.3)
        
        if not predictions:
            # Return uniform distribution over candidate languages or all languages
            candidates = candidate_langs or list(LANGUAGE_CODES.keys())
            prob = 1.0 / len(candidates)
            return {lang: prob for lang in candidates}
        
        # Merge predictions
        merged = merge_predictions(predictions, weights)
        
        # Apply constraints
        constrained = apply_language_constraints(
            merged, candidate_langs, script_hints)
        
        return constrained


class SequenceModel(nn.Module):
    """Neural sequence model for smoothing language predictions."""
    
    def __init__(self, num_languages: int, 
                 input_dim: int = 64, 
                 hidden_dim: int = 128):
        super(SequenceModel, self).__init__()
        
        self.num_languages = num_languages
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer for language features
        self.feature_projection = nn.Linear(num_languages + 32, input_dim)
        
        # BiLSTM for sequence modeling
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, 
                           batch_first=True, bidirectional=True)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_languages)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features: torch.Tensor, lengths: torch.Tensor = None):
        """Forward pass through the sequence model."""
        # Project features
        projected = self.feature_projection(features)
        projected = self.dropout(projected)
        
        # LSTM processing
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(projected)
        
        lstm_out = self.dropout(lstm_out)
        
        # Output projection
        output = self.output_projection(lstm_out)
        
        return output


class HybridLanguageDetector:
    """Main hybrid language detection system."""
    
    def __init__(self, 
                 fasttext_model_path: Optional[str] = None,
                 use_sequence_model: bool = True):
        
        self.sentence_detector = SentenceLevelDetector()
        self.token_detector = TokenLevelDetector(fasttext_model_path)
        
        self.use_sequence_model = use_sequence_model
        self.sequence_model = None
        self.language_to_idx = {lang: i for i, lang in enumerate(LANGUAGE_CODES.keys())}
        self.idx_to_language = {i: lang for lang, i in self.language_to_idx.items()}
        
        # Feature extractors
        self.feature_scaler = StandardScaler()
        self.is_trained = False
    
    def load_models(self, model_dir: str):
        """Load all pre-trained models."""
        # Load FastText models
        fasttext_path = os.path.join(model_dir, 'lid.176.bin')
        if os.path.exists(fasttext_path):
            self.sentence_detector.load_model(fasttext_path)
            self.token_detector.load_fasttext_model(fasttext_path)
        
        # Load character n-gram models
        ngram_path = os.path.join(model_dir, 'char_ngram_models.pkl')
        if os.path.exists(ngram_path):
            with open(ngram_path, 'rb') as f:
                self.token_detector.char_ngram_models = pickle.load(f)
        
        # Load sequence model
        seq_model_path = os.path.join(model_dir, 'sequence_model.pt')
        if os.path.exists(seq_model_path) and self.use_sequence_model:
            self.sequence_model = SequenceModel(len(LANGUAGE_CODES))
            self.sequence_model.load_state_dict(torch.load(seq_model_path, map_location='cpu'))
            self.sequence_model.eval()
        
        # Load feature scaler
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
        
        self.is_trained = True
        logger.info("Models loaded successfully")
    
    def save_models(self, model_dir: str):
        """Save all trained models."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save character n-gram models
        ngram_path = os.path.join(model_dir, 'char_ngram_models.pkl')
        with open(ngram_path, 'wb') as f:
            pickle.dump(self.token_detector.char_ngram_models, f)
        
        # Save sequence model
        if self.sequence_model:
            seq_model_path = os.path.join(model_dir, 'sequence_model.pt')
            torch.save(self.sequence_model.state_dict(), seq_model_path)
        
        # Save feature scaler
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        logger.info(f"Models saved to {model_dir}")
    
    def extract_features(self, token: str, 
                        token_predictions: Dict[str, float],
                        script_type: str = None) -> np.ndarray:
        """Extract features for sequence model."""
        features = []
        
        # Language prediction probabilities
        lang_probs = [token_predictions.get(lang, 0.0) 
                     for lang in LANGUAGE_CODES.keys()]
        features.extend(lang_probs)
        
        # Token-level features
        features.extend([
            len(token),  # Token length
            1.0 if token.isalpha() else 0.0,  # Is alphabetic
            1.0 if token.isdigit() else 0.0,  # Is numeric
            1.0 if token.isupper() else 0.0,  # Is uppercase
            1.0 if token.islower() else 0.0,  # Is lowercase
            1.0 if token.istitle() else 0.0,  # Is title case
            entropy(token_predictions),  # Prediction entropy
        ])
        
        # Script-based features (one-hot encoding)
        script_features = [0.0] * 10  # Max 10 script types
        if script_type:
            script_idx = hash(script_type) % 10
            script_features[script_idx] = 1.0
        features.extend(script_features)
        
        # Statistical features
        max_prob = max(token_predictions.values()) if token_predictions else 0.0
        features.extend([
            max_prob,  # Maximum probability
            len([p for p in token_predictions.values() if p > 0.1]),  # Number of confident predictions
        ])
        
        # Pad to fixed size
        while len(features) < 64:
            features.append(0.0)
        
        return np.array(features[:64])
    
    def predict_sequence(self, tokens: List[str], 
                        token_predictions: List[Dict[str, float]],
                        script_types: List[str] = None) -> List[Dict[str, float]]:
        """Predict languages for a sequence of tokens."""
        if not self.sequence_model or not self.is_trained:
            return token_predictions
        
        if script_types is None:
            script_types = [None] * len(tokens)
        
        # Extract features
        features = []
        for token, pred, script in zip(tokens, token_predictions, script_types):
            feat = self.extract_features(token, pred, script)
            features.append(feat)
        
        features = np.array(features)
        
        # Normalize features
        try:
            features = self.feature_scaler.transform(features)
        except:
            # If scaler not fitted, skip normalization
            pass
        
        # Run through sequence model
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            output = self.sequence_model(features_tensor)
            output = torch.softmax(output, dim=-1)
            output = output.squeeze(0).numpy()
        
        # Convert back to language predictions
        smoothed_predictions = []
        for i, logits in enumerate(output):
            pred_dict = {}
            for j, prob in enumerate(logits):
                lang = self.idx_to_language[j]
                pred_dict[lang] = float(prob)
            smoothed_predictions.append(pred_dict)
        
        return smoothed_predictions
    
    def predict(self, text: str, tokens_info: List[Dict] = None) -> List[Dict]:
        """Main prediction method."""
        if tokens_info is None:
            # Use basic whitespace tokenization as fallback
            tokens_info = [{'text': token, 'script': None, 'language_hints': []} 
                          for token in text.split()]
        
        # Step 1: Get sentence-level candidates
        sentence_candidates = self.sentence_detector.detect_languages(text, top_k=5)
        candidate_langs = [lang for lang, _ in sentence_candidates if lang in LANGUAGE_CODES]
        
        # Step 2: Get token-level predictions
        token_predictions = []
        tokens = []
        script_types = []
        
        for token_info in tokens_info:
            token = token_info['text']
            script = token_info.get('script')
            hints = token_info.get('language_hints', [])
            
            # Filter candidate languages with hints
            filtered_candidates = [lang for lang in candidate_langs 
                                 if not hints or lang in hints]
            if not filtered_candidates:
                filtered_candidates = candidate_langs
            
            # Get token prediction
            pred = self.token_detector.predict_token(
                token, 
                filtered_candidates, 
                [str(script)] if script else None
            )
            
            tokens.append(token)
            token_predictions.append(pred)
            script_types.append(str(script) if script else None)
        
        # Step 3: Apply sequence modeling
        if self.use_sequence_model and len(tokens) > 1:
            smoothed_predictions = self.predict_sequence(
                tokens, token_predictions, script_types)
        else:
            smoothed_predictions = token_predictions
        
        # Step 4: Convert to final format
        results = []
        for token, pred in zip(tokens, smoothed_predictions):
            best_lang = max(pred, key=pred.get) if pred else 'en'
            confidence = pred.get(best_lang, 0.0) if pred else 0.0
            
            results.append({
                'text': token,
                'language': best_lang,
                'confidence': confidence,
                'all_predictions': pred
            })
        
        return results