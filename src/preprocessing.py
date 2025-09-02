"""
Preprocessing utilities for text normalization and noise handling.
"""
import re
import unicodedata
from typing import List, Tuple, Dict
import regex


class TextPreprocessor:
    """Handles text preprocessing including normalization, noise handling, and glue splitting."""
    
    def __init__(self):
        # Patterns for different types of tokens
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.emoji_pattern = regex.compile(r'\p{Emoji}')
        
        # Pattern for detecting potentially glued words (long Latin sequences)
        self.glue_pattern = re.compile(r'[a-zA-Z]{12,}')
        
        # Common prefixes and suffixes for glue splitting
        self.common_prefixes = {
            'en': ['anti', 'auto', 'co', 'de', 'dis', 'em', 'fore', 'in', 'im', 'inter', 'mid', 'mis', 'non', 'over', 'pre', 'semi', 'sub', 'super', 'trans', 'un', 'under'],
            'es': ['anti', 'auto', 'co', 'contra', 'des', 'dis', 'ex', 'extra', 'inter', 'pre', 'pro', 're', 'sub', 'super', 'trans'],
            'fr': ['anti', 'auto', 'co', 'contre', 'dé', 'des', 'dis', 'ex', 'extra', 'inter', 'pré', 'pro', 're', 'sous', 'super', 'trans'],
            'de': ['ab', 'an', 'auf', 'aus', 'bei', 'durch', 'ein', 'ent', 'er', 'ge', 'hinter', 'in', 'mit', 'nach', 'über', 'um', 'unter', 'ver', 'vor', 'zu'],
            'pt': ['anti', 'auto', 'co', 'contra', 'des', 'dis', 'ex', 'extra', 'inter', 'pré', 'pro', 're', 'sub', 'super', 'trans'],
            'it': ['anti', 'auto', 'co', 'contro', 'dis', 'ex', 'extra', 'inter', 'pre', 'pro', 're', 'sotto', 'super', 'trans'],
            'nl': ['aan', 'af', 'bij', 'door', 'in', 'mee', 'na', 'om', 'onder', 'op', 'over', 'te', 'tegen', 'uit', 'van', 'voor'],
        }
        
    def normalize_text(self, text: str) -> str:
        """Apply Unicode normalization and basic cleaning."""
        # Unicode NFKC normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters except common whitespace
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\t\n\r ')
        
        return text
    
    def handle_special_tokens(self, text: str) -> Tuple[str, Dict[str, List[Tuple[int, int, str]]]]:
        """Extract and mark special tokens (URLs, mentions, hashtags, emojis)."""
        special_tokens = {
            'urls': [],
            'mentions': [],
            'hashtags': [],
            'emojis': []
        }
        
        # Find all special tokens and their positions
        for match in self.url_pattern.finditer(text):
            special_tokens['urls'].append((match.start(), match.end(), match.group()))
        
        for match in self.mention_pattern.finditer(text):
            special_tokens['mentions'].append((match.start(), match.end(), match.group()))
        
        for match in self.hashtag_pattern.finditer(text):
            special_tokens['hashtags'].append((match.start(), match.end(), match.group()))
        
        for match in self.emoji_pattern.finditer(text):
            special_tokens['emojis'].append((match.start(), match.end(), match.group()))
        
        return text, special_tokens
    
    def split_glued_words(self, token: str, candidate_langs: List[str] = None) -> List[str]:
        """Attempt to split potentially glued words using linguistic heuristics."""
        if len(token) < 12 or not self.glue_pattern.match(token):
            return [token]
        
        # Try to split based on common prefixes and suffixes
        candidate_langs = candidate_langs or ['en', 'es', 'fr', 'de', 'pt', 'it', 'nl']
        
        for lang in candidate_langs:
            if lang in self.common_prefixes:
                for prefix in sorted(self.common_prefixes[lang], key=len, reverse=True):
                    if token.lower().startswith(prefix) and len(token) > len(prefix) + 3:
                        remainder = token[len(prefix):]
                        # Recursively try to split the remainder
                        remainder_splits = self.split_glued_words(remainder, candidate_langs)
                        if len(remainder_splits) > 1 or len(remainder_splits[0]) < len(remainder):
                            return [token[:len(prefix)]] + remainder_splits
        
        # Try to split at capital letters (camelCase detection)
        camel_splits = re.findall(r'[A-Z][a-z]*|[a-z]+', token)
        if len(camel_splits) > 1 and all(len(split) >= 2 for split in camel_splits):
            return camel_splits
        
        # If no good split found, return original token
        return [token]
    
    def preprocess(self, text: str, candidate_langs: List[str] = None) -> Tuple[str, Dict]:
        """Main preprocessing pipeline."""
        # Step 1: Normalize Unicode
        text = self.normalize_text(text)
        
        # Step 2: Handle special tokens
        text, special_tokens = self.handle_special_tokens(text)
        
        # Step 3: Basic tokenization for glue detection
        tokens = text.split()
        processed_tokens = []
        
        for token in tokens:
            # Remove punctuation for glue detection
            clean_token = re.sub(r'[^\w]', '', token)
            if clean_token:
                splits = self.split_glued_words(clean_token, candidate_langs)
                if len(splits) > 1:
                    # Replace original token with splits, preserving surrounding punctuation
                    prefix_punct = token[:token.find(clean_token)]
                    suffix_punct = token[token.find(clean_token) + len(clean_token):]
                    
                    processed_splits = []
                    for i, split in enumerate(splits):
                        if i == 0:
                            processed_splits.append(prefix_punct + split)
                        elif i == len(splits) - 1:
                            processed_splits.append(split + suffix_punct)
                        else:
                            processed_splits.append(split)
                    
                    processed_tokens.extend(processed_splits)
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        processed_text = ' '.join(processed_tokens)
        
        return processed_text, {
            'special_tokens': special_tokens,
            'original_text': text
        }