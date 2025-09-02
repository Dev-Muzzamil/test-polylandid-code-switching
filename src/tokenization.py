"""
Script-aware tokenization utilities.
"""
import re
import unicodedata
from typing import List, Tuple, Dict
from enum import Enum


class ScriptType(Enum):
    """Enumeration of different script types."""
    LATIN = "Latin"
    CYRILLIC = "Cyrillic"
    ARABIC = "Arabic"
    DEVANAGARI = "Devanagari"
    HAN = "Han"
    HIRAGANA = "Hiragana"
    KATAKANA = "Katakana"
    HANGUL = "Hangul"
    THAI = "Thai"
    BENGALI = "Bengali"
    UNKNOWN = "Unknown"


class ScriptAwareTokenizer:
    """Handles script-aware tokenization with script detection."""
    
    def __init__(self):
        # Script detection patterns
        self.script_patterns = {
            ScriptType.LATIN: re.compile(r'[a-zA-ZÀ-ÿĀ-žŠ-ŸĀ-ſ]+'),
            ScriptType.CYRILLIC: re.compile(r'[а-яё]+', re.IGNORECASE),
            ScriptType.ARABIC: re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+'),
            ScriptType.DEVANAGARI: re.compile(r'[\u0900-\u097F]+'),
            ScriptType.HAN: re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]+'),
            ScriptType.HIRAGANA: re.compile(r'[\u3040-\u309F]+'),
            ScriptType.KATAKANA: re.compile(r'[\u30A0-\u30FF]+'),
            ScriptType.HANGUL: re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]+'),
            ScriptType.THAI: re.compile(r'[\u0E00-\u0E7F]+'),
            ScriptType.BENGALI: re.compile(r'[\u0980-\u09FF]+'),
        }
        
        # Language hints based on script
        self.script_language_hints = {
            ScriptType.LATIN: ['en', 'es', 'fr', 'de', 'pt', 'it', 'nl', 'pl', 'tr', 'id', 'vi'],
            ScriptType.CYRILLIC: ['ru'],
            ScriptType.ARABIC: ['ar', 'ur'],
            ScriptType.DEVANAGARI: ['hi'],
            ScriptType.HAN: ['zh', 'ja'],
            ScriptType.HIRAGANA: ['ja'],
            ScriptType.KATAKANA: ['ja'],
            ScriptType.HANGUL: ['ko'],
            ScriptType.THAI: ['th'],
            ScriptType.BENGALI: ['bn'],
        }
        
        # Punctuation and separators
        self.punctuation = re.compile(r'[^\w\s]', re.UNICODE)
        
        # Word boundaries for different scripts
        self.word_boundaries = {
            ScriptType.LATIN: re.compile(r'\s+'),
            ScriptType.CYRILLIC: re.compile(r'\s+'),
            ScriptType.ARABIC: re.compile(r'\s+'),
            ScriptType.DEVANAGARI: re.compile(r'\s+'),
            ScriptType.HAN: re.compile(r''),  # Character-level for Chinese
            ScriptType.HIRAGANA: re.compile(r''),  # Character-level
            ScriptType.KATAKANA: re.compile(r''),  # Character-level
            ScriptType.HANGUL: re.compile(r'\s+'),
            ScriptType.THAI: re.compile(r''),  # Requires dictionary-based segmentation
            ScriptType.BENGALI: re.compile(r'\s+'),
        }
    
    def detect_script(self, text: str) -> ScriptType:
        """Detect the primary script of a text segment."""
        script_counts = {}
        
        for script, pattern in self.script_patterns.items():
            matches = pattern.findall(text)
            if matches:
                script_counts[script] = sum(len(match) for match in matches)
        
        if not script_counts:
            return ScriptType.UNKNOWN
        
        # Return script with highest character count
        return max(script_counts, key=script_counts.get)
    
    def get_language_hints(self, script: ScriptType) -> List[str]:
        """Get possible languages for a given script."""
        return self.script_language_hints.get(script, [])
    
    def tokenize_by_script(self, text: str, script: ScriptType) -> List[str]:
        """Tokenize text based on script type."""
        if script == ScriptType.HAN:
            # Character-level tokenization for Chinese
            return [char for char in text if char.strip() and not self.punctuation.match(char)]
        
        elif script in [ScriptType.HIRAGANA, ScriptType.KATAKANA]:
            # Character-level tokenization for Japanese kana
            return [char for char in text if char.strip() and not self.punctuation.match(char)]
        
        elif script == ScriptType.THAI:
            # For Thai, we'll use simple character-level for now
            # In a production system, you'd use PyThaiNLP
            return [char for char in text if char.strip() and not self.punctuation.match(char)]
        
        else:
            # Whitespace tokenization for other scripts
            boundary = self.word_boundaries.get(script, re.compile(r'\s+'))
            tokens = boundary.split(text.strip())
            return [token for token in tokens if token.strip()]
    
    def tokenize_mixed_script(self, text: str) -> List[Tuple[str, ScriptType, List[str]]]:
        """Tokenize text with mixed scripts, preserving script boundaries."""
        tokens_with_scripts = []
        
        # Split text into script-homogeneous segments
        current_segment = ""
        current_script = None
        
        for char in text:
            if char.isspace():
                if current_segment:
                    # Process accumulated segment
                    script = self.detect_script(current_segment)
                    language_hints = self.get_language_hints(script)
                    tokens = self.tokenize_by_script(current_segment, script)
                    
                    tokens_with_scripts.append((current_segment, script, language_hints))
                    current_segment = ""
                    current_script = None
                continue
            
            char_script = self.detect_script(char)
            
            if current_script is None:
                current_script = char_script
                current_segment = char
            elif current_script == char_script:
                current_segment += char
            else:
                # Script change detected
                if current_segment:
                    script = self.detect_script(current_segment)
                    language_hints = self.get_language_hints(script)
                    tokens_with_scripts.append((current_segment, script, language_hints))
                
                current_segment = char
                current_script = char_script
        
        # Process final segment
        if current_segment:
            script = self.detect_script(current_segment)
            language_hints = self.get_language_hints(script)
            tokens_with_scripts.append((current_segment, script, language_hints))
        
        return tokens_with_scripts
    
    def tokenize(self, text: str) -> List[Dict]:
        """Main tokenization method returning detailed token information."""
        segments = self.tokenize_mixed_script(text)
        
        tokens = []
        char_offset = 0
        
        for segment_text, script, language_hints in segments:
            # Find actual position in original text
            start_pos = text.find(segment_text, char_offset)
            if start_pos == -1:
                start_pos = char_offset
            
            # Tokenize the segment
            segment_tokens = self.tokenize_by_script(segment_text, script)
            
            token_offset = start_pos
            for token_text in segment_tokens:
                # Find token position within segment
                token_start = text.find(token_text, token_offset)
                if token_start == -1:
                    token_start = token_offset
                
                token_info = {
                    'text': token_text,
                    'start': token_start,
                    'end': token_start + len(token_text),
                    'script': script,
                    'language_hints': language_hints
                }
                
                tokens.append(token_info)
                token_offset = token_start + len(token_text)
            
            char_offset = start_pos + len(segment_text)
        
        return tokens