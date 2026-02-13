//! Text tokenization functionality.
//!
//! Provides methods to split text into regex-based, word-level (and
//! punctuation-level) tokens. Tokenization is necessary for alignment
//! between extracted data and the source text and for forming sentence
//! boundaries for LLM information extraction.

use crate::exceptions::{LangExtractError, LangExtractResult};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Enumeration of token types produced during tokenization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenType {
    /// Represents an alphabetical word token
    Word = 0,
    /// Represents a numeric token
    Number = 1,
    /// Represents punctuation characters
    Punctuation = 2,
    /// Represents an acronym or slash-delimited abbreviation
    Acronym = 3,
}

/// Represents a character interval in text
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCharSpan {
    /// Starting character index (inclusive)
    pub start_pos: usize,
    /// Ending character index (exclusive)
    pub end_pos: usize,
}

impl TokenCharSpan {
    /// Create a new character interval
    pub fn new(start_pos: usize, end_pos: usize) -> Self {
        Self { start_pos, end_pos }
    }

    /// Get the length of the interval
    pub fn length(&self) -> usize {
        self.end_pos.saturating_sub(self.start_pos)
    }
}

impl From<TokenCharSpan> for crate::data::CharInterval {
    fn from(span: TokenCharSpan) -> Self {
        crate::data::CharInterval::new(Some(span.start_pos), Some(span.end_pos))
    }
}

/// Represents a token interval over tokens in tokenized text
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenInterval {
    /// The index of the first token in the interval
    pub start_index: usize,
    /// The index one past the last token in the interval
    pub end_index: usize,
}

impl TokenInterval {
    /// Create a new token interval
    pub fn new(start_index: usize, end_index: usize) -> LangExtractResult<Self> {
        if start_index >= end_index {
            return Err(LangExtractError::invalid_input(format!(
                "Start index {} must be < end index {}",
                start_index, end_index
            )));
        }
        Ok(Self {
            start_index,
            end_index,
        })
    }
}

/// Represents a token extracted from text
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    /// The position of the token in the sequence of tokens
    pub index: usize,
    /// The type of the token
    pub token_type: TokenType,
    /// The character interval within the original text that this token spans
    pub char_interval: TokenCharSpan,
    /// True if the token immediately follows a newline or carriage return
    pub first_token_after_newline: bool,
}

impl Token {
    /// Create a new token
    pub fn new(
        index: usize,
        token_type: TokenType,
        char_interval: TokenCharSpan,
        first_token_after_newline: bool,
    ) -> Self {
        Self {
            index,
            token_type,
            char_interval,
            first_token_after_newline,
        }
    }
}

/// Holds the result of tokenizing a text string
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenizedText {
    /// The original text that was tokenized
    pub text: String,
    /// A list of Token objects extracted from the text
    pub tokens: Vec<Token>,
}

impl TokenizedText {
    /// Create a new tokenized text
    pub fn new(text: String) -> Self {
        Self {
            text,
            tokens: Vec::new(),
        }
    }

    /// Get the number of tokens
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if tokenized text is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Text tokenizer for splitting text into tokens
pub struct Tokenizer {
    _letters_pattern: Regex,
    digits_pattern: Regex,
    _symbols_pattern: Regex,
    slash_abbrev_pattern: Regex,
    token_pattern: Regex,
    word_pattern: Regex,
    end_of_sentence_pattern: Regex,
    known_abbreviations: HashSet<String>,
}

impl Tokenizer {
    /// Create a new tokenizer
    pub fn new() -> LangExtractResult<Self> {
        // Regex patterns for tokenization (matching Python implementation)
        let letters_pattern = Regex::new(r"[A-Za-z]+").map_err(|e| {
            LangExtractError::configuration(format!("Failed to compile letters regex: {}", e))
        })?;

        let digits_pattern = Regex::new(r"[0-9]+").map_err(|e| {
            LangExtractError::configuration(format!("Failed to compile digits regex: {}", e))
        })?;

        let symbols_pattern = Regex::new(r"[^A-Za-z0-9\s]+").map_err(|e| {
            LangExtractError::configuration(format!("Failed to compile symbols regex: {}", e))
        })?;

        let slash_abbrev_pattern = Regex::new(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+").map_err(|e| {
            LangExtractError::configuration(format!("Failed to compile slash abbreviation regex: {}", e))
        })?;

        let token_pattern = Regex::new(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+|[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]+").map_err(|e| {
            LangExtractError::configuration(format!("Failed to compile token regex: {}", e))
        })?;

        let word_pattern = Regex::new(r"^(?:[A-Za-z]+|[0-9]+)$").map_err(|e| {
            LangExtractError::configuration(format!("Failed to compile word regex: {}", e))
        })?;

        let end_of_sentence_pattern = Regex::new(r"[.?!]$").map_err(|e| {
            LangExtractError::configuration(format!("Failed to compile end of sentence regex: {}", e))
        })?;

        // Known abbreviations that should not count as sentence enders
        let known_abbreviations = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Ave.", "Blvd.", "Rd.", "Ltd.", "Inc.", "Corp.",
            "vs.", "etc.", "et al.", "i.e.", "e.g.", "cf.", "a.m.", "p.m.", "U.S.", "U.K.", "Ph.D.",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Ok(Self {
            _letters_pattern: letters_pattern,
            digits_pattern,
            _symbols_pattern: symbols_pattern,
            slash_abbrev_pattern,
            token_pattern,
            word_pattern,
            end_of_sentence_pattern,
            known_abbreviations,
        })
    }

    /// Tokenize text into tokens
    pub fn tokenize(&self, text: &str) -> LangExtractResult<TokenizedText> {
        let mut tokenized = TokenizedText::new(text.to_string());
        let mut previous_end = 0;

        for (token_index, token_match) in self.token_pattern.find_iter(text).enumerate() {
            let start_pos = token_match.start();
            let end_pos = token_match.end();
            let matched_text = token_match.as_str();

            // Check if there's a newline in the gap before this token
            let first_token_after_newline = if token_index > 0 {
                let gap = &text[previous_end..start_pos];
                gap.contains('\n') || gap.contains('\r')
            } else {
                false
            };

            // Classify token type
            let token_type = self.classify_token(matched_text);

            let token = Token::new(
                token_index,
                token_type,
                TokenCharSpan::new(start_pos, end_pos),
                first_token_after_newline,
            );

            tokenized.tokens.push(token);
            previous_end = end_pos;
        }

        Ok(tokenized)
    }

    /// Classify a token's type based on its content
    fn classify_token(&self, text: &str) -> TokenType {
        if self.digits_pattern.is_match(text) {
            TokenType::Number
        } else if self.slash_abbrev_pattern.is_match(text) {
            TokenType::Acronym
        } else if self.word_pattern.is_match(text) {
            TokenType::Word
        } else {
            TokenType::Punctuation
        }
    }

    /// Reconstruct text from a token interval
    pub fn tokens_text(
        &self,
        tokenized_text: &TokenizedText,
        token_interval: &TokenInterval,
    ) -> LangExtractResult<String> {
        if token_interval.start_index >= token_interval.end_index {
            return Err(LangExtractError::invalid_input(format!(
                "Invalid token interval: start_index={}, end_index={}",
                token_interval.start_index, token_interval.end_index
            )));
        }

        if token_interval.end_index > tokenized_text.tokens.len() {
            return Err(LangExtractError::invalid_input(format!(
                "Token interval end_index {} exceeds token count {}",
                token_interval.end_index,
                tokenized_text.tokens.len()
            )));
        }

        if tokenized_text.tokens.is_empty() {
            return Ok(String::new());
        }

        let start_token = &tokenized_text.tokens[token_interval.start_index];
        let end_token = &tokenized_text.tokens[token_interval.end_index - 1];

        let start_char = start_token.char_interval.start_pos;
        let end_char = end_token.char_interval.end_pos;

        Ok(tokenized_text.text[start_char..end_char].to_string())
    }

    /// Check if a punctuation token ends a sentence
    pub fn is_end_of_sentence_token(
        &self,
        text: &str,
        tokens: &[Token],
        current_idx: usize,
    ) -> bool {
        if current_idx >= tokens.len() {
            return false;
        }

        let current_token = &tokens[current_idx];
        let current_token_text = &text[current_token.char_interval.start_pos..current_token.char_interval.end_pos];

        if self.end_of_sentence_pattern.is_match(current_token_text) {
            // Check if it's part of a known abbreviation
            if current_idx > 0 {
                let prev_token = &tokens[current_idx - 1];
                let prev_token_text = &text[prev_token.char_interval.start_pos..prev_token.char_interval.end_pos];
                let combined = format!("{}{}", prev_token_text, current_token_text);
                
                if self.known_abbreviations.contains(&combined) {
                    return false;
                }
            }
            return true;
        }
        false
    }

    /// Check if there's a sentence break after a newline
    pub fn is_sentence_break_after_newline(
        &self,
        text: &str,
        tokens: &[Token],
        current_idx: usize,
    ) -> bool {
        if current_idx + 1 >= tokens.len() {
            return false;
        }

        let current_token = &tokens[current_idx];
        let next_token = &tokens[current_idx + 1];

        // Check for newline in the gap between tokens
        let gap_start = current_token.char_interval.end_pos;
        let gap_end = next_token.char_interval.start_pos;
        
        if gap_start >= gap_end {
            return false;
        }

        let gap_text = &text[gap_start..gap_end];
        if !gap_text.contains('\n') {
            return false;
        }

        // Check if next token starts with uppercase
        let next_token_text = &text[next_token.char_interval.start_pos..next_token.char_interval.end_pos];
        !next_token_text.is_empty() && next_token_text.chars().next().unwrap().is_uppercase()
    }

    /// Find sentence range starting from a given token index
    pub fn find_sentence_range(
        &self,
        text: &str,
        tokens: &[Token],
        start_token_index: usize,
    ) -> LangExtractResult<TokenInterval> {
        if start_token_index >= tokens.len() {
            return Err(LangExtractError::invalid_input(format!(
                "start_token_index {} out of range. Total tokens: {}",
                start_token_index,
                tokens.len()
            )));
        }

        let mut i = start_token_index;
        while i < tokens.len() {
            if tokens[i].token_type == TokenType::Punctuation {
                if self.is_end_of_sentence_token(text, tokens, i) {
                    return TokenInterval::new(start_token_index, i + 1);
                }
            }
            if self.is_sentence_break_after_newline(text, tokens, i) {
                return TokenInterval::new(start_token_index, i + 1);
            }
            i += 1;
        }

        TokenInterval::new(start_token_index, tokens.len())
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new().expect("Failed to create default tokenizer")
    }
}

#[cfg(test)]
mod tests;

/// Iterator for processing sentences in tokenized text
pub struct SentenceIterator<'a> {
    tokenized_text: &'a TokenizedText,
    tokenizer: &'a Tokenizer,
    current_token_pos: usize,
    token_len: usize,
}

impl<'a> SentenceIterator<'a> {
    /// Create a new sentence iterator
    pub fn new(
        tokenized_text: &'a TokenizedText,
        tokenizer: &'a Tokenizer,
        current_token_pos: usize,
    ) -> LangExtractResult<Self> {
        let token_len = tokenized_text.tokens.len();
        
        if current_token_pos > token_len {
            return Err(LangExtractError::invalid_input(format!(
                "Current token position {} is past the length of the document {}",
                current_token_pos, token_len
            )));
        }

        Ok(Self {
            tokenized_text,
            tokenizer,
            current_token_pos,
            token_len,
        })
    }
}

impl<'a> Iterator for SentenceIterator<'a> {
    type Item = LangExtractResult<TokenInterval>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_token_pos >= self.token_len {
            return None;
        }

        // Find the sentence range starting from current position
        match self.tokenizer.find_sentence_range(
            &self.tokenized_text.text,
            &self.tokenized_text.tokens,
            self.current_token_pos,
        ) {
            Ok(sentence_range) => {
                // Start the sentence from the current token position.
                // If we are in the middle of a sentence, we should start from there.
                let adjusted_range = match TokenInterval::new(
                    self.current_token_pos,
                    sentence_range.end_index,
                ) {
                    Ok(range) => range,
                    Err(e) => return Some(Err(e)),
                };

                self.current_token_pos = sentence_range.end_index;
                Some(Ok(adjusted_range))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Convenience function for creating a tokenizer and tokenizing text
pub fn tokenize(text: &str) -> LangExtractResult<TokenizedText> {
    let tokenizer = Tokenizer::new()?;
    tokenizer.tokenize(text)
}
