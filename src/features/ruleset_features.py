"""Empirical ruleset feature extraction"""

import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

def extract_ruleset_features(df):
    """
    Extract ruleset features exactly as per paper Table 1

    Args:
        df (pd.DataFrame): DataFrame with job posting data

    Returns:
        pd.DataFrame: Ruleset features
    """
    features = pd.DataFrame(index=df.index)

    # LINGUISTIC FEATURES
    spam_words = ['online', 'extra', 'cash', 'earn', 'money', 'quick', 'easy',
                  'fast', 'guaranteed', 'immediate', 'urgent', 'home']

    def check_spam_words(text):
        text_lower = str(text).lower()
        return int(any(word in text_lower for word in spam_words))

    features['contains_spamwords'] = df['combined_text'].apply(check_spam_words)

    def count_consecutive_punct(text):
        matches = re.findall(r'[!?.]{2,}', str(text))
        return len(matches)

    features['consecutive_punct'] = df['combined_text'].apply(count_consecutive_punct)

    def has_money_in_title(title):
        money_patterns = [r'\$', r'£', r'€', r'\bdollars?\b', r'\bmoney\b', r'\bcash\b']
        title_str = str(title).lower()
        return int(any(re.search(pattern, title_str) for pattern in money_patterns))

    features['money_in_title'] = df['title'].apply(has_money_in_title)

    def has_money_in_description(text):
        money_patterns = [r'\$', r'£', r'€', r'\bdollars?\b', r'\bmoney\b', r'\bcash\b']
        text_str = str(text).lower()
        return int(any(re.search(pattern, text_str) for pattern in money_patterns))

    features['money_in_description'] = df['combined_text'].apply(has_money_in_description)

    def has_url_email(text):
        patterns = [r'http\S+', r'www\.\S+', r'\S+@\S+', r'click here', r'apply at']
        text_str = str(text).lower()
        return int(any(re.search(pattern, text_str) for pattern in patterns))

    features['url_in_text'] = df['combined_text'].apply(has_url_email)

    # CONTEXTUAL FEATURES
    def has_external_application(text):
        patterns = [r'apply at', r'send resume to', r'external link', r'external website']
        text_str = str(text).lower()
        return int(any(re.search(pattern, text_str) for pattern in patterns))

    features['external_application'] = df['combined_text'].apply(has_external_application)

    def addresses_lower_education(text):
        patterns = [r'high school', r'no degree', r'no experience', r'no qualifications']
        text_str = str(text).lower()
        return int(any(re.search(pattern, text_str) for pattern in patterns))

    features['addresses_lower_education'] = df['combined_text'].apply(addresses_lower_education)

    def has_incomplete_attributes(row):
        fields_to_check = ['industry', 'function', 'required_education', 'employment_type']
        missing_count = 0
        for field in fields_to_check:
            if field in row:
                val = row[field]
                if pd.isna(val) or str(val).strip() == '' or str(val).strip().lower() == 'nan':
                    missing_count += 1
        return int(missing_count > 0)

    features['has_incomplete_extra_attributes'] = df.apply(has_incomplete_attributes, axis=1)

    # Company profile features
    def check_company_profile(profile):
        if not isinstance(profile, str):
            return 1, 0, 0

        profile_len = len(profile.strip())
        if profile_len == 0:
            return 1, 0, 0
        elif profile_len < 10:
            return 0, 1, 0
        elif 10 <= profile_len < 100:
            return 0, 0, 1
        else:
            return 0, 0, 0

    profile_results = df['company_profile'].apply(check_company_profile)
    features['has_no_company_profile'] = [r[0] for r in profile_results]
    features['has_short_company_profile'] = [r[1] for r in profile_results]
    features['has_no_long_company_profile'] = [r[2] for r in profile_results]

    features['has_short_description'] = df['description'].apply(
        lambda x: 1 if isinstance(x, str) and len(str(x).strip()) < 10 else 0
    )

    features['has_short_requirements'] = df['requirements'].apply(
        lambda x: 1 if isinstance(x, str) and len(str(x).strip()) < 10 else 0
    )

    # METADATA FEATURES
    if 'telecommuting' in df.columns:
        features['telecommuting'] = df['telecommuting'].map({'f': 0, 't': 1}).fillna(0).astype(int)
    else:
        features['telecommuting'] = 0

    if 'has_company_logo' in df.columns:
        features['has_company_logo'] = df['has_company_logo'].map({'f': 0, 't': 1}).fillna(0).astype(int)
    else:
        features['has_company_logo'] = 0

    if 'has_questions' in df.columns:
        features['has_questions'] = df['has_questions'].map({'f': 0, 't': 1}).fillna(0).astype(int)
    else:
        features['has_questions'] = 0

    # PART-OF-SPEECH TAGS
    def extract_pos_counts(text):
        if not isinstance(text, str) or len(text) < 20:
            return {'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'pron': 0}

        try:
            tokens = word_tokenize(text[:500])
            pos_tags = pos_tag(tokens)

            counts = {'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'pron': 0}

            for _, tag in pos_tags:
                if tag.startswith('NN'):
                    counts['noun'] += 1
                elif tag.startswith('VB'):
                    counts['verb'] += 1
                elif tag.startswith('JJ'):
                    counts['adj'] += 1
                elif tag.startswith('RB'):
                    counts['adv'] += 1
                elif tag.startswith('PR'):
                    counts['pron'] += 1

            return counts
        except:
            return {'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'pron': 0}

    pos_results = []
    for text in tqdm(df['processed_text'].head(len(df)), desc="POS tagging", unit="docs"):
        pos_results.append(extract_pos_counts(text))

    if len(pos_results) < len(df):
        for _ in range(len(df) - len(pos_results)):
            pos_results.append({'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'pron': 0})

    features['noun_count'] = [r['noun'] for r in pos_results]
    features['verb_count'] = [r['verb'] for r in pos_results]
    features['adj_count'] = [r['adj'] for r in pos_results]
    features['adv_count'] = [r['adv'] for r in pos_results]
    features['pron_count'] = [r['pron'] for r in pos_results]

    if 'type' in df.columns:
        features['type'] = df['type'].values

    return features