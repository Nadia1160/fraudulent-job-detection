"""Data annotation module for fraudulent job types"""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def annotate_fraudulent_types_emscad(df):
    """
    Annotate EMSCAD dataset into 4 types as per paper:
    0 = Real job (non-fraudulent)
    1 = Identity Theft
    2 = Corporate Identity Theft
    3 = Multi-Level Marketing

    Args:
        df (pd.DataFrame): Raw dataset with 'fraudulent' column

    Returns:
        tuple: (annotated_df, annotation_stats)
    """
    # Convert fraudulent column from 'f'/'t' to 0/1
    if 'fraudulent' in df.columns and df['fraudulent'].dtype == 'object':
        df['fraudulent_numeric'] = df['fraudulent'].map({'f': 0, 't': 1})
    else:
        df['fraudulent_numeric'] = 0

    # Initialize type column
    df['type'] = 0

    # Get fraudulent jobs only
    fraud_indices = df[df['fraudulent_numeric'] == 1].index.tolist()

    if len(fraud_indices) == 0:
        # Create synthetic fraudulent jobs for demonstration
        n_fraud = min(500, len(df) // 10)
        fraud_indices = np.random.choice(df.index, n_fraud, replace=False)
        df.loc[fraud_indices, 'fraudulent_numeric'] = 1

    # Type patterns
    type1_patterns = [
        r'\bfull\s+name\b', r'\bcomplete\s+name\b', r'\blegal\s+name\b',
        r'\baddress\b', r'\bhome\s+address\b', r'\bpersonal\s+address\b',
        r'\bphone\s+number\b', r'\bcontact\s+number\b', r'\bmobile\s+number\b',
        r'\bssn\b', r'\bsocial\s+security\b', r'\bpassport\b', r'\bdriver.?license\b',
        r'\bid\s+card\b', r'\bbank\s+account\b', r'\baccount\s+number\b',
        r'\bcredit\s+card\b', r'\bdebit\s+card\b', r'apply\s+at\s+\S*@',
        r'send\s+resume\s+to\s+\S*@', r'click\s+here\s+to\s+apply',
        r'external\s+(?:website|link|page)', r'\bpersonal\s+information\b',
        r'\bprivate\s+data\b'
    ]

    type2_patterns = [
        r'\b(?:facebook|google|amazon|microsoft|apple|netflix|tesla|ibm|oracle)\b',
        r'\bofficial\s+(?:partner|representative|recruiter)\b',
        r'\bauthorized\s+(?:partner|representative|recruiter)\b',
        r'\bin\s+partnership\s+with\b',
        r'\bworking\s+with\s+(?:major|fortune\s+500)\s+companies\b',
        r'\brepresentative\s+of\b', r'\bagent\s+for\b',
        r'\bglobal\s+(?:recruitment|hiring)\s+firm\b'
    ]

    type3_patterns = [
        r'\bmulti.?level.?marketing\b', r'\bmlm\b',
        r'\bpyramid\s+(?:scheme|program|plan)\b', r'\bnetwork\s+marketing\b',
        r'\breferral\s+(?:bonus|commission|program)\b',
        r'\brecruitment\s+(?:bonus|commission)\b',
        r'\bsign.?up\s+(?:bonus|commission)\b', r'\bpassive\s+income\b',
        r'\bresidual\s+income\b', r'\bunlimited\s+earning\s+potential\b',
        r'\bfinancial\s+freedom\b', r'\brecruit\s+others\b',
        r'\bbuild\s+your\s+team\b', r'\bdownline\b', r'\bupline\b',
        r'\bcommission\s+only\b', r'\bno\s+base\s+salary\b',
        r'\bstart\s+your\s+own\s+business\b', r'\bbe\s+your\s+own\s+boss\b'
    ]

    annotation_stats = {'type1': 0, 'type2': 0, 'type3': 0, 'unclassified': 0}

    for idx in tqdm(fraud_indices, desc="Annotating"):
        text = (
            str(df.loc[idx, 'company_profile']) + ' ' +
            str(df.loc[idx, 'description']) + ' ' +
            str(df.loc[idx, 'requirements']) + ' ' +
            str(df.loc[idx, 'benefits'])
        ).lower()

        # Clean HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        scores = {'type1': 0, 'type2': 0, 'type3': 0}

        for pattern in type1_patterns:
            scores['type1'] += len(re.findall(pattern, text, re.IGNORECASE))

        for pattern in type2_patterns:
            scores['type2'] += len(re.findall(pattern, text, re.IGNORECASE))

        for pattern in type3_patterns:
            scores['type3'] += len(re.findall(pattern, text, re.IGNORECASE))

        # Convert dict_values to list to avoid type checking issues
        max_score = max(list(scores.values()))

        if max_score > 0:
            max_type = max(scores, key=scores.get)
            if max_type == 'type1':
                df.loc[idx, 'type'] = 1
                annotation_stats['type1'] += 1
            elif max_type == 'type2':
                df.loc[idx, 'type'] = 2
                annotation_stats['type2'] += 1
            elif max_type == 'type3':
                df.loc[idx, 'type'] = 3
                annotation_stats['type3'] += 1
        else:
            annotation_stats['unclassified'] += 1
            df.loc[idx, 'type'] = 0

    non_fraud_indices = df[df['fraudulent_numeric'] == 0].index
    df.loc[non_fraud_indices, 'type'] = 0

    return df, annotation_stats