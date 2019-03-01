from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
from tqdm import tqdm

from .sample import docs_from_paths, json_docs_from_paths

def idf_weights(docs: Dict[str, List[str]],
                vectorizer: TfidfVectorizer = TfidfVectorizer()) -> Dict[str, float]:
    """ Compute the idf-weights for each term in the corpus """
    # convert the texts into a format usable by our vectorizer
    texts = [' '.join(text) for text in docs.values()]
    # fit the vectorizer to get the idf weights
    vectorizer.fit(texts)
    # return a vocab-like item with the idf weights
    return { term: vectorizer.idf_[i] for term, i in tqdm(vectorizer.vocabulary_.items()) }

def save_idf_weights(weights: Dict[str, float], fp) -> None:
    for word, weight in weights.items():
        fp.write(f'{word} {weight}\n')

if __name__ == '__main__':
    paths =  [
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/DEV/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/ANALYSIS1/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/ANALYSIS2/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/EVAL1/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/EVAL2/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/EVAL3/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/DEV/audio/mt_store/umd-smt-v2.4_material-asr-sw-v5.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/ANALYSIS1/audio/mt_store/umd-smt-v2.4_material-asr-sw-v5.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/ANALYSIS2/audio/mt_store/umd-smt-v2.4_material-asr-sw-v5.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/EVAL1/audio/mt_store/umd-smt-v2.4_material-asr-sw-v5.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/EVAL2/audio/mt_store/umd-smt-v2.4_material-asr-sw-v5.0/*.txt",
        # "/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A/EVAL3/audio/mt_store/umd-smt-v2.4_material-asr-sw-v5.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/DEV/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/DEV/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS1/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS1/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS2/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS2/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL1/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL1/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL2/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL2/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL3/text/mt_store/umd-smt-v2.4_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL3/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/*.txt"
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/DEV/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/DEV/audio/morphology_store/material-scripts-morph-v4.1_material-asr-so-v7.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/ANALYSIS1/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/ANALYSIS1/audio/morphology_store/material-scripts-morph-v4.1_material-asr-so-v7.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/ANALYSIS2/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/ANALYSIS2/audio/morphology_store/material-scripts-morph-v4.1_material-asr-so-v7.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/EVAL1/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/EVAL1/audio/morphology_store/material-scripts-morph-v4.1_material-asr-so-v7.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/EVAL2/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/EVAL2/audio/morphology_store/material-scripts-morph-v4.1_material-asr-so-v7.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/EVAL3/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        # "/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S/EVAL3/audio/morphology_store/material-scripts-morph-v4.1_material-asr-so-v7.0/*.txt"
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/DEV/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/DEV/audio/morphology_store/material-scripts-morph-v4.1_material-asr-tl-v5.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS1/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS1/audio/morphology_store/material-scripts-morph-v4.1_material-asr-tl-v5.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS2/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/ANALYSIS2/audio/morphology_store/material-scripts-morph-v4.1_material-asr-tl-v5.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL1/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL1/audio/morphology_store/material-scripts-morph-v4.1_material-asr-tl-v5.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL2/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL2/audio/morphology_store/material-scripts-morph-v4.1_material-asr-tl-v5.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL3/text/morphology_store/material-scripts-morph-v4.1_cu-code-switching-v7.0_sent-split-v3.0/*.txt",
        "/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B/EVAL3/audio/morphology_store/material-scripts-morph-v4.1_material-asr-tl-v5.0/*.txt"
    ]

    print('Loading docs')
    docs = json_docs_from_paths(paths)
    print('Computing tf-idf weights')
    idfs = idf_weights(docs)

    with open('idf_weights/tl_idf.txt', 'w') as fp:
        save_idf_weights(idfs, fp)
