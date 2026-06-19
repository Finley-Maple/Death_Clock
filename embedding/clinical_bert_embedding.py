"""
BioClinicalBERT Extractor for Patient Descriptions

Uses emilyalsentzer/Bio_ClinicalBERT (110M-parameter bidirectional BERT
pre-trained on MIMIC-III clinical notes) to extract mean-pooled text embeddings
for patient descriptions.

Output dimension: 768 (native BERT hidden size).
Max sequence length: 512 tokens (hard BERT limit; long texts are truncated).

IMPORTANT — truncation caveat for S3-Full (Full Clinical Prose):
The full clinical summary can exceed 512 tokens once demographics, 21 lab
values, 8 comorbidity flags, and disease history are concatenated.  BioBERT
silently truncates to 512 tokens, so the tail of the disease-history section
(which comes last) may be lost.  This is noted as a limitation in the
manuscript and makes Qwen3-8B (max_length=8192) more suitable for S3-Full.

Usage:
    # Run on a text CSV (for Settings 3 or S3-Full):
    python embedding/clinical_bert_embedding.py \\
        --input-csv data/preprocessed/text_before60.csv \\
        --output-dir data/preprocessed/embeddings_bcb_text \\
        --tag patient

    # Run on the unified-prose CSV (S3-Full):
    python embedding/clinical_bert_embedding.py \\
        --input-csv data/preprocessed/biomarker_text_before60.csv \\
        --output-dir data/preprocessed/embeddings_bcb_summary \\
        --tag patient

Requires: transformers>=4.30.0, torch>=2.0.0
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BertEmbeddingConfig:
    """Configuration for Bio_ClinicalBERT embedding extraction."""
    # Revision 3c22c28 has model.safetensors; older default (d5892b) only has .bin
    # which triggers a torch-version security error in transformers>=4.53.
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    revision: str = "3c22c28ae9c1619228e31dc7630645fee6081c98"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512          # BERT hard limit
    batch_size: int = 32           # BCB is small, can use larger batches
    normalize: bool = True         # L2-normalize outputs (matches Qwen convention)
    hf_token: Optional[str] = None


# ---------------------------------------------------------------------------
# Mean pooling (standard for BERT-family encoders)
# ---------------------------------------------------------------------------

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Average the token representations over non-padding tokens.

    This is the standard pooling method for sentence/text embeddings from
    bidirectional encoders (BERT, RoBERTa, ClinicalBERT).
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
    token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / token_counts


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ClinicalBertExtractor:
    """Extract embeddings using emilyalsentzer/Bio_ClinicalBERT."""

    def __init__(self, config: BertEmbeddingConfig = None):
        self.config = config or BertEmbeddingConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load Bio_ClinicalBERT using AutoModel/AutoTokenizer."""
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading {self.config.model_name} on {self.config.device}...")

        load_kw = {}
        if self.config.hf_token:
            load_kw["token"] = self.config.hf_token
        if self.config.revision:
            load_kw["revision"] = self.config.revision

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            **load_kw,
        )

        model_kw = {}
        if self.config.device == "cuda":
            model_kw["torch_dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            **{**model_kw, **load_kw},
        ).to(self.config.device)
        self.model.eval()

        dim = self.model.config.hidden_size
        logger.info(f"Loaded {self.config.model_name}: dim={dim}, "
                    f"max_length={self.config.max_length}, device={self.config.device}")

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def extract_embedding(self, text: str) -> np.ndarray:
        """Extract embedding for a single text string."""
        return self.extract_batch_embeddings([text])[0]

    def extract_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract mean-pooled embeddings for a batch of texts.

        Returns ndarray of shape (len(texts), 768).
        Texts exceeding 512 tokens are silently truncated — expected behaviour
        for BERT-family models.
        """
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = mean_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

            if self.config.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().float().numpy()


# ---------------------------------------------------------------------------
# EID-based patient embedding pipeline
# ---------------------------------------------------------------------------

class ClinicalBertProcessor:
    """Process patient texts keyed by eid and extract BioClinicalBERT embeddings."""

    def __init__(self, config: BertEmbeddingConfig = None,
                 extractor: Optional[ClinicalBertExtractor] = None):
        self.config = config or BertEmbeddingConfig()
        self._extractor = extractor
        self._load_failed = False

    @property
    def extractor(self) -> ClinicalBertExtractor:
        """Lazy-load the model on first use."""
        if self._load_failed:
            raise RuntimeError("Model loading previously failed. Fix the issue and restart.")
        if self._extractor is None:
            try:
                self._extractor = ClinicalBertExtractor(self.config)
            except Exception:
                self._load_failed = True
                raise
        return self._extractor

    # -- input loaders (no model needed) ------------------------------------

    @staticmethod
    def load_from_directory(input_dir: Path) -> Dict[int, str]:
        """Load eid -> text from a directory of eid_<id>.txt files."""
        texts: Dict[int, str] = {}
        for f in sorted(input_dir.glob("eid_*.txt")):
            eid = int(f.stem.split("_", 1)[1])
            texts[eid] = f.read_text(encoding="utf-8")
        logger.info(f"Loaded {len(texts)} patient texts from {input_dir}")
        return texts

    @staticmethod
    def load_from_csv(csv_path: Path, text_col: str = "text") -> Dict[int, str]:
        """Load eid -> text from a CSV with columns [eid, <text_col>]."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        df["eid"] = df["eid"].astype(int)
        texts = dict(zip(df["eid"], df[text_col].astype(str)))
        logger.info(f"Loaded {len(texts)} patient texts from {csv_path}")
        return texts

    # -- main pipeline ------------------------------------------------------

    def process(self, texts: Dict[int, str]) -> Dict[int, np.ndarray]:
        """Extract an embedding for each eid, batched for efficiency."""
        if not texts:
            logger.warning("No texts to embed.")
            return {}

        try:
            _ = self.extractor
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return {}

        eids = list(texts.keys())
        all_texts = [texts[eid] for eid in eids]
        batch_size = self.config.batch_size
        embeddings: Dict[int, np.ndarray] = {}

        for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding patients (BCB)"):
            batch_eids = eids[i: i + batch_size]
            batch_texts = all_texts[i: i + batch_size]
            try:
                batch_embs = self.extractor.extract_batch_embeddings(batch_texts)
                for eid, emb in zip(batch_eids, batch_embs):
                    embeddings[eid] = emb
            except Exception as e:
                logger.error(f"Error embedding batch starting at index {i}: {e}")
                for eid, text in zip(batch_eids, batch_texts):
                    try:
                        embeddings[eid] = self.extractor.extract_embedding(text)
                    except Exception as e2:
                        logger.error(f"Error embedding eid {eid}: {e2}")

        return embeddings

    # -- save / load (same interface as PatientEmbeddingProcessor) ----------

    @staticmethod
    def save(embeddings: Dict[int, np.ndarray], output_dir: Path, tag: str = "patient"):
        """Save embeddings as {tag}_embeddings.npz and metadata JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not embeddings:
            logger.warning("No embeddings to save.")
            return

        npz_path = output_dir / f"{tag}_embeddings.npz"
        str_keyed = {str(eid): emb for eid, emb in embeddings.items()}
        np.savez_compressed(npz_path, **str_keyed)
        logger.info(f"Saved {len(embeddings)} embeddings to {npz_path}")

        sample_emb = next(iter(embeddings.values()))
        meta = {
            "num_patients": len(embeddings),
            "embedding_dim": int(sample_emb.shape[0]) if sample_emb.ndim > 0 else 0,
            "model": "emilyalsentzer/Bio_ClinicalBERT",
            "eids": sorted(embeddings.keys()),
        }
        meta_path = output_dir / f"{tag}_embedding_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved metadata to {meta_path}")

    @staticmethod
    def load(output_dir: Path, tag: str = "patient") -> Dict[int, np.ndarray]:
        """Load previously saved embeddings."""
        npz_path = output_dir / f"{tag}_embeddings.npz"
        data = np.load(npz_path, allow_pickle=False)
        return {int(k): data[k] for k in data.files}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Bio_ClinicalBERT text embeddings for patient texts (eid-based).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setting 3 — disease-history prose:
  python embedding/clinical_bert_embedding.py \\
      --input-csv data/preprocessed/text_before60.csv \\
      --output-dir data/preprocessed/embeddings_bcb_text \\
      --tag patient

  # S3-Full — full clinical prose (note 512-token truncation):
  python embedding/clinical_bert_embedding.py \\
      --input-csv data/preprocessed/biomarker_text_before60.csv \\
      --output-dir data/preprocessed/embeddings_bcb_summary \\
      --tag patient
""",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input-dir", type=Path,
                     help="Directory with eid_*.txt files.")
    grp.add_argument("--input-csv", type=Path,
                     help="CSV with [eid, text] columns.")

    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to save embeddings (.npz + metadata JSON).")
    parser.add_argument("--tag", type=str, default="patient",
                        help="File prefix for output files (default: 'patient').")
    parser.add_argument("--text-col", type=str, default="text",
                        help="Text column name in CSV (default: 'text').")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding extraction (default: 32).")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length — BERT hard limit is 512 (default: 512).")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip L2 normalization of embeddings.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace access token (or set HF_TOKEN env var).")
    args = parser.parse_args()

    import os as _os
    hf_token = args.hf_token or _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if args.input_dir:
        texts = ClinicalBertProcessor.load_from_directory(args.input_dir)
    else:
        texts = ClinicalBertProcessor.load_from_csv(args.input_csv, text_col=args.text_col)

    if not texts:
        logger.error("No input texts found.")
        return

    config = BertEmbeddingConfig(
        max_length=args.max_length,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        hf_token=hf_token,
    )
    processor = ClinicalBertProcessor(config=config)
    embeddings = processor.process(texts)
    ClinicalBertProcessor.save(embeddings, args.output_dir, tag=args.tag)

    logger.info(f"Done! Processed {len(embeddings)} patients.")
    if embeddings:
        sample = next(iter(embeddings.values()))
        logger.info(f"Embedding dim: {sample.shape}")


if __name__ == "__main__":
    main()
