# Copyright 2025-26 StreamDiffusionV2 Authors
"""
TensorRT runtime engine wrapper for T5 text encoder.

Provides drop-in replacement for WanTextEncoder with:
- Tokenization handling
- Batch inference support
"""

import logging
from typing import Dict, List, Optional

import torch

from ..utilities import Engine

logger = logging.getLogger(__name__)


class T5EncoderEngine:
    """
    TensorRT engine wrapper for T5 text encoder.
    
    Usage:
        encoder = T5EncoderEngine(
            engine_path="./trt_engines/t5_encoder.engine",
            tokenizer=tokenizer,
            stream=cuda_stream,
        )
        
        embeddings = encoder(["A cat running", "A dog jumping"])
    """
    
    def __init__(
        self,
        engine_path: str,
        tokenizer,
        stream,
        text_len: int = 512,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        self.stream = stream
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.use_cuda_graph = use_cuda_graph
        self.device = device
        
        # Load engine
        self.engine = Engine(engine_path)
        self.engine.load()
        self.engine.activate()
        
        # Shape tracking
        self._last_batch_size = None
        
        logger.info(f"Loaded T5 TensorRT engine: {engine_path}")
    
    def __call__(self, text_prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode text prompts to embeddings.
        
        Args:
            text_prompts: List of text strings
        
        Returns:
            Dict containing "prompt_embeds" tensor [B, text_len, dim]
        """
        batch_size = len(text_prompts)
        
        # Tokenize
        ids, mask = self.tokenizer(
            text_prompts,
            return_mask=True,
            add_special_tokens=True,
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        
        # Allocate buffers if batch size changed
        if batch_size != self._last_batch_size:
            shape_dict = {
                "input_ids": (batch_size, self.text_len),
                "attention_mask": (batch_size, self.text_len),
            }
            self.engine.allocate_buffers(shape_dict, self.device)
            if self.use_cuda_graph:
                self.engine.reset_cuda_graph()
            self._last_batch_size = batch_size
        
        # Run inference
        outputs = self.engine.infer(
            {
                "input_ids": ids.long(),
                "attention_mask": mask.long(),
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        self.stream.synchronize()
        
        # Get embeddings and apply sequence length masking
        embeddings = outputs["text_embeddings"]
        seq_lens = mask.gt(0).sum(dim=1).long()
        
        # Zero out padding
        for i, seq_len in enumerate(seq_lens):
            embeddings[i, seq_len:] = 0.0
        
        return {"prompt_embeds": embeddings}
    
    def encode(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Simple encode returning just the embeddings tensor.
        
        Args:
            text_prompts: List of text strings
        
        Returns:
            Embeddings tensor [B, text_len, dim]
        """
        return self(text_prompts)["prompt_embeds"]
    
    def reset_cuda_graph(self):
        """Reset CUDA graph for recapture."""
        self.engine.reset_cuda_graph()
        self._last_batch_size = None


class T5EncoderEngineWithTokenizer:
    """
    Complete T5 encoder with integrated tokenizer.
    
    Self-contained replacement for WanTextEncoder.
    """
    
    def __init__(
        self,
        engine_path: str,
        tokenizer_path: str,
        stream,
        text_len: int = 512,
        use_cuda_graph: bool = True,
        device: str = "cuda",
    ):
        from causvid.models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
        
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=text_len,
            clean='whitespace'
        )
        
        self.encoder = T5EncoderEngine(
            engine_path=engine_path,
            tokenizer=self.tokenizer,
            stream=stream,
            text_len=text_len,
            use_cuda_graph=use_cuda_graph,
            device=device,
        )
    
    def __call__(self, text_prompts: List[str]) -> Dict[str, torch.Tensor]:
        return self.encoder(text_prompts)
    
    @property
    def device(self):
        return self.encoder.device
