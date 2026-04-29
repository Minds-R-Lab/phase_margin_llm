"""HuggingFace Transformers client for direct embedding-space probing.

Implements section 5.1 of the paper:  the perturbation
``u_k = u*_k + epsilon * v * cos(Omega*k)`` is added directly to the last
input token's *embedding vector* before the transformer forward pass.
The observable is the same model's own last-hidden-state at that token,
so injection and measurement live in the same hidden-state space.

This bypasses the textual-modifier bottleneck identified in the v3 smoke
tests (text-only instructions cannot reliably push the LLM along
arbitrary embedding directions).  Path beta of the validation plan.

Usage
-----
    from phase_margin.llm.transformers_client import TransformersClient

    llm = TransformersClient("Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
    text, h = llm.chat_with_perturbation(
        messages=[Message(role="system", content="..."),
                  Message(role="user",   content="...")],
        perturbation_vector=eps * v,           # shape = (hidden_dim,)
    )
    # h.shape == (hidden_dim,)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import LLMClient, Message


class TransformersClient(LLMClient):
    """qwen-style HF causal model with input-embedding perturbation hooks."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "TransformersClient requires `transformers` and `torch`. "
                "Install with: pip install transformers accelerate sentencepiece"
            ) from e

        self._torch = torch
        self.model_name = model_name
        self._device = device
        try:
            self._dtype = getattr(torch, dtype)
        except AttributeError as e:
            raise ValueError(f"unknown torch dtype '{dtype}'") from e

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).to(device).eval()

        self._embedding_layer = self.model.get_input_embeddings()
        self.hidden_dim = int(self._embedding_layer.embedding_dim)

    # ------------------------------------------------------------------
    # LLMClient protocol
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.hidden_dim

    @property
    def name(self) -> str:
        return f"transformers:{self.model_name}"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        reply, _ = self._chat_internal(
            messages=messages, perturbation_vector=None,
            seed=seed, temperature=temperature, max_tokens=max_tokens,
        )
        return reply

    # ------------------------------------------------------------------
    # The Path-beta-specific entry point
    # ------------------------------------------------------------------
    def chat_with_perturbation(
        self,
        messages: Sequence[Message],
        *,
        perturbation_vector: Optional[np.ndarray],
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> tuple[str, np.ndarray]:
        """Generate a reply AND return the last-input-token last-hidden-state.

        If ``perturbation_vector`` is supplied (shape ``(hidden_dim,)``)
        a forward hook on the embedding layer adds it to the last input
        token's embedding before any forward pass.  The hidden state
        returned is therefore the *perturbed* representation that the
        model used to produce the reply.
        """
        return self._chat_internal(
            messages=messages, perturbation_vector=perturbation_vector,
            seed=seed, temperature=temperature, max_tokens=max_tokens,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Mean-pooled last-hidden-state embedding of each text.

        Used by the data-driven probing-basis builder so the basis lives
        in the *same* hidden-state space as the perturbations.
        """
        torch = self._torch
        out = np.zeros((len(texts), self.hidden_dim), dtype=float)
        for i, t in enumerate(texts):
            if not t:
                continue
            ids = self.tok(t, return_tensors="pt").input_ids.to(self._device)
            with torch.no_grad():
                h = self.model(ids, output_hidden_states=True).hidden_states[-1]
            out[i] = h[0].float().mean(dim=0).cpu().numpy()
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _chat_internal(
        self,
        messages: Sequence[Message],
        *,
        perturbation_vector: Optional[np.ndarray],
        seed: int | None,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, np.ndarray]:
        torch = self._torch

        msgs = [{"role": m.role, "content": m.content} for m in messages]
        prompt_text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        ids = self.tok(prompt_text, return_tensors="pt").input_ids.to(self._device)
        attn = (ids != self.tok.pad_token_id).long()

        # ---- Embedding-layer hook (the heart of Path beta) -------------
        handle = None
        if perturbation_vector is not None:
            v_np = np.asarray(perturbation_vector, dtype=float).ravel()
            if v_np.size != self.hidden_dim:
                raise ValueError(
                    f"perturbation_vector dim {v_np.size} != "
                    f"model hidden_dim {self.hidden_dim}"
                )
            v_t = torch.as_tensor(v_np, dtype=self._dtype, device=self._device)
            n_input_tokens = int(ids.shape[1])

            def _hook(module, inp, out):
                # out shape: (batch, seq_len, hidden_dim).
                # Add v to the LAST INPUT TOKEN'S embedding only -- not
                # generated tokens.  We detect that by checking seq_len:
                # the prompt-encoding pass has seq_len == n_input_tokens;
                # decoding passes feed tokens one at a time after that.
                if out.shape[1] == n_input_tokens:
                    new_out = out.clone()
                    new_out[:, -1, :] = new_out[:, -1, :] + v_t
                    return new_out
                return out

            handle = self._embedding_layer.register_forward_hook(_hook)

        try:
            if seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

            # 1) Forward pass with the hook ON to extract the perturbed
            #    last-input-token hidden state.
            with torch.no_grad():
                fwd = self.model(
                    ids, attention_mask=attn, output_hidden_states=True,
                )
                last_hidden = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

            # 2) Generate the reply.  The hook adds the perturbation only
            #    to the prompt-encoding pass, not to per-token decoding.
            do_sample = bool(temperature and temperature > 0.0)
            with torch.no_grad():
                gen = self.model.generate(
                    ids,
                    attention_mask=attn,
                    max_new_tokens=int(max_tokens),
                    do_sample=do_sample,
                    temperature=float(max(temperature, 1e-6)) if do_sample else 1.0,
                    pad_token_id=self.tok.pad_token_id,
                )
            new_tokens = gen[0, ids.shape[1]:]
            reply = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        finally:
            if handle is not None:
                handle.remove()

        return reply, last_hidden
