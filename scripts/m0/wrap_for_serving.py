#!/usr/bin/env python3
"""Save a merged M0 in the form vLLM can serve: the base's multimodal wrapper, not the text-only extract.

Qwen3.5 (4B and 9B) ships as a MULTIMODAL checkpoint -- the base repo's config.json is the ``Qwen3_5ForConditionalGeneration`` wrapper (a ``text_config`` + a ``vision_config``), and vLLM's Qwen3.5 handler ONLY loads that wrapper (``Qwen3_5Config``). But the whole M0 build loads the base with ``AutoModelForCausalLM.from_pretrained(base)``, which extracts the TEXT sub-model (``Qwen3_5ForCausalLM`` / ``Qwen3_5TextConfig``). Merging the LoRA and calling ``save_pretrained`` on that extract writes a text-only checkpoint, and ``vllm serve`` dies on it with ``Invalid type of HuggingFace config. Expected type: Qwen3_5Config, but found type: Qwen3_5TextConfig``. This is not 4B-specific; without this step the 9B build fails identically.

The fix, and why it is safe. The CausalLM extraction does NOT rename the weights -- the merged text tensors keep the base's own ``model.language_model.*`` names -- so we can load a fresh multimodal wrapper, overwrite its ``language_model`` weights with the merged ones (the vision tower and everything else stay at base values, unused when serving text-only), and save the wrapper plus its image/video preprocessor. The result serves under vLLM AND still loads text-only under ``AutoModelForCausalLM`` (the identical extraction path that built M0), so every downstream SFT arm that points ``--base_model`` at M0 is completely unchanged. The only on-disk difference from the old text-only save is +~1 GB of vision weights and the two preprocessor configs.

Memory, and why the device is chosen rather than assumed. Peak is base-wrapper + merged text weights CO-RESIDENT: ~18 GB for 4B on the 24 GB L4, ~34 GB for 9B on the 48 GB L40S. ``.to(dev)`` is a no-op when the merged model is already on the GPU (the warm-start case), so it does not add a third copy. That co-residency is the whole problem at scale: it is roughly 2x the model, so a 27B (~54 GB in bf16) needs ~108 GB and OOMs on every single card short of a B200 -- at the LAST step of the run, after training has already succeeded and after you have paid to download the base weights. Picking ``cuda`` unconditionally, which this function used to do, is therefore a size-dependent trap that only fires once the study scales up.

So the device is now MEASURED, not assumed: ``wrap_device="auto"`` computes how many bytes the re-wrap would have to add to the card and falls back to CPU when they do not fit with headroom. 4B and 9B are unaffected -- they fit, so they still run on the GPU exactly as before, and the byte-identical output is what makes the sizes comparable. CPU is slower but this runs once per build and costs minutes, which is nothing against losing the run. Pass ``cuda`` or ``cpu`` to override when you want to force the issue.
"""

from __future__ import annotations

from pathlib import Path


def _load_wrapper(base_model, dtype):
    """Load the full multimodal wrapper for ``base_model`` (``Qwen3_5ForConditionalGeneration``)."""
    from transformers import AutoModelForImageTextToText
    try:
        return AutoModelForImageTextToText.from_pretrained(
            base_model, trust_remote_code=True, dtype=dtype,
        )
    except (ValueError, KeyError):
        # Auto class did not map this arch; instantiate the concrete class the config names.
        import transformers
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        cls = getattr(transformers, cfg.architectures[0])
        return cls.from_pretrained(base_model, trust_remote_code=True, dtype=dtype)


def _save_preprocessor(base_model, out_dir):
    """Write the image/video preprocessor next to the weights -- vLLM needs it for the wrapper arch."""
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(base_model, trust_remote_code=True).save_pretrained(out_dir)
        print(f"[m0] wrote image/video preprocessor from {base_model}")
    except Exception as exc:  # noqa: BLE001 -- best effort; leave a precise manual fallback
        print(f"[m0] WARNING: could not save the preprocessor ({exc}). vLLM will refuse the model "
              f"without it; fetch it by hand with:\n"
              f"    hf download {base_model} preprocessor_config.json "
              f"video_preprocessor_config.json --local-dir {out_dir}")


def _bytes_of(model):
    """Resident size of a model's parameters and buffers, in bytes."""
    return sum(t.numel() * t.element_size()
               for t in list(model.parameters()) + list(model.buffers()))


def _pick_wrap_device(merged, requested):
    """Resolve ``wrap_device``: does the co-resident re-wrap actually fit on the card?

    Returns the device string. ``auto`` measures rather than guesses -- see the module docstring for
    why an unconditional ``cuda`` is a trap that only fires at 27B.
    """
    import torch

    if requested in ("cuda", "cpu"):
        return requested
    if requested != "auto":
        raise ValueError(f"wrap_device must be one of auto/cuda/cpu, got {requested!r}")
    if not torch.cuda.is_available():
        return "cpu"

    merged_bytes = _bytes_of(merged)
    on_cuda = any(p.is_cuda for p in merged.parameters())
    # The wrapper is the same text stack plus a vision tower; 1.1x the merged size is a fair estimate.
    # Only the merged tensors we still have to COPY across count as additional demand, which is none
    # when the merged model already lives on the card.
    need = merged_bytes * 1.1 + (0 if on_cuda else merged_bytes)
    free, _total = torch.cuda.mem_get_info()
    headroom = 4 * 1024 ** 3          # allocator fragmentation and the save-time buffers
    gib = 1024 ** 3
    fits = free > need + headroom
    print(f"[m0] re-wrap sizing: merged {merged_bytes / gib:.1f} GiB "
          f"({'on GPU' if on_cuda else 'on CPU'}), needs ~{need / gib:.1f} GiB more, "
          f"{free / gib:.1f} GiB free -> {'cuda' if fits else 'cpu'}")
    if not fits:
        print("[m0] re-wrap falling back to CPU: the wrapper and the merged weights cannot be "
              "co-resident on this card. Slower, but it is this or an OOM at the final step.")
    return "cuda" if fits else "cpu"


def save_servable_m0(merged, base_model, out_dir, tokenizer, wrap_device="auto"):
    """Re-embed ``merged`` (a text-extracted, LoRA-merged M0) into the base multimodal wrapper and save it.

    Replaces a plain ``merged.save_pretrained(out_dir)`` -- see the module docstring for why that is
    unservable. Aborts loudly if the merged weights do not map cleanly onto the wrapper's
    ``language_model.*`` namespace, so a silent no-op export can never ship.

    ``wrap_device`` is ``auto`` (measure and fall back to CPU if the co-resident re-wrap will not
    fit), ``cuda``, or ``cpu``.
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = _pick_wrap_device(merged, wrap_device)
    wrapper = _load_wrapper(base_model, torch.bfloat16).to(dev)
    wsd = wrapper.state_dict()

    # The CausalLM extraction may FLATTEN the wrapper's `model.language_model.*` namespace to `model.*`
    # (transformers-version dependent), so an exact key match drops every language tensor and the guard
    # below aborts. Match on the key with the `language_model.` infix removed -- stable across both
    # layouts -- then write into the wrapper under ITS (infixed) names.
    text_state = merged.state_dict()

    def _norm(k):
        return k.replace("language_model.", "")

    wsd_norm = {_norm(k): k for k in wsd}   # normalized wrapper key -> actual wrapper key
    applicable, skipped = {}, []
    for k, v in text_state.items():
        wk = wsd_norm.get(_norm(k))
        if wk is not None:
            applicable[wk] = v.to(dev)
        else:
            skipped.append(k)

    lang_not_applied = [k for k in wsd if "language_model" in k and k not in applicable]
    if lang_not_applied:
        raise SystemExit(
            f"[m0] re-wrap would leave {len(lang_not_applied)} language weights at base values "
            f"(e.g. {lang_not_applied[:4]}) -- M0's fine-tuning would be silently dropped. Aborting."
        )
    noise = [k for k in skipped if "lm_head" not in k]  # a tied lm_head has no separate home; that is fine
    if noise:
        raise SystemExit(
            f"[m0] merged model has {len(noise)} weights with no slot in the wrapper "
            f"(e.g. {noise[:4]}); the language_model.* namespace assumption is wrong -- inspect first."
        )

    wrapper.load_state_dict(applicable, strict=False)  # in-place copy into the wrapper's language_model
    print(f"[m0] re-wrapped for serving: {len(applicable)} merged text tensors -> {base_model} wrapper "
          f"({len(wsd) - len(applicable)} vision/misc kept from base)")

    del applicable, text_state
    if dev == "cuda":
        torch.cuda.empty_cache()

    wrapper.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    _save_preprocessor(base_model, out_dir)
