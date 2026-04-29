"""Utilities for pretty-printing model topology and status."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from reservoir.core.types import DetailsMeta, ShapesMeta, TopologyMeta


def _fmt_dim(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "?"
    return "[" + "x".join(str(d) for d in shape) + "]"


def _fmt_layers(layers: Iterable[int] | None) -> str:
    if not layers:
        return "None"
    return "->".join(str(int(v)) for v in layers)


def print_topology(meta: TopologyMeta) -> None:
    """
    Render a 7-step topology report combining orchestrator (steps 1-4) and model (steps 5-7) metadata.
    Required keys in meta['shapes'] (may be None):
      input, preprocessed, projected, adapter, internal, feature, output
    Optional details: preprocess, agg_mode, student_layers
    """
    if not meta:
        return

    shapes: ShapesMeta
    existing_shapes = meta.get("shapes")
    if existing_shapes is None:
        shapes = {}
    else:
        shapes = existing_shapes

    details: DetailsMeta
    existing_details = meta.get("details")
    if existing_details is None:
        details = {}
    else:
        details = existing_details

    s_in = shapes.get("input")
    s_pre = shapes.get("preprocessed") or s_in
    s_proj = shapes.get("projected")
    s_adapter = shapes.get("adapter")
    s_internal = shapes.get("internal")
    s_feat = shapes.get("feature")
    s_out = shapes.get("output")

    preprocess_method = details.get("preprocess") or "None"
    agg_mode = details.get("agg_mode") or "None"
    readout_label = details.get("readout")
    adapter_label = details.get("adapter") or "Skipped"
    student_layers = details.get("student_layers")
    _fmt_layers(student_layers)

    print(f"=== Model Topology: {meta.get('type', 'UNKNOWN')} ===")
    print(f"1. Input Data      : {_fmt_dim(s_in)}")
    if preprocess_method in {"None", None, "RAW", "Raw", "raw"}:
        print("2. Preprocessing   : Skipped")
    else:
        print(f"2. Preprocessing   : {_fmt_dim(s_in)} -> {_fmt_dim(s_pre)} (Method: {preprocess_method})")
    if s_proj is not None:
        print(f"3. Input Projection: {_fmt_dim(s_pre)} -> {_fmt_dim(s_proj)}")
    else:
        print("3. Input Projection: Skipped")

    if s_adapter is not None:
        print(f"4. Adapter         : {_fmt_dim(s_proj or s_pre)} -> {_fmt_dim(s_adapter)} (Flatten/Struct)")
    else:
        print(f"4. Adapter         : {adapter_label}")

    # Step 5/6: internal/model
    has_aggregation = agg_mode not in ("None", None)

    if student_layers is not None and len(student_layers) > 0 and s_adapter is not None:
        chain = [_fmt_dim(s_adapter)]
        chain.extend(f"[{v}]" for v in student_layers)
        if not has_aggregation and s_feat is not None:
            chain.append(_fmt_dim(s_feat))
        model_desc = " -> ".join(chain)
    elif s_internal is not None:
        # If internal is already the output (e.g., linear), avoid duplicating output.
        if not has_aggregation and s_feat is not None and s_internal == s_feat:
            model_desc = f"{_fmt_dim(s_proj or s_pre)} -> {_fmt_dim(s_internal)}"
        else:
            model_desc = f"{_fmt_dim(s_proj or s_pre)} -> {_fmt_dim(s_internal)}"
            if not has_aggregation and s_feat is not None and s_feat != s_internal:
                model_desc = f"{model_desc} -> {_fmt_dim(s_feat)}"
    else:
        model_desc = "n/a"
    print(f"5. Model           : {model_desc}")

    prev_dim = s_internal or s_adapter or s_proj or s_pre
    if not agg_mode or agg_mode == "None":
        print("6. Aggregation     : Skipped (Architecture implies flat output)")
    else:
        # User Request: Report flattened 2D output for Sequence Aggregation
        s_feat_disp = s_feat
        if str(agg_mode).lower() == "sequence" and s_feat and len(s_feat) == 3:
             # (Batch, Time, Feat) -> (Batch*Time, Feat)
             flattened_dim = s_feat[0] * s_feat[1]
             s_feat_disp = (flattened_dim, s_feat[2])
        
        print(f"6. Aggregation     : {_fmt_dim(prev_dim)} -> {_fmt_dim(s_feat_disp)} (Mode: {agg_mode})")

    if readout_label in ("None", None):
        print("7. Readout         : Skipped (end-to-end model output)")
    else:
        # User Request: Show flattened 2D topology for Sequence tasks and W_out size
        w_size = ""
        feat_dim_str = _fmt_dim(s_feat)
        out_dim_str = _fmt_dim(s_out)
        
        # W_out only makes sense for Ridge (single weight matrix), not for FNN
        is_ridge = "Ridge" in str(readout_label)
        if is_ridge and s_feat and len(s_feat) > 0 and s_out and len(s_out) > 0:
             w_rows = s_feat[-1]
             w_cols = s_out[-1]
             w_size = f", W_out=[{w_rows}x{w_cols}]"
             
             # If Sequence mode, show flattened shapes: [BxT, F] -> [BxT, O]
             if agg_mode == "sequence" and len(s_feat) >= 3:
                  # infer BxT
                  total_time = 1
                  for d in s_feat[:-1]:
                      total_time *= int(d)
                  feat_dim_str = f"[{total_time}x{w_rows}]"
                  
                  # infer output BxT
                  # output might be (B, T, O)
                  if len(s_out) >= 3:
                      total_time_out = 1
                      for d in s_out[:-1]:
                          total_time_out *= int(d)
                      out_dim_str = f"[{total_time_out}x{w_cols}]"

        print(f"7. Readout         : {feat_dim_str} -> {out_dim_str} ({readout_label}{w_size})")
    print("=" * 60)
