"""Optional CUDA/Triton operators; importing STEPQuant does not load Triton.

STEPQuant decode is dispatched by ``pool``:
  * ``readout`` / ``segment_read`` read the stored state and produce output.
  * ``rows`` computes key-row scales and affine update coefficients.
  * ``fit`` performs one weighted column fit and final requantization.
  * ``writeback`` schedules the latter two stages on a background stream;
    ``resources`` bounds its SM usage and ``priority_graph`` prioritizes decode.
    ``profiles`` selects STEPQuant@6 writer budgets per captured batch; @4/@6
    share the same one-fit kernels and tight 2/4/6/8-bit code storage.

``packed_read``, ``bitpack`` and ``byte_codec`` handle storage layouts.
``fused`` supplies synchronous updates and initial state encoding.
``format_pool``, ``formats`` and ``floating`` implement synchronous formats.
"""
