# DF11 Integration

## DF11 Eager mode

- python `tests/models/df11/dff11_inference.py`
    - Eager mode, `enforce_eager=True`
    - HF transformers-based, `model_impl="transformers"`
    - V0, V1: Done


## DF11 vLLM implementation

- python `tests/models/df11/dff11_inference.py`
    - No eager mode, `enforce_eager=False` (default is False)
    - V0: Done
    - V1: WIP