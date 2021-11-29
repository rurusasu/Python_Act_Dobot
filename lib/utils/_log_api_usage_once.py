import torch


def _log_api_usage_once(obj: str) -> None:  # type: ignore
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return
    # NOTE: obj can be an object as well, but mocking it here to be
    # only a string to appease torchscript
    if isinstance(obj, str):
        torch._C._log_api_usage_once(obj)
    else:
        torch._C._log_api_usage_once(f"{obj.__module__}.{obj.__class__.__name__}")
