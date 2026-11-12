"""Public exceptions distinguishing unsupported models from solver bugs."""


class UnsupportedModelError(ValueError):
    """The requested symbolic method cannot represent or eliminate this model.

    This is a known capability limit, not evidence of infeasibility. The
    message identifies the unsupported structure. Unexpected implementation
    exceptions are deliberately not converted to this class.
    """
