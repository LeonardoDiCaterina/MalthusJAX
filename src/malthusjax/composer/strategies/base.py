from flax import struct


@struct.dataclass
class BaseStrategy:
    """
    Base class for all algorithmic strategies in MalthusJAX.
    A Strategy configures the Composer with exactly which Engine and which Emitters to instantiate.
    """

    pass
