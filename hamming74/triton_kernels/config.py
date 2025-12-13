import torch

HAMMING74_BLOCK_SIZE = 1024
HAMMING84_BLOCK_SIZE = 1024
GOLAY_BLOCK_SIZE = 256
FAULT_INJECTION_BLOCK_SIZE = 1024
INTERPOLATION_BLOCK_SIZE = 1024


def get_physical_dtype(codec):
    if codec == "hamming74":
        return torch.uint8
    elif codec == "hamming84":
        return torch.uint8
    elif codec == "golay":
        return torch.int32
    elif codec == "int4":
        return torch.uint8
    elif codec == "none":
        return torch.float16
    else:
        raise ValueError(f"Unknown codec: {codec}")


def get_codeword_bits(codec):
    if codec == "hamming74":
        return 7
    elif codec == "hamming84":
        return 8
    elif codec == "golay":
        return 24
    else:
        raise ValueError(f"Unknown codec: {codec}")


def get_data_bits(codec):
    if codec == "hamming74":
        return 4
    elif codec == "hamming84":
        return 4
    elif codec == "golay":
        return 12
    else:
        raise ValueError(f"Unknown codec: {codec}")


SYNDROME_LUT_HAMMING74 = torch.tensor(
    [
        -1,
        4,
        5,
        0,
        6,
        1,
        2,
        3,
    ],
    dtype=torch.int8,
)


SYNDROME_LUT_HAMMING84 = torch.tensor(
    [
        -1,
        4,
        5,
        0,
        6,
        1,
        2,
        3,
    ],
    dtype=torch.int8,
)


GOLAY_PARITY_MASKS = torch.tensor(
    [
        0b110111000111,
        0b101110001110,
        0b011100011101,
        0b111000111010,
        0b110001110101,
        0b100011101011,
        0b000111010111,
        0b001110101110,
        0b011101011100,
        0b111010111000,
        0b110101110001,
        0b101011100011,
    ],
    dtype=torch.int32,
)


class ErrorType:
    NO_ERROR = 0
    SINGLE_CORRECTED = 1
    DOUBLE_DETECTED = 2
    PARITY_ONLY = 3
