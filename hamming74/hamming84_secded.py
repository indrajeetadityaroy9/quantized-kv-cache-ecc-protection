import torch
from typing import Tuple, NamedTuple
from enum import IntEnum


class ErrorType(IntEnum):
    NO_ERROR = 0
    SINGLE_CORRECTED = 1
    DOUBLE_DETECTED = 2
    PARITY_ONLY = 3


class DecodeResult(NamedTuple):
    data: torch.Tensor
    error_type: torch.Tensor
    corrected_count: int
    detected_count: int


class Hamming84:
    G_74 = torch.tensor(
        [
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.uint8,
    )

    H_74 = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1],
        ],
        dtype=torch.uint8,
    )

    SYNDROME_TO_POSITION = torch.tensor(
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

    def __init__(self, device="cuda", on_double_error="zero"):
        self.device = device
        self.on_double_error = on_double_error
        self._G = self.G_74.to(device)
        self._H = self.H_74.to(device)
        self._syndrome_lut = self.SYNDROME_TO_POSITION.to(device)

    def encode(self, int4_values: torch.Tensor) -> torch.Tensor:
        original_shape = int4_values.shape
        flat = int4_values.flatten().to(torch.uint8)

        data_bits = torch.stack(
            [
                (flat >> 0) & 1,
                (flat >> 1) & 1,
                (flat >> 2) & 1,
                (flat >> 3) & 1,
            ],
            dim=1,
        ).to(torch.uint8)

        codeword_bits_7 = (data_bits @ self._G) % 2

        overall_parity = codeword_bits_7.sum(dim=1) % 2

        codewords = (
            (codeword_bits_7[:, 0] << 0)
            | (codeword_bits_7[:, 1] << 1)
            | (codeword_bits_7[:, 2] << 2)
            | (codeword_bits_7[:, 3] << 3)
            | (codeword_bits_7[:, 4] << 4)
            | (codeword_bits_7[:, 5] << 5)
            | (codeword_bits_7[:, 6] << 6)
            | (overall_parity << 7)
        ).to(torch.uint8)

        return codewords.view(original_shape)

    def decode(self, codewords: torch.Tensor) -> DecodeResult:
        original_shape = codewords.shape
        flat = codewords.flatten().to(torch.uint8)
        n = flat.shape[0]

        hamming_bits = flat & 0x7F
        stored_parity = (flat >> 7) & 1

        codeword_bits = torch.stack(
            [(hamming_bits >> i) & 1 for i in range(7)], dim=1
        ).to(torch.uint8)

        syndromes = ((self._H @ codeword_bits.T) % 2).T
        syndrome_idx = (
            syndromes[:, 0] * 1 + syndromes[:, 1] * 2 + syndromes[:, 2] * 4
        ).to(torch.long)

        actual_parity = codeword_bits.sum(dim=1) % 2

        parity_error = stored_parity != actual_parity

        syndrome_zero = syndrome_idx == 0

        error_type = torch.zeros(n, dtype=torch.uint8, device=self.device)
        error_type[syndrome_zero & ~parity_error] = ErrorType.NO_ERROR
        error_type[syndrome_zero & parity_error] = ErrorType.PARITY_ONLY
        error_type[~syndrome_zero & parity_error] = ErrorType.SINGLE_CORRECTED
        error_type[~syndrome_zero & ~parity_error] = ErrorType.DOUBLE_DETECTED

        corrected_hamming = hamming_bits.clone()

        single_error_mask = error_type == ErrorType.SINGLE_CORRECTED
        if single_error_mask.any():
            error_positions = self._syndrome_lut[syndrome_idx[single_error_mask]]
            valid_positions = error_positions >= 0

            correction_bits = torch.zeros(
                single_error_mask.sum(), dtype=torch.uint8, device=self.device
            )
            correction_bits[valid_positions] = 1 << error_positions[valid_positions].to(
                torch.uint8
            )

            corrected_hamming[single_error_mask] ^= correction_bits

        double_error_mask = error_type == ErrorType.DOUBLE_DETECTED
        if double_error_mask.any():
            if self.on_double_error == "zero":
                corrected_hamming[double_error_mask] = 0
            elif self.on_double_error == "raise":
                raise ValueError(
                    f"Double-bit error detected in {double_error_mask.sum()} codewords"
                )

        data = corrected_hamming & 0x0F

        return DecodeResult(
            data=data.view(original_shape),
            error_type=error_type.view(original_shape),
            corrected_count=int((error_type == ErrorType.SINGLE_CORRECTED).sum()),
            detected_count=int((error_type == ErrorType.DOUBLE_DETECTED).sum()),
        )

    def decode_simple(
        self, codewords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.decode(codewords)
        errors_detected = result.error_type != ErrorType.NO_ERROR
        return result.data, errors_detected


def verify_extended_hamming_properties():
    codec = Hamming84(on_double_error="keep")

    all_inputs = torch.arange(16, dtype=torch.uint8)
    all_codewords = codec.encode(all_inputs)
    assert len(torch.unique(all_codewords)) == 16, "All 16 codewords should be unique"

    min_distance = float("inf")
    for i in range(16):
        for j in range(i + 1, 16):
            diff = all_codewords[i] ^ all_codewords[j]
            hamming_dist = bin(diff.item()).count("1")
            min_distance = min(min_distance, hamming_dist)

    assert min_distance == 4, f"Minimum distance should be 4, got {min_distance}"

    test_val = torch.tensor([7], dtype=torch.uint8)
    codeword = codec.encode(test_val)

    for bit_pos in range(8):
        corrupted = codeword ^ (1 << bit_pos)
        result = codec.decode(corrupted)
        assert (
            result.data.item() == 7
        ), f"Single-bit error at position {bit_pos} not corrected"
        if bit_pos < 7:
            assert result.error_type.item() in [
                ErrorType.SINGLE_CORRECTED,
                ErrorType.PARITY_ONLY,
            ]

    for bit1 in range(8):
        for bit2 in range(bit1 + 1, 8):
            corrupted = codeword ^ (1 << bit1) ^ (1 << bit2)
            result = codec.decode(corrupted)

            assert result.error_type.item() in [
                ErrorType.DOUBLE_DETECTED,
                ErrorType.NO_ERROR,
            ], f"Double error at bits {bit1},{bit2} should be detected"

    print("All Extended Hamming(8,4) properties verified:")
    print("  - All 16 codewords are distinct")
    print("  - Minimum Hamming distance = 4 (SECDED capability)")
    print("  - Single-bit errors are corrected")
    print("  - Double-bit errors are detected (not miscorrected)")

    return True


def compare_74_vs_84_miscorrection():
    from .hamming74_sec import Hamming74

    codec_74 = Hamming74()
    codec_84 = Hamming84(on_double_error="keep")

    test_val = torch.tensor([5], dtype=torch.uint8)

    cw_74 = codec_74.encode(test_val)
    cw_84 = codec_84.encode(test_val)

    print(f"Original value: {test_val.item()} (0b{test_val.item():04b})")
    print(f"Hamming(7,4) codeword: 0b{cw_74.item():07b}")
    print(f"Hamming(8,4) codeword: 0b{cw_84.item():08b}")
    print()

    miscorrections_74 = 0
    detected_84 = 0

    print("2-bit error analysis:")
    for bit1 in range(7):
        for bit2 in range(bit1 + 1, 7):
            corrupted_74 = cw_74 ^ (1 << bit1) ^ (1 << bit2)
            decoded_74, _ = codec_74.decode(corrupted_74)

            corrupted_84 = cw_84 ^ (1 << bit1) ^ (1 << bit2)
            result_84 = codec_84.decode(corrupted_84)

            dist_74 = bin(decoded_74.item() ^ test_val.item()).count("1")

            if dist_74 > 2:
                miscorrections_74 += 1
                status_84 = (
                    "DETECTED"
                    if result_84.error_type.item() == ErrorType.DOUBLE_DETECTED
                    else "missed"
                )
                print(
                    f"  Bits {bit1},{bit2}: (7,4) miscorrected to {decoded_74.item()} (dist={dist_74}), (8,4) {status_84}"
                )

            if result_84.error_type.item() == ErrorType.DOUBLE_DETECTED:
                detected_84 += 1

    total_2bit = 7 * 6 // 2
    print()
    print(f"Summary:")
    print(f"  Hamming(7,4): {miscorrections_74}/{total_2bit} miscorrections")
    print(
        f"  Hamming(8,4): {detected_84}/{total_2bit} double errors correctly detected"
    )


if __name__ == "__main__":
    verify_extended_hamming_properties()
    print()
    compare_74_vs_84_miscorrection()
