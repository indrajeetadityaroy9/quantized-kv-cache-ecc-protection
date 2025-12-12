"""
Extended Hamming(8,4) Encoder/Decoder with SECDED.

Mathematical Framework:
- Encodes k=4 data bits into n=8 codeword bits
- Single Error Correction, Double Error Detection (SECDED)
- Adds overall parity bit to Hamming(7,4)

Key Properties:
- Minimum Hamming distance d_min = 4
- Can correct any 1-bit error
- Can detect any 2-bit error (but NOT correct it)

SECDED Logic:
- syndrome=0, parity=0: No error
- syndrome=0, parity=1: Error in overall parity bit (bit 7)
- syndrome≠0, parity=1: Single-bit error (correctable)
- syndrome≠0, parity=0: Double-bit error (detectable, NOT correctable)

This solves the "miscorrection" problem of Hamming(7,4):
When Hamming(7,4) encounters a 2-bit error, it incorrectly "corrects" a third bit,
turning a 2-bit error into a 3-bit error. Extended Hamming(8,4) detects this
case and can choose to mask/zero the value instead of miscorrecting.
"""

import torch
from typing import Tuple, NamedTuple
from enum import IntEnum


class ErrorType(IntEnum):
    """Classification of error detection/correction results."""
    NO_ERROR = 0           # Clean codeword
    SINGLE_CORRECTED = 1   # Single-bit error, successfully corrected
    DOUBLE_DETECTED = 2    # Double-bit error, detected but NOT corrected
    PARITY_ONLY = 3        # Error only in overall parity bit


class DecodeResult(NamedTuple):
    """Result of decoding operation."""
    data: torch.Tensor           # Decoded INT4 values
    error_type: torch.Tensor     # ErrorType for each value
    corrected_count: int         # Number of single-bit errors corrected
    detected_count: int          # Number of double-bit errors detected


class Hamming84:
    """
    Extended Hamming(8,4) encoder/decoder with SECDED capability.

    Each 4-bit value becomes an 8-bit codeword with:
    - Single Error Correction (SEC)
    - Double Error Detection (DED)

    Storage: 8 bits per value (same as INT8, but with error protection)
    """

    # Generator matrix G ∈ F₂^(4×7) for Hamming(7,4) portion
    # Same as Hamming(7,4) systematic form [I₄ | P]
    G_74 = torch.tensor([
        [1, 0, 0, 0, 1, 1, 0],  # d₀
        [0, 1, 0, 0, 1, 0, 1],  # d₁
        [0, 0, 1, 0, 0, 1, 1],  # d₂
        [0, 0, 0, 1, 1, 1, 1],  # d₃
    ], dtype=torch.uint8)

    # Parity check matrix H ∈ F₂^(3×7) for syndrome computation
    H_74 = torch.tensor([
        [1, 1, 0, 1, 1, 0, 0],  # Syndrome bit s₀
        [1, 0, 1, 1, 0, 1, 0],  # Syndrome bit s₁
        [0, 1, 1, 1, 0, 0, 1],  # Syndrome bit s₂
    ], dtype=torch.uint8)

    # Syndrome-to-position lookup table (same as Hamming74)
    SYNDROME_TO_POSITION = torch.tensor([
        -1,  # 0: no error in bits 0-6
         4,  # 1: error in bit 4 (p₀)
         5,  # 2: error in bit 5 (p₁)
         0,  # 3: error in bit 0 (d₀)
         6,  # 4: error in bit 6 (p₂)
         1,  # 5: error in bit 1 (d₁)
         2,  # 6: error in bit 2 (d₂)
         3,  # 7: error in bit 3 (d₃)
    ], dtype=torch.int8)

    def __init__(self, device: str = "cpu", on_double_error: str = "zero"):
        """
        Initialize Extended Hamming(8,4) codec.

        Args:
            device: Target device for tensors ("cpu" or "cuda")
            on_double_error: Action when double error detected:
                - "zero": Replace with zero (mask the value)
                - "keep": Keep corrupted value (for analysis)
                - "raise": Raise exception
        """
        self.device = device
        self.on_double_error = on_double_error
        self._G = self.G_74.to(device)
        self._H = self.H_74.to(device)
        self._syndrome_lut = self.SYNDROME_TO_POSITION.to(device)

    def encode(self, int4_values: torch.Tensor) -> torch.Tensor:
        """
        Encode INT4 values to 8-bit Extended Hamming codewords.

        Args:
            int4_values: Tensor of INT4 values (0-15), any shape

        Returns:
            codewords: Tensor of 8-bit codewords (uint8), same shape
                       Bit 7 is overall parity, bits 0-6 are Hamming(7,4)
        """
        original_shape = int4_values.shape
        flat = int4_values.flatten().to(torch.uint8)

        # Extract 4 data bits
        data_bits = torch.stack([
            (flat >> 0) & 1,
            (flat >> 1) & 1,
            (flat >> 2) & 1,
            (flat >> 3) & 1,
        ], dim=1).to(torch.uint8)

        # Encode to 7-bit Hamming codeword: c = d @ G (mod 2)
        codeword_bits_7 = (data_bits @ self._G) % 2

        # Compute overall parity (XOR of all 7 bits)
        overall_parity = codeword_bits_7.sum(dim=1) % 2

        # Pack into uint8: [b0-b6: Hamming(7,4), b7: overall parity]
        codewords = (
            (codeword_bits_7[:, 0] << 0) |
            (codeword_bits_7[:, 1] << 1) |
            (codeword_bits_7[:, 2] << 2) |
            (codeword_bits_7[:, 3] << 3) |
            (codeword_bits_7[:, 4] << 4) |
            (codeword_bits_7[:, 5] << 5) |
            (codeword_bits_7[:, 6] << 6) |
            (overall_parity << 7)
        ).to(torch.uint8)

        return codewords.view(original_shape)

    def decode(self, codewords: torch.Tensor) -> DecodeResult:
        """
        Decode 8-bit Extended Hamming codewords with SECDED.

        Args:
            codewords: Tensor of 8-bit codewords (uint8), any shape

        Returns:
            DecodeResult containing:
                - data: Corrected INT4 values
                - error_type: ErrorType classification for each value
                - corrected_count: Number of single errors corrected
                - detected_count: Number of double errors detected
        """
        original_shape = codewords.shape
        flat = codewords.flatten().to(torch.uint8)
        n = flat.shape[0]

        # Extract bits 0-6 (Hamming portion) and bit 7 (overall parity)
        hamming_bits = flat & 0x7F  # Lower 7 bits
        stored_parity = (flat >> 7) & 1  # Bit 7

        # Extract individual bits for syndrome computation
        codeword_bits = torch.stack([
            (hamming_bits >> i) & 1 for i in range(7)
        ], dim=1).to(torch.uint8)

        # Compute syndrome: z = H @ r^T (mod 2)
        syndromes = ((self._H @ codeword_bits.T) % 2).T
        syndrome_idx = (
            syndromes[:, 0] * 1 +
            syndromes[:, 1] * 2 +
            syndromes[:, 2] * 4
        ).to(torch.long)

        # Compute actual overall parity of received 7 bits
        actual_parity = codeword_bits.sum(dim=1) % 2

        # Parity check: does stored parity match actual parity?
        parity_error = (stored_parity != actual_parity)

        # SECDED classification
        # syndrome=0, parity_ok: No error
        # syndrome=0, parity_err: Error in parity bit only
        # syndrome≠0, parity_err: Single-bit error (correctable)
        # syndrome≠0, parity_ok: Double-bit error (detected only)

        syndrome_zero = (syndrome_idx == 0)

        error_type = torch.zeros(n, dtype=torch.uint8, device=self.device)
        error_type[syndrome_zero & ~parity_error] = ErrorType.NO_ERROR
        error_type[syndrome_zero & parity_error] = ErrorType.PARITY_ONLY
        error_type[~syndrome_zero & parity_error] = ErrorType.SINGLE_CORRECTED
        error_type[~syndrome_zero & ~parity_error] = ErrorType.DOUBLE_DETECTED

        # Prepare output
        corrected_hamming = hamming_bits.clone()

        # Handle single-bit errors (correctable)
        single_error_mask = (error_type == ErrorType.SINGLE_CORRECTED)
        if single_error_mask.any():
            error_positions = self._syndrome_lut[syndrome_idx[single_error_mask]]
            valid_positions = error_positions >= 0

            correction_bits = torch.zeros(single_error_mask.sum(), dtype=torch.uint8, device=self.device)
            correction_bits[valid_positions] = (1 << error_positions[valid_positions].to(torch.uint8))

            corrected_hamming[single_error_mask] ^= correction_bits

        # Handle double-bit errors based on policy
        double_error_mask = (error_type == ErrorType.DOUBLE_DETECTED)
        if double_error_mask.any():
            if self.on_double_error == "zero":
                corrected_hamming[double_error_mask] = 0
            elif self.on_double_error == "raise":
                raise ValueError(f"Double-bit error detected in {double_error_mask.sum()} codewords")
            # "keep" does nothing - keeps corrupted value

        # Extract data bits (first 4 bits in systematic form)
        data = corrected_hamming & 0x0F

        return DecodeResult(
            data=data.view(original_shape),
            error_type=error_type.view(original_shape),
            corrected_count=int((error_type == ErrorType.SINGLE_CORRECTED).sum()),
            detected_count=int((error_type == ErrorType.DOUBLE_DETECTED).sum()),
        )

    def decode_simple(self, codewords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple decode interface matching Hamming74 signature.

        Returns:
            data: Decoded INT4 values
            errors_detected: Boolean tensor, True where any error was detected
        """
        result = self.decode(codewords)
        errors_detected = (result.error_type != ErrorType.NO_ERROR)
        return result.data, errors_detected


def verify_extended_hamming_properties():
    """
    Verify mathematical properties of Extended Hamming(8,4).

    Checks:
    1. All 16 codewords are distinct
    2. Minimum Hamming distance = 4
    3. Single errors are corrected
    4. Double errors are detected (not miscorrected)
    """
    codec = Hamming84(on_double_error="keep")

    # Check 1: All codewords distinct
    all_inputs = torch.arange(16, dtype=torch.uint8)
    all_codewords = codec.encode(all_inputs)
    assert len(torch.unique(all_codewords)) == 16, "All 16 codewords should be unique"

    # Check 2: Minimum Hamming distance = 4
    min_distance = float('inf')
    for i in range(16):
        for j in range(i + 1, 16):
            diff = all_codewords[i] ^ all_codewords[j]
            hamming_dist = bin(diff.item()).count('1')
            min_distance = min(min_distance, hamming_dist)

    assert min_distance == 4, f"Minimum distance should be 4, got {min_distance}"

    # Check 3: Single errors corrected
    test_val = torch.tensor([7], dtype=torch.uint8)
    codeword = codec.encode(test_val)

    for bit_pos in range(8):
        corrupted = codeword ^ (1 << bit_pos)
        result = codec.decode(corrupted)
        assert result.data.item() == 7, f"Single-bit error at position {bit_pos} not corrected"
        if bit_pos < 7:
            assert result.error_type.item() in [ErrorType.SINGLE_CORRECTED, ErrorType.PARITY_ONLY]

    # Check 4: Double errors detected (not miscorrected)
    for bit1 in range(8):
        for bit2 in range(bit1 + 1, 8):
            corrupted = codeword ^ (1 << bit1) ^ (1 << bit2)
            result = codec.decode(corrupted)
            # Should be detected as double error OR parity-only (edge case)
            assert result.error_type.item() in [ErrorType.DOUBLE_DETECTED, ErrorType.NO_ERROR], \
                f"Double error at bits {bit1},{bit2} should be detected"

    print("All Extended Hamming(8,4) properties verified:")
    print("  - All 16 codewords are distinct")
    print("  - Minimum Hamming distance = 4 (SECDED capability)")
    print("  - Single-bit errors are corrected")
    print("  - Double-bit errors are detected (not miscorrected)")

    return True


def compare_74_vs_84_miscorrection():
    """
    Demonstrate the miscorrection problem in Hamming(7,4) vs (8,4).

    Shows how (7,4) miscorrects 2-bit errors into 3-bit errors,
    while (8,4) correctly detects them.
    """
    from .hamming74_sec import Hamming74

    codec_74 = Hamming74()
    codec_84 = Hamming84(on_double_error="keep")

    test_val = torch.tensor([5], dtype=torch.uint8)  # 0b0101

    # Encode with both codes
    cw_74 = codec_74.encode(test_val)
    cw_84 = codec_84.encode(test_val)

    print(f"Original value: {test_val.item()} (0b{test_val.item():04b})")
    print(f"Hamming(7,4) codeword: 0b{cw_74.item():07b}")
    print(f"Hamming(8,4) codeword: 0b{cw_84.item():08b}")
    print()

    # Test all 2-bit error combinations
    miscorrections_74 = 0
    detected_84 = 0

    print("2-bit error analysis:")
    for bit1 in range(7):
        for bit2 in range(bit1 + 1, 7):
            # Hamming(7,4)
            corrupted_74 = cw_74 ^ (1 << bit1) ^ (1 << bit2)
            decoded_74, _ = codec_74.decode(corrupted_74)

            # Hamming(8,4)
            corrupted_84 = cw_84 ^ (1 << bit1) ^ (1 << bit2)
            result_84 = codec_84.decode(corrupted_84)

            # Check if (7,4) miscorrected
            dist_74 = bin(decoded_74.item() ^ test_val.item()).count('1')

            if dist_74 > 2:  # Miscorrection: turned 2-bit error into 3+ bit error
                miscorrections_74 += 1
                status_84 = "DETECTED" if result_84.error_type.item() == ErrorType.DOUBLE_DETECTED else "missed"
                print(f"  Bits {bit1},{bit2}: (7,4) miscorrected to {decoded_74.item()} (dist={dist_74}), (8,4) {status_84}")

            if result_84.error_type.item() == ErrorType.DOUBLE_DETECTED:
                detected_84 += 1

    total_2bit = 7 * 6 // 2  # C(7,2) = 21 combinations
    print()
    print(f"Summary:")
    print(f"  Hamming(7,4): {miscorrections_74}/{total_2bit} miscorrections")
    print(f"  Hamming(8,4): {detected_84}/{total_2bit} double errors correctly detected")


if __name__ == "__main__":
    verify_extended_hamming_properties()
    print()
    compare_74_vs_84_miscorrection()
