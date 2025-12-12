"""
Hamming(7,4) Encoder/Decoder for INT4 Values.

Mathematical Framework (from proposal):
- Encodes k=4 data bits into n=7 codeword bits
- Single-error correction (SEC) capability
- Uses systematic form: codeword = [data | parity]

Generator Matrix G ∈ F₂^(4×7):
    G = [I₄ | P] where P generates parity bits

Parity Check Matrix H ∈ F₂^(3×7):
    H spans the null space of valid codewords
    GH^T = 0

Syndrome Decoding:
    z = H @ r^T
    - If z = 0: no error (or undetectable multi-bit error)
    - If z ≠ 0: z encodes the error position
"""

import torch
from typing import Tuple


class Hamming74:
    """
    Hamming(7,4) encoder/decoder for INT4 values.

    Each 4-bit value becomes a 7-bit codeword with single-error correction.
    This provides isolated error budgets per value (unlike SECDED which
    shares error budget across 16 values in a 64-bit block).
    """

    # Generator matrix G ∈ F₂^(4×7) in systematic form [I₄ | P]
    # Rows: basis vectors for encoding 4-bit data
    G = torch.tensor([
        [1, 0, 0, 0, 1, 1, 0],  # d₀ → d₀, p₀=d₀⊕d₁⊕d₃, p₁=d₀⊕d₂⊕d₃
        [0, 1, 0, 0, 1, 0, 1],  # d₁ → d₁, contributes to p₀, p₂
        [0, 0, 1, 0, 0, 1, 1],  # d₂ → d₂, contributes to p₁, p₂
        [0, 0, 0, 1, 1, 1, 1],  # d₃ → d₃, contributes to all parity
    ], dtype=torch.uint8)

    # Parity check matrix H ∈ F₂^(3×7)
    # Columns: binary representations of 1-7 (non-zero 3-bit patterns)
    # Used for syndrome decoding
    H = torch.tensor([
        [1, 1, 0, 1, 1, 0, 0],  # Syndrome bit s₀
        [1, 0, 1, 1, 0, 1, 0],  # Syndrome bit s₁
        [0, 1, 1, 1, 0, 0, 1],  # Syndrome bit s₂
    ], dtype=torch.uint8)

    # Syndrome-to-position lookup table
    # syndrome value (0-7) → bit position to flip (-1 means no error)
    # Columns of H read as binary: 1→col0(d₀), 2→col1(d₁), 3→col2(d₂), etc.
    SYNDROME_TO_POSITION = torch.tensor([
        -1,  # 0: no error
         4,  # 1: error in bit 4 (p₀)
         5,  # 2: error in bit 5 (p₁)
         0,  # 3: error in bit 0 (d₀)
         6,  # 4: error in bit 6 (p₂)
         1,  # 5: error in bit 1 (d₁)
         2,  # 6: error in bit 2 (d₂)
         3,  # 7: error in bit 3 (d₃)
    ], dtype=torch.int8)

    def __init__(self, device: str = "cpu"):
        """
        Initialize Hamming(7,4) codec.

        Args:
            device: Target device for tensors ("cpu" or "cuda")
        """
        self.device = device
        self._G = self.G.to(device)
        self._H = self.H.to(device)
        self._syndrome_lut = self.SYNDROME_TO_POSITION.to(device)

    def encode(self, int4_values: torch.Tensor) -> torch.Tensor:
        """
        Encode INT4 values to 7-bit Hamming codewords.

        Args:
            int4_values: Tensor of INT4 values (0-15), any shape

        Returns:
            codewords: Tensor of 7-bit codewords packed as uint8, same shape
                       (Only lower 7 bits are used, MSB is always 0)
        """
        original_shape = int4_values.shape
        flat = int4_values.flatten().to(torch.uint8)

        # Extract 4 data bits: d₀ (LSB) to d₃ (MSB)
        # Shape: (N, 4)
        data_bits = torch.stack([
            (flat >> 0) & 1,
            (flat >> 1) & 1,
            (flat >> 2) & 1,
            (flat >> 3) & 1,
        ], dim=1).to(torch.uint8)

        # Encode: c = d @ G (mod 2)
        # Shape: (N, 4) @ (4, 7) → (N, 7)
        codeword_bits = (data_bits @ self._G) % 2

        # Pack 7 bits into uint8 (bit 7 unused)
        codewords = (
            (codeword_bits[:, 0] << 0) |
            (codeword_bits[:, 1] << 1) |
            (codeword_bits[:, 2] << 2) |
            (codeword_bits[:, 3] << 3) |
            (codeword_bits[:, 4] << 4) |
            (codeword_bits[:, 5] << 5) |
            (codeword_bits[:, 6] << 6)
        )

        return codewords.view(original_shape)

    def decode(self, codewords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode 7-bit Hamming codewords, correcting single-bit errors.

        Args:
            codewords: Tensor of 7-bit codewords (uint8), any shape

        Returns:
            int4_values: Corrected INT4 values (0-15), same shape
            errors_detected: Boolean tensor, True where errors were corrected
        """
        original_shape = codewords.shape
        flat = codewords.flatten().to(torch.uint8)

        # Extract 7 codeword bits
        # Shape: (N, 7)
        codeword_bits = torch.stack([
            (flat >> 0) & 1,
            (flat >> 1) & 1,
            (flat >> 2) & 1,
            (flat >> 3) & 1,
            (flat >> 4) & 1,
            (flat >> 5) & 1,
            (flat >> 6) & 1,
        ], dim=1).to(torch.uint8)

        # Compute syndrome: z = H @ r^T (mod 2)
        # Shape: (3, 7) @ (N, 7)^T → (3, N) → (N, 3)
        syndromes = ((self._H @ codeword_bits.T) % 2).T

        # Convert syndrome bits to integer index (0-7)
        syndrome_idx = (
            syndromes[:, 0] * 1 +
            syndromes[:, 1] * 2 +
            syndromes[:, 2] * 4
        ).to(torch.long)

        # Look up error position (-1 if no error)
        error_positions = self._syndrome_lut[syndrome_idx]

        # Track where errors were detected/corrected
        errors_detected = (error_positions >= 0)

        # Correct errors by flipping the indicated bit
        # Create correction mask: 1 at error position, 0 elsewhere
        correction_mask = torch.zeros_like(flat, dtype=torch.uint8)
        has_error = errors_detected
        correction_mask[has_error] = (1 << error_positions[has_error].to(torch.uint8))

        # Apply correction via XOR
        corrected = flat ^ correction_mask

        # Extract data bits (first 4 bits in systematic form)
        int4_values = corrected & 0x0F

        return int4_values.view(original_shape), errors_detected.view(original_shape)

    def encode_batch(self, int4_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a tensor of INT4 values.
        Alias for encode() with clearer semantics for batch operations.
        """
        return self.encode(int4_tensor)

    def decode_batch(self, codeword_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode a tensor of codewords.
        Alias for decode() with clearer semantics for batch operations.
        """
        return self.decode(codeword_tensor)


def verify_hamming_properties():
    """
    Verify mathematical properties of the Hamming(7,4) code.

    Checks:
    1. G @ H^T = 0 (null space property)
    2. All 16 codewords are distinct
    3. Minimum Hamming distance = 3
    """
    G = Hamming74.G.float()
    H = Hamming74.H.float()

    # Check 1: Null space property
    product = (G @ H.T) % 2
    assert (product == 0).all(), "G @ H^T should be zero matrix"

    # Check 2: All codewords distinct
    codec = Hamming74()
    all_inputs = torch.arange(16, dtype=torch.uint8)
    all_codewords = codec.encode(all_inputs)
    assert len(torch.unique(all_codewords)) == 16, "All 16 codewords should be unique"

    # Check 3: Minimum Hamming distance
    min_distance = float('inf')
    for i in range(16):
        for j in range(i + 1, 16):
            # Count differing bits
            diff = all_codewords[i] ^ all_codewords[j]
            hamming_dist = bin(diff.item()).count('1')
            min_distance = min(min_distance, hamming_dist)

    assert min_distance == 3, f"Minimum distance should be 3, got {min_distance}"

    print("All Hamming(7,4) properties verified:")
    print("  - G @ H^T = 0 (null space property)")
    print("  - All 16 codewords are distinct")
    print("  - Minimum Hamming distance = 3 (single-error correction)")

    return True


if __name__ == "__main__":
    # Verify properties
    verify_hamming_properties()

    # Demo encoding/decoding
    codec = Hamming74()

    print("\nEncoding demo:")
    for val in [0, 5, 10, 15]:
        codeword = codec.encode(torch.tensor([val], dtype=torch.uint8))
        decoded, _ = codec.decode(codeword)
        print(f"  {val:2d} (0b{val:04b}) → 0b{codeword.item():07b} → {decoded.item()}")

    print("\nError correction demo:")
    original = torch.tensor([7], dtype=torch.uint8)  # 0b0111
    codeword = codec.encode(original)
    print(f"  Original: {original.item()} → codeword: 0b{codeword.item():07b}")

    # Inject single-bit errors at each position
    for bit_pos in range(7):
        corrupted = codeword ^ (1 << bit_pos)
        recovered, error_detected = codec.decode(corrupted)
        status = "corrected" if error_detected.item() else "clean"
        print(f"  Flip bit {bit_pos}: 0b{corrupted.item():07b} → {recovered.item()} ({status})")
