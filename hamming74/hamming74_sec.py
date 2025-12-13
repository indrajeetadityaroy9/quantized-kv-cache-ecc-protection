import torch
from typing import Tuple


class Hamming74:
    G = torch.tensor(
        [
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.uint8,
    )

    H = torch.tensor(
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

    def __init__(self, device="cuda"):
        self.device = device
        self._G = self.G.to(device)
        self._H = self.H.to(device)
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

        codeword_bits = (data_bits @ self._G) % 2

        codewords = (
            (codeword_bits[:, 0] << 0)
            | (codeword_bits[:, 1] << 1)
            | (codeword_bits[:, 2] << 2)
            | (codeword_bits[:, 3] << 3)
            | (codeword_bits[:, 4] << 4)
            | (codeword_bits[:, 5] << 5)
            | (codeword_bits[:, 6] << 6)
        )

        return codewords.view(original_shape)

    def decode(self, codewords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = codewords.shape
        flat = codewords.flatten().to(torch.uint8)

        codeword_bits = torch.stack(
            [
                (flat >> 0) & 1,
                (flat >> 1) & 1,
                (flat >> 2) & 1,
                (flat >> 3) & 1,
                (flat >> 4) & 1,
                (flat >> 5) & 1,
                (flat >> 6) & 1,
            ],
            dim=1,
        ).to(torch.uint8)

        syndromes = ((self._H @ codeword_bits.T) % 2).T

        syndrome_idx = (
            syndromes[:, 0] * 1 + syndromes[:, 1] * 2 + syndromes[:, 2] * 4
        ).to(torch.long)

        error_positions = self._syndrome_lut[syndrome_idx]

        errors_detected = error_positions >= 0

        correction_mask = torch.zeros_like(flat, dtype=torch.uint8)
        has_error = errors_detected
        correction_mask[has_error] = 1 << error_positions[has_error].to(torch.uint8)

        corrected = flat ^ correction_mask

        int4_values = corrected & 0x0F

        return int4_values.view(original_shape), errors_detected.view(original_shape)

    def encode_batch(self, int4_tensor: torch.Tensor) -> torch.Tensor:
        return self.encode(int4_tensor)

    def decode_batch(
        self, codeword_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(codeword_tensor)


def verify_hamming_properties():
    G = Hamming74.G.float()
    H = Hamming74.H.float()

    product = (G @ H.T) % 2
    assert (product == 0).all(), "G @ H^T should be zero matrix"

    codec = Hamming74()
    all_inputs = torch.arange(16, dtype=torch.uint8)
    all_codewords = codec.encode(all_inputs)
    assert len(torch.unique(all_codewords)) == 16, "All 16 codewords should be unique"

    min_distance = float("inf")
    for i in range(16):
        for j in range(i + 1, 16):
            diff = all_codewords[i] ^ all_codewords[j]
            hamming_dist = bin(diff.item()).count("1")
            min_distance = min(min_distance, hamming_dist)

    assert min_distance == 3, f"Minimum distance should be 3, got {min_distance}"

    print("All Hamming(7,4) properties verified:")
    print("  - G @ H^T = 0 (null space property)")
    print("  - All 16 codewords are distinct")
    print("  - Minimum Hamming distance = 3 (single-error correction)")

    return True


if __name__ == "__main__":
    verify_hamming_properties()

    codec = Hamming74()

    print("\nEncoding demo:")
    for val in [0, 5, 10, 15]:
        codeword = codec.encode(torch.tensor([val], dtype=torch.uint8))
        decoded, _ = codec.decode(codeword)
        print(f"  {val:2d} (0b{val:04b}) → 0b{codeword.item():07b} → {decoded.item()}")

    print("\nError correction demo:")
    original = torch.tensor([7], dtype=torch.uint8)
    codeword = codec.encode(original)
    print(f"  Original: {original.item()} → codeword: 0b{codeword.item():07b}")

    for bit_pos in range(7):
        corrupted = codeword ^ (1 << bit_pos)
        recovered, error_detected = codec.decode(corrupted)
        status = "corrected" if error_detected.item() else "clean"
        print(
            f"  Flip bit {bit_pos}: 0b{corrupted.item():07b} → {recovered.item()} ({status})"
        )
