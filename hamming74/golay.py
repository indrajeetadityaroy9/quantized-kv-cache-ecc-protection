import torch
from typing import Tuple, NamedTuple


class GolayDecodeResult(NamedTuple):
    data: torch.Tensor
    errors_corrected: int
    uncorrectable_count: int


class Golay2412:
    UNCORRECTABLE = 0xFFFFFFFF

    def __init__(self, device="cuda"):
        self.device = device

        self.G, self.H, self.P = self._build_matrices()

        self._precompute_syndrome_masks()

        self.syndrome_table, self.error_weights = self._build_syndrome_table()

    def _build_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = torch.tensor(
            [
                [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.uint8,
            device=self.device,
        )

        I_12 = torch.eye(12, dtype=torch.uint8, device=self.device)
        G = torch.cat([I_12, B], dim=1)

        H = torch.cat([B.T, I_12], dim=1)

        return G, H, B

    def _precompute_syndrome_masks(self):
        self.h_row_masks = torch.zeros(12, dtype=torch.int64, device=self.device)
        for i in range(12):
            mask = 0
            for j in range(24):
                if self.H[i, j].item() == 1:
                    mask |= 1 << j
            self.h_row_masks[i] = mask

    def _build_syndrome_table(self) -> Tuple[torch.Tensor, torch.Tensor]:
        table = torch.full(
            (4096,), self.UNCORRECTABLE, dtype=torch.int64, device=self.device
        )
        weights = torch.zeros(4096, dtype=torch.int64, device=self.device)

        table[0] = 0
        weights[0] = 0

        for i in range(24):
            error = 1 << i
            syndrome = self._compute_syndrome_scalar(error)
            table[syndrome] = error
            weights[syndrome] = 1

        for i in range(24):
            for j in range(i + 1, 24):
                error = (1 << i) | (1 << j)
                syndrome = self._compute_syndrome_scalar(error)
                if table[syndrome] == self.UNCORRECTABLE:
                    table[syndrome] = error
                    weights[syndrome] = 2

        for i in range(24):
            for j in range(i + 1, 24):
                for k in range(j + 1, 24):
                    error = (1 << i) | (1 << j) | (1 << k)
                    syndrome = self._compute_syndrome_scalar(error)
                    if table[syndrome] == self.UNCORRECTABLE:
                        table[syndrome] = error
                        weights[syndrome] = 3

        return table, weights

    def _compute_syndrome_scalar(self, codeword: int) -> int:
        syndrome = 0
        for i in range(12):
            masked = codeword & self.h_row_masks[i].item()
            parity = bin(masked).count("1") & 1
            syndrome |= parity << i
        return syndrome

    def _compute_syndrome_vectorized(self, codewords: torch.Tensor) -> torch.Tensor:
        N = codewords.shape[0]
        syndromes = torch.zeros(N, dtype=torch.int64, device=self.device)

        for i in range(12):
            mask = self.h_row_masks[i].item()

            masked = codewords & mask

            parity = masked
            for shift in [1, 2, 4, 8, 16]:
                parity = parity ^ (parity >> shift)
            parity = parity & 1

            syndromes = syndromes | (parity << i)

        return syndromes

    def encode(self, triplets: torch.Tensor) -> torch.Tensor:
        if triplets.dim() == 1:
            triplets = triplets.unsqueeze(0)

        N = triplets.shape[0]
        triplets = triplets.to(self.device)

        data_12bit = (
            (triplets[:, 0].to(torch.int64) & 0xF)
            | ((triplets[:, 1].to(torch.int64) & 0xF) << 4)
            | ((triplets[:, 2].to(torch.int64) & 0xF) << 8)
        )

        data_bits = torch.zeros((N, 12), dtype=torch.float32, device=self.device)
        for i in range(12):
            data_bits[:, i] = ((data_12bit >> i) & 1).float()

        codeword_bits = (data_bits @ self.G.float()) % 2

        codewords = torch.zeros(N, dtype=torch.int64, device=self.device)
        for i in range(24):
            codewords = codewords | (codeword_bits[:, i].to(torch.int64) << i)

        return codewords

    def decode(self, codewords: torch.Tensor) -> GolayDecodeResult:
        if codewords.dim() == 0:
            codewords = codewords.unsqueeze(0)

        N = codewords.shape[0]
        codewords = codewords.to(torch.int64).to(self.device)

        syndromes = self._compute_syndrome_vectorized(codewords)

        error_patterns = self.syndrome_table[syndromes]
        error_weights = self.error_weights[syndromes]

        correctable_mask = error_patterns != self.UNCORRECTABLE

        corrected_codewords = codewords.clone()
        corrected_codewords[correctable_mask] = (
            corrected_codewords[correctable_mask] ^ error_patterns[correctable_mask]
        )

        data_12bit = corrected_codewords & 0xFFF

        corrected_data = torch.zeros((N, 3), dtype=torch.uint8, device=self.device)
        corrected_data[:, 0] = (data_12bit & 0xF).to(torch.uint8)
        corrected_data[:, 1] = ((data_12bit >> 4) & 0xF).to(torch.uint8)
        corrected_data[:, 2] = ((data_12bit >> 8) & 0xF).to(torch.uint8)

        errors_corrected = error_weights[correctable_mask].sum().item()
        uncorrectable_count = (~correctable_mask).sum().item()

        return GolayDecodeResult(
            data=corrected_data,
            errors_corrected=errors_corrected,
            uncorrectable_count=uncorrectable_count,
        )

    def verify_properties(self) -> bool:
        product = (self.G.float() @ self.H.T.float()) % 2
        if product.sum() != 0:
            print("FAIL: G @ H^T != 0")
            return False

        min_dist = float("inf")
        for i in range(1, 4096, 16):
            triplet = torch.tensor(
                [[i & 0xF, (i >> 4) & 0xF, (i >> 8) & 0xF]], dtype=torch.uint8
            )
            cw = self.encode(triplet)
            weight = bin(cw[0].item()).count("1")
            if weight > 0:
                min_dist = min(min_dist, weight)

        if min_dist != 8:
            print(f"FAIL: Minimum distance is {min_dist}, expected 8")
            return False

        triplet = torch.tensor([[5, 10, 3]], dtype=torch.uint8)
        cw = self.encode(triplet)

        errors = torch.tensor(
            [1 << i for i in range(24)], dtype=torch.int64, device=self.device
        )
        corrupted = cw.expand(24) ^ errors
        result = self.decode(corrupted)

        if not torch.all(result.data == triplet.expand(24, -1)):
            print("FAIL: Not all single-bit errors were corrected")
            return False

        print("All Golay(24,12) properties verified!")
        return True


def verify_golay_properties() -> bool:
    codec = Golay2412()
    return codec.verify_properties()


if __name__ == "__main__":
    print("Golay(24,12) Code Verification - Vectorized Implementation")
    print("=" * 60)

    codec = Golay2412()

    triplets = torch.tensor(
        [
            [5, 10, 3],
            [0, 0, 0],
            [15, 15, 15],
            [7, 8, 9],
        ],
        dtype=torch.uint8,
    )

    print(f"Input triplets:\n{triplets}")

    codewords = codec.encode(triplets)
    print(f"\nEncoded codewords: {codewords.tolist()}")
    print(f"Codeword weights: {[bin(cw.item()).count('1') for cw in codewords]}")

    result = codec.decode(codewords)
    print(f"\nDecoded triplets:\n{result.data}")
    print(f"Errors corrected: {result.errors_corrected}")

    print("\n" + "=" * 60)
    print("Testing error correction (vectorized)...")

    for n_errors in [1, 2, 3]:
        cw = codewords[0:1].clone()
        error = sum(1 << i for i in range(n_errors))
        cw_corrupted = cw ^ error
        result = codec.decode(cw_corrupted)
        success = torch.equal(result.data[0], triplets[0])
        print(f"  {n_errors}-bit error: {'CORRECTED' if success else 'FAILED'}")

    print("\n" + "=" * 60)
    print("Speed test (100,000 codewords)...")
    import time

    large_triplets = torch.randint(0, 16, (100000, 3), dtype=torch.uint8)
    large_cw = codec.encode(large_triplets)

    noise = torch.rand(100000) < 0.05
    large_cw_noisy = large_cw.clone()
    large_cw_noisy[noise] = large_cw_noisy[noise] ^ 1

    start = time.time()
    result = codec.decode(large_cw_noisy)
    elapsed = time.time() - start
    print(f"  Decoded 100,000 codewords in {elapsed*1000:.1f}ms")
    print(f"  Errors corrected: {result.errors_corrected}")

    print("\n" + "=" * 60)
    codec.verify_properties()
