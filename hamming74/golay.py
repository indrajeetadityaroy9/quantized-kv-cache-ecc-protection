"""
Extended Binary Golay Code (24, 12, 8) Implementation.

The Golay code bundles 3 INT4 values (12 bits) into a 24-bit codeword and can
correct up to 3 bit errors anywhere in the codeword. This provides stronger
protection than Hamming(8,4) at extreme bit error rates.

Mathematical Properties:
- Generator Matrix G: 12×24 in systematic form [I₁₂ | P]
- Parity Check Matrix H: 12×24 in form [P^T | I₁₂]
- Minimum Distance: d_min = 8 (can correct 3 errors, detect 7)
- Code Rate: 0.5 (same as Hamming(8,4))
- Syndrome Table: 2^12 = 4096 entries

IMPORTANT: This implementation uses FULLY VECTORIZED operations for speed.
No Python loops over codewords - all operations use tensor broadcasting.

References:
- Golay, M.J.E. (1949). Notes on digital coding. Proc. IRE, 37.
- Lin, S. & Costello, D.J. (2004). Error Control Coding. Ch. 4.
"""

import torch
from typing import Tuple, NamedTuple


class GolayDecodeResult(NamedTuple):
    """Result of Golay decoding operation."""
    data: torch.Tensor          # Decoded 12-bit values as (N, 3) INT4 triplets
    errors_corrected: int       # Number of errors corrected (up to 3 per codeword)
    uncorrectable_count: int    # Number of codewords with >3 errors (uncorrectable)


class Golay2412:
    """
    Extended Binary Golay Code (24, 12, 8) - Fully Vectorized.

    Bundles 3 INT4 values (12 bits) → 24-bit codeword.
    Corrects up to 3 bit errors per codeword.

    All encode/decode operations are FULLY VECTORIZED using:
    - Pre-computed syndrome lookup table (4096 entries)
    - Bitwise tensor operations for syndrome computation
    - Tensor indexing for error pattern lookup

    Usage:
        codec = Golay2412()

        # Encode triplets of INT4 values
        triplets = torch.tensor([[5, 10, 3], [7, 2, 15]], dtype=torch.uint8)
        codewords = codec.encode(triplets)  # Shape: (2,), dtype: int64

        # Decode (with error correction) - FAST vectorized
        result = codec.decode(codewords)
        # result.data: recovered triplets
        # result.errors_corrected: total bit flips corrected
    """

    # Marker for uncorrectable errors (>3 bit flips)
    UNCORRECTABLE = 0xFFFFFFFF

    def __init__(self, device: str = "cpu"):
        """
        Initialize Golay(24,12) codec.

        Args:
            device: Device to store matrices on ("cpu" recommended for uint8 ops)
        """
        self.device = device

        # Build generator and parity check matrices
        self.G, self.H, self.P = self._build_matrices()

        # Pre-compute bit masks for vectorized syndrome computation
        # MUST be called before _build_syndrome_table() since it uses h_row_masks
        self._precompute_syndrome_masks()

        # Build syndrome lookup table for VECTORIZED decoding
        self.syndrome_table, self.error_weights = self._build_syndrome_table()

    def _build_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Generator and Parity Check matrices using the standard Golay construction.

        Returns:
            G: 12×24 Generator matrix [I₁₂ | B]
            H: 12×24 Parity check matrix [B^T | I₁₂]
            B: 12×12 parity sub-matrix
        """
        # Standard Golay B matrix (12x12)
        # Source: MacWilliams & Sloane, "The Theory of Error-Correcting Codes"
        B = torch.tensor([
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
        ], dtype=torch.uint8, device=self.device)

        # Generator matrix: [I₁₂ | B]
        I_12 = torch.eye(12, dtype=torch.uint8, device=self.device)
        G = torch.cat([I_12, B], dim=1)  # Shape: (12, 24)

        # Parity check matrix: [B^T | I₁₂]
        H = torch.cat([B.T, I_12], dim=1)  # Shape: (12, 24)

        return G, H, B

    def _precompute_syndrome_masks(self):
        """Pre-compute masks for vectorized syndrome computation."""
        # For each syndrome bit position (0-11), store which codeword bits contribute
        # H is 12x24, so syndrome[i] = XOR of codeword bits where H[i,j] = 1
        self.h_row_masks = torch.zeros(12, dtype=torch.int64, device=self.device)
        for i in range(12):
            mask = 0
            for j in range(24):
                if self.H[i, j].item() == 1:
                    mask |= (1 << j)
            self.h_row_masks[i] = mask

    def _build_syndrome_table(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build syndrome lookup table for VECTORIZED error correction.

        Returns:
            syndrome_table: Tensor of shape (4096,) containing 24-bit error patterns
            error_weights: Tensor of shape (4096,) containing number of bit errors
        """
        # Initialize tables
        table = torch.full((4096,), self.UNCORRECTABLE, dtype=torch.int64, device=self.device)
        weights = torch.zeros(4096, dtype=torch.int64, device=self.device)

        # Zero syndrome = no error
        table[0] = 0
        weights[0] = 0

        # Weight-1 errors: single bit flips (24 patterns)
        for i in range(24):
            error = 1 << i
            syndrome = self._compute_syndrome_scalar(error)
            table[syndrome] = error
            weights[syndrome] = 1

        # Weight-2 errors: two bit flips (276 patterns)
        for i in range(24):
            for j in range(i + 1, 24):
                error = (1 << i) | (1 << j)
                syndrome = self._compute_syndrome_scalar(error)
                if table[syndrome] == self.UNCORRECTABLE:
                    table[syndrome] = error
                    weights[syndrome] = 2

        # Weight-3 errors: three bit flips (2024 patterns)
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
        """Compute syndrome for a single codeword (used during table building)."""
        syndrome = 0
        for i in range(12):
            # Popcount of (codeword AND h_row_mask) mod 2
            masked = codeword & self.h_row_masks[i].item()
            parity = bin(masked).count('1') & 1
            syndrome |= (parity << i)
        return syndrome

    def _compute_syndrome_vectorized(self, codewords: torch.Tensor) -> torch.Tensor:
        """
        Compute syndromes for a batch of codewords - FULLY VECTORIZED.

        Args:
            codewords: Tensor of shape (N,) with 24-bit codewords

        Returns:
            syndromes: Tensor of shape (N,) with 12-bit syndromes
        """
        N = codewords.shape[0]
        syndromes = torch.zeros(N, dtype=torch.int64, device=self.device)

        # For each syndrome bit, compute parity of masked codeword bits
        for i in range(12):
            mask = self.h_row_masks[i].item()
            # Vectorized: AND with mask, then count 1s mod 2
            masked = codewords & mask

            # Compute popcount mod 2 for each element
            # Use bit manipulation: fold the bits
            parity = masked
            for shift in [1, 2, 4, 8, 16]:
                parity = parity ^ (parity >> shift)
            parity = parity & 1

            syndromes = syndromes | (parity << i)

        return syndromes

    def encode(self, triplets: torch.Tensor) -> torch.Tensor:
        """
        Encode INT4 triplets to 24-bit Golay codewords - VECTORIZED.

        Args:
            triplets: Tensor of shape (N, 3) with INT4 values (0-15) per position

        Returns:
            codewords: Tensor of shape (N,) with 24-bit codewords
        """
        if triplets.dim() == 1:
            triplets = triplets.unsqueeze(0)

        N = triplets.shape[0]
        triplets = triplets.to(self.device)

        # Pack 3 INT4 values into 12-bit data word
        data_12bit = (
            (triplets[:, 0].to(torch.int64) & 0xF) |
            ((triplets[:, 1].to(torch.int64) & 0xF) << 4) |
            ((triplets[:, 2].to(torch.int64) & 0xF) << 8)
        )

        # Vectorized encoding: expand bits, multiply with G, pack result
        # Extract 12 data bits
        data_bits = torch.zeros((N, 12), dtype=torch.float32, device=self.device)
        for i in range(12):
            data_bits[:, i] = ((data_12bit >> i) & 1).float()

        # Matrix multiply with G (mod 2)
        codeword_bits = (data_bits @ self.G.float()) % 2

        # Pack 24 bits back into int64
        codewords = torch.zeros(N, dtype=torch.int64, device=self.device)
        for i in range(24):
            codewords = codewords | (codeword_bits[:, i].to(torch.int64) << i)

        return codewords

    def decode(self, codewords: torch.Tensor) -> GolayDecodeResult:
        """
        Decode Golay codewords with error correction - FULLY VECTORIZED.

        Uses pre-computed syndrome lookup table for O(1) error pattern lookup.
        No Python loops over codewords.

        Args:
            codewords: Tensor of shape (N,) with 24-bit codewords

        Returns:
            GolayDecodeResult with:
                - data: Tensor of shape (N, 3) with recovered INT4 triplets
                - errors_corrected: Total number of bit errors corrected
                - uncorrectable_count: Number of codewords with >3 errors
        """
        if codewords.dim() == 0:
            codewords = codewords.unsqueeze(0)

        N = codewords.shape[0]
        codewords = codewords.to(torch.int64).to(self.device)

        # Step 1: Compute syndromes - VECTORIZED
        syndromes = self._compute_syndrome_vectorized(codewords)

        # Step 2: Lookup error patterns from table - VECTORIZED
        # Use syndromes as indices into the pre-computed table
        error_patterns = self.syndrome_table[syndromes]
        error_weights = self.error_weights[syndromes]

        # Step 3: Apply corrections - VECTORIZED
        # Mask for correctable errors (not UNCORRECTABLE)
        correctable_mask = (error_patterns != self.UNCORRECTABLE)

        # XOR with error patterns where correctable
        corrected_codewords = codewords.clone()
        corrected_codewords[correctable_mask] = (
            corrected_codewords[correctable_mask] ^ error_patterns[correctable_mask]
        )

        # Step 4: Extract data bits - VECTORIZED
        # Data is in first 12 bits (systematic form)
        data_12bit = corrected_codewords & 0xFFF

        # Unpack to 3 INT4 values
        corrected_data = torch.zeros((N, 3), dtype=torch.uint8, device=self.device)
        corrected_data[:, 0] = (data_12bit & 0xF).to(torch.uint8)
        corrected_data[:, 1] = ((data_12bit >> 4) & 0xF).to(torch.uint8)
        corrected_data[:, 2] = ((data_12bit >> 8) & 0xF).to(torch.uint8)

        # Compute statistics
        errors_corrected = error_weights[correctable_mask].sum().item()
        uncorrectable_count = (~correctable_mask).sum().item()

        return GolayDecodeResult(
            data=corrected_data,
            errors_corrected=errors_corrected,
            uncorrectable_count=uncorrectable_count
        )

    def verify_properties(self) -> bool:
        """
        Verify mathematical properties of the Golay code.

        Checks:
        1. G @ H^T = 0 (mod 2)
        2. All codewords have minimum distance 8
        3. Syndrome table correctly maps errors

        Returns:
            True if all properties verified
        """
        # Check G @ H^T = 0 (mod 2)
        product = (self.G.float() @ self.H.T.float()) % 2
        if product.sum() != 0:
            print("FAIL: G @ H^T != 0")
            return False

        # Check minimum distance (sample-based for speed)
        min_dist = float('inf')
        for i in range(1, 4096, 16):  # Sample every 16th codeword
            triplet = torch.tensor([[i & 0xF, (i >> 4) & 0xF, (i >> 8) & 0xF]], dtype=torch.uint8)
            cw = self.encode(triplet)
            weight = bin(cw[0].item()).count('1')
            if weight > 0:
                min_dist = min(min_dist, weight)

        if min_dist != 8:
            print(f"FAIL: Minimum distance is {min_dist}, expected 8")
            return False

        # Verify single-bit error correction (vectorized test)
        triplet = torch.tensor([[5, 10, 3]], dtype=torch.uint8)
        cw = self.encode(triplet)

        # Create all 24 single-bit error patterns
        errors = torch.tensor([1 << i for i in range(24)], dtype=torch.int64, device=self.device)
        corrupted = cw.expand(24) ^ errors
        result = self.decode(corrupted)

        if not torch.all(result.data == triplet.expand(24, -1)):
            print("FAIL: Not all single-bit errors were corrected")
            return False

        print("All Golay(24,12) properties verified!")
        return True


def verify_golay_properties() -> bool:
    """Standalone verification function."""
    codec = Golay2412()
    return codec.verify_properties()


if __name__ == "__main__":
    print("Golay(24,12) Code Verification - Vectorized Implementation")
    print("=" * 60)

    codec = Golay2412()

    # Test basic encode/decode
    triplets = torch.tensor([
        [5, 10, 3],
        [0, 0, 0],
        [15, 15, 15],
        [7, 8, 9],
    ], dtype=torch.uint8)

    print(f"Input triplets:\n{triplets}")

    codewords = codec.encode(triplets)
    print(f"\nEncoded codewords: {codewords.tolist()}")
    print(f"Codeword weights: {[bin(cw.item()).count('1') for cw in codewords]}")

    result = codec.decode(codewords)
    print(f"\nDecoded triplets:\n{result.data}")
    print(f"Errors corrected: {result.errors_corrected}")

    # Test with errors - VECTORIZED
    print("\n" + "=" * 60)
    print("Testing error correction (vectorized)...")

    # Test 1, 2, 3 bit errors on first codeword
    for n_errors in [1, 2, 3]:
        cw = codewords[0:1].clone()
        error = sum(1 << i for i in range(n_errors))
        cw_corrupted = cw ^ error
        result = codec.decode(cw_corrupted)
        success = torch.equal(result.data[0], triplets[0])
        print(f"  {n_errors}-bit error: {'CORRECTED' if success else 'FAILED'}")

    # Speed test
    print("\n" + "=" * 60)
    print("Speed test (100,000 codewords)...")
    import time

    large_triplets = torch.randint(0, 16, (100000, 3), dtype=torch.uint8)
    large_cw = codec.encode(large_triplets)

    # Inject ~5% BER
    noise = torch.rand(100000) < 0.05
    large_cw_noisy = large_cw.clone()
    large_cw_noisy[noise] = large_cw_noisy[noise] ^ 1

    start = time.time()
    result = codec.decode(large_cw_noisy)
    elapsed = time.time() - start
    print(f"  Decoded 100,000 codewords in {elapsed*1000:.1f}ms")
    print(f"  Errors corrected: {result.errors_corrected}")

    # Verify properties
    print("\n" + "=" * 60)
    codec.verify_properties()
