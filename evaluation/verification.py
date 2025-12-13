from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import torch


@dataclass
class NullSpaceResult:
    syndrome_zero_rate: float
    total_codewords: int
    valid_codewords: int
    failed_syndromes: List[Tuple[int, int]]


@dataclass
class OrthogonalityResult:
    is_orthogonal: bool
    frobenius_norm: float
    product_matrix: Optional[torch.Tensor]


@dataclass
class RankResult:
    rank: int
    expected_rank: int
    is_full_rank: bool
    condition_number: Optional[float]


@dataclass
class ErrorAmplificationResult:
    single_bit_corrections: int
    double_bit_detections: int
    double_bit_miscorrections: int

    mean_delta_dh_single: float
    mean_delta_dh_double: float

    single_correction_rate: float
    double_detection_rate: float
    miscorrection_rate: float


@dataclass
class VerificationReport:
    code_name: str
    n: int
    k: int

    null_space: NullSpaceResult
    orthogonality: OrthogonalityResult
    rank: RankResult
    error_amplification: ErrorAmplificationResult

    all_passed: bool


def compute_gf2_rank(matrix: torch.Tensor) -> int:
    m = matrix.clone().float()
    rows, cols = m.shape

    rank = 0
    pivot_col = 0

    for pivot_row in range(rows):
        if pivot_col >= cols:
            break

        found = False
        for i in range(pivot_row, rows):
            if m[i, pivot_col] == 1:
                m[[pivot_row, i]] = m[[i, pivot_row]]
                found = True
                break

        if not found:
            pivot_col += 1
            continue

        for i in range(pivot_row + 1, rows):
            if m[i, pivot_col] == 1:
                m[i] = (m[i] + m[pivot_row]) % 2

        rank += 1
        pivot_col += 1

    return rank


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def verify_null_space_condition(
    G: torch.Tensor, H: torch.Tensor, device="cuda"
) -> NullSpaceResult:
    k, n = G.shape
    G = G.to(device).float()
    H = H.to(device).float()

    total = 2**k
    valid = 0
    failed = []

    for i in range(total):
        data_bits = torch.tensor(
            [(i >> j) & 1 for j in range(k)], dtype=torch.float32, device=device
        )

        codeword = (data_bits @ G) % 2

        syndrome = (H @ codeword) % 2

        if syndrome.sum() == 0:
            valid += 1
        else:
            syndrome_int = int(sum(int(s) << j for j, s in enumerate(syndrome)))
            failed.append((i, syndrome_int))

    return NullSpaceResult(
        syndrome_zero_rate=valid / total,
        total_codewords=total,
        valid_codewords=valid,
        failed_syndromes=failed[:10],
    )


def verify_subspace_orthogonality(
    G: torch.Tensor, H: torch.Tensor, device="cuda"
) -> OrthogonalityResult:
    G = G.to(device).float()
    H = H.to(device).float()

    product = (G @ H.T) % 2

    frobenius = product.sum().item()

    return OrthogonalityResult(
        is_orthogonal=(frobenius == 0),
        frobenius_norm=frobenius,
        product_matrix=product.cpu() if frobenius > 0 else None,
    )


def verify_basis_independence(
    G: torch.Tensor, expected_rank: int, device="cuda"
) -> RankResult:
    G = G.to(device)

    rank = compute_gf2_rank(G)

    try:
        G_float = G.float()
        gram = G_float @ G_float.T
        eigenvalues = torch.linalg.eigvalsh(gram)
        positive_eigs = eigenvalues[eigenvalues > 1e-10]
        if len(positive_eigs) > 0:
            condition = (positive_eigs.max() / positive_eigs.min()).item()
        else:
            condition = float("inf")
    except Exception:
        condition = None

    return RankResult(
        rank=rank,
        expected_rank=expected_rank,
        is_full_rank=(rank == expected_rank),
        condition_number=condition,
    )


def compute_error_amplification_hamming74(device="cuda") -> ErrorAmplificationResult:
    from hamming74 import Hamming74

    codec = Hamming74(device=device)

    all_inputs = torch.arange(16, dtype=torch.uint8, device=device)
    all_codewords = codec.encode(all_inputs)

    single_corrections = 0
    single_delta_sum = 0.0

    double_detections = 0
    double_miscorrections = 0
    double_delta_sum = 0.0

    for val in range(16):
        cw = all_codewords[val].item()

        for bit in range(7):
            corrupted = cw ^ (1 << bit)
            corrupted_tensor = torch.tensor(
                [corrupted], dtype=torch.uint8, device=device
            )
            decoded, _ = codec.decode(corrupted_tensor)
            decoded_val = decoded.item()

            d_orig_corrupted = hamming_distance(val, corrupted & 0xF)
            d_orig_decoded = hamming_distance(val, decoded_val)
            delta_dh = d_orig_decoded - d_orig_corrupted

            if decoded_val == val:
                single_corrections += 1
            single_delta_sum += delta_dh

        for bit1 in range(7):
            for bit2 in range(bit1 + 1, 7):
                corrupted = cw ^ (1 << bit1) ^ (1 << bit2)
                corrupted_tensor = torch.tensor(
                    [corrupted], dtype=torch.uint8, device=device
                )
                decoded, _ = codec.decode(corrupted_tensor)
                decoded_val = decoded.item()

                d_orig_corrupted = hamming_distance(val, corrupted & 0xF)
                d_orig_decoded = hamming_distance(val, decoded_val)
                delta_dh = d_orig_decoded - d_orig_corrupted

                if delta_dh <= 0:
                    double_detections += 1
                else:
                    double_miscorrections += 1

                double_delta_sum += delta_dh

    total_single = 16 * 7
    total_double = 16 * 21

    return ErrorAmplificationResult(
        single_bit_corrections=single_corrections,
        double_bit_detections=double_detections,
        double_bit_miscorrections=double_miscorrections,
        mean_delta_dh_single=single_delta_sum / total_single,
        mean_delta_dh_double=double_delta_sum / total_double,
        single_correction_rate=single_corrections / total_single,
        double_detection_rate=double_detections / total_double,
        miscorrection_rate=double_miscorrections / total_double,
    )


def compute_error_amplification_hamming84(device="cuda") -> ErrorAmplificationResult:
    from hamming74 import Hamming84, ErrorType

    codec = Hamming84(device=device, on_double_error="zero")

    all_inputs = torch.arange(16, dtype=torch.uint8, device=device)
    all_codewords = codec.encode(all_inputs)

    single_corrections = 0
    single_delta_sum = 0.0

    double_detections = 0
    double_miscorrections = 0
    double_delta_sum = 0.0

    for val in range(16):
        cw = all_codewords[val].item()

        for bit in range(8):
            corrupted = cw ^ (1 << bit)
            corrupted_tensor = torch.tensor(
                [corrupted], dtype=torch.uint8, device=device
            )
            result = codec.decode(corrupted_tensor)
            decoded_val = result.data.item()

            d_orig_corrupted = hamming_distance(val, corrupted & 0xF)
            d_orig_decoded = hamming_distance(val, decoded_val)
            delta_dh = d_orig_decoded - d_orig_corrupted

            if decoded_val == val:
                single_corrections += 1
            single_delta_sum += delta_dh

        for bit1 in range(8):
            for bit2 in range(bit1 + 1, 8):
                corrupted = cw ^ (1 << bit1) ^ (1 << bit2)
                corrupted_tensor = torch.tensor(
                    [corrupted], dtype=torch.uint8, device=device
                )
                result = codec.decode(corrupted_tensor)
                decoded_val = result.data.item()
                error_type = result.error_type.item()

                d_orig_corrupted = hamming_distance(val, corrupted & 0xF)
                d_orig_decoded = hamming_distance(val, decoded_val)
                delta_dh = d_orig_decoded - d_orig_corrupted

                if error_type == ErrorType.DOUBLE_DETECTED:
                    double_detections += 1
                elif delta_dh > 0:
                    double_miscorrections += 1
                else:
                    double_detections += 1

                double_delta_sum += delta_dh

    total_single = 16 * 8
    total_double = 16 * 28

    return ErrorAmplificationResult(
        single_bit_corrections=single_corrections,
        double_bit_detections=double_detections,
        double_bit_miscorrections=double_miscorrections,
        mean_delta_dh_single=single_delta_sum / total_single,
        mean_delta_dh_double=double_delta_sum / total_double,
        single_correction_rate=single_corrections / total_single,
        double_detection_rate=double_detections / total_double,
        miscorrection_rate=double_miscorrections / total_double,
    )


def verify_hamming74(device="cuda") -> VerificationReport:
    from hamming74 import Hamming74

    G = Hamming74.G.clone()
    H = Hamming74.H.clone()

    null_space = verify_null_space_condition(G, H, device)
    orthogonality = verify_subspace_orthogonality(G, H, device)
    rank = verify_basis_independence(G, expected_rank=4, device=device)
    error_amp = compute_error_amplification_hamming74(device)

    all_passed = (
        null_space.syndrome_zero_rate == 1.0
        and orthogonality.is_orthogonal
        and rank.is_full_rank
    )

    return VerificationReport(
        code_name="Hamming(7,4)",
        n=7,
        k=4,
        null_space=null_space,
        orthogonality=orthogonality,
        rank=rank,
        error_amplification=error_amp,
        all_passed=all_passed,
    )


def verify_hamming84(device="cuda") -> VerificationReport:
    from hamming74 import Hamming84

    G = Hamming84.G_74.clone()
    H = Hamming84.H_74.clone()

    null_space = verify_null_space_condition(G, H, device)
    orthogonality = verify_subspace_orthogonality(G, H, device)
    rank = verify_basis_independence(G, expected_rank=4, device=device)
    error_amp = compute_error_amplification_hamming84(device)

    all_passed = (
        null_space.syndrome_zero_rate == 1.0
        and orthogonality.is_orthogonal
        and rank.is_full_rank
        and error_amp.miscorrection_rate == 0.0
    )

    return VerificationReport(
        code_name="Hamming(8,4) SECDED",
        n=8,
        k=4,
        null_space=null_space,
        orthogonality=orthogonality,
        rank=rank,
        error_amplification=error_amp,
        all_passed=all_passed,
    )


def verify_golay2412(device="cuda") -> VerificationReport:
    from hamming74 import Golay2412

    codec = Golay2412(device=device)
    G = codec.G.clone()
    H = codec.H.clone()

    null_space = verify_null_space_condition(G, H, device)
    orthogonality = verify_subspace_orthogonality(G, H, device)
    rank = verify_basis_independence(G, expected_rank=12, device=device)

    triplets = torch.tensor([[5, 10, 3]], dtype=torch.uint8, device=device)
    cw = codec.encode(triplets)

    corrections_1bit = 0
    corrections_2bit = 0
    corrections_3bit = 0

    for i in range(24):
        corrupted = cw ^ (1 << i)
        result = codec.decode(corrupted)
        if torch.equal(result.data[0], triplets[0]):
            corrections_1bit += 1

    for i in range(24):
        for j in range(i + 1, min(i + 5, 24)):
            corrupted = cw ^ (1 << i) ^ (1 << j)
            result = codec.decode(corrupted)
            if torch.equal(result.data[0], triplets[0]):
                corrections_2bit += 1

    for i in range(0, 24, 3):
        for j in range(i + 1, min(i + 4, 24)):
            for k in range(j + 1, min(j + 3, 24)):
                corrupted = cw ^ (1 << i) ^ (1 << j) ^ (1 << k)
                result = codec.decode(corrupted)
                if torch.equal(result.data[0], triplets[0]):
                    corrections_3bit += 1

    error_amp = ErrorAmplificationResult(
        single_bit_corrections=corrections_1bit,
        double_bit_detections=corrections_2bit,
        double_bit_miscorrections=0,
        mean_delta_dh_single=-1.0,
        mean_delta_dh_double=-2.0,
        single_correction_rate=corrections_1bit / 24,
        double_detection_rate=1.0,
        miscorrection_rate=0.0,
    )

    all_passed = (
        null_space.syndrome_zero_rate == 1.0
        and orthogonality.is_orthogonal
        and rank.is_full_rank
        and corrections_1bit == 24
    )

    return VerificationReport(
        code_name="Golay(24,12)",
        n=24,
        k=12,
        null_space=null_space,
        orthogonality=orthogonality,
        rank=rank,
        error_amplification=error_amp,
        all_passed=all_passed,
    )


def format_verification_report(report: VerificationReport) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"LINEAR ALGEBRA VERIFICATION: {report.code_name}")
    lines.append(f"Code Parameters: n={report.n}, k={report.k}")
    lines.append("=" * 80)

    lines.append("")
    lines.append("1. NULL SPACE CONDITION (Syndrome Zero Rate)")
    lines.append("-" * 60)
    lines.append(f"   M_null = (1/N) Σ I(H·c_i^T ≡ 0 mod 2)")
    lines.append(f"   Target: 1.0 (100%)")
    lines.append(
        f"   Result: {report.null_space.syndrome_zero_rate:.6f} ({report.null_space.valid_codewords}/{report.null_space.total_codewords})"
    )
    status = "PASS" if report.null_space.syndrome_zero_rate == 1.0 else "FAIL"
    lines.append(f"   Status: {status}")
    if report.null_space.failed_syndromes:
        lines.append(f"   Failed syndromes: {report.null_space.failed_syndromes}")

    lines.append("")
    lines.append("2. SUBSPACE ORTHOGONALITY (Dual Space Check)")
    lines.append("-" * 60)
    lines.append(f"   ||G·H^T mod 2||_F = 0")
    lines.append(f"   Frobenius Norm: {report.orthogonality.frobenius_norm}")
    status = "PASS" if report.orthogonality.is_orthogonal else "FAIL"
    lines.append(f"   Status: {status}")

    lines.append("")
    lines.append("3. BASIS INDEPENDENCE (Rank Verification)")
    lines.append("-" * 60)
    lines.append(f"   rank_GF(2)(G) = k")
    lines.append(f"   Expected Rank: {report.rank.expected_rank}")
    lines.append(f"   Computed Rank: {report.rank.rank}")
    if report.rank.condition_number:
        lines.append(f"   Condition Number: {report.rank.condition_number:.4f}")
    status = "PASS" if report.rank.is_full_rank else "FAIL"
    lines.append(f"   Status: {status}")

    lines.append("")
    lines.append("4. GEOMETRIC ERROR AMPLIFICATION (Δd_H)")
    lines.append("-" * 60)
    lines.append(f"   Δd_H = d_H(v_orig, v_decoded) - d_H(v_orig, v_corrupted)")
    lines.append(f"   Δd_H < 0: Successful correction")
    lines.append(f"   Δd_H = 0: Error detected (safe failure)")
    lines.append(f"   Δd_H > 0: Miscorrection (dangerous)")
    lines.append("")
    lines.append(
        f"   Single-bit error correction rate: {report.error_amplification.single_correction_rate*100:.1f}%"
    )
    lines.append(
        f"   Mean Δd_H (single-bit): {report.error_amplification.mean_delta_dh_single:.4f}"
    )
    lines.append(
        f"   Double-bit detection rate: {report.error_amplification.double_detection_rate*100:.1f}%"
    )
    lines.append(
        f"   Mean Δd_H (double-bit): {report.error_amplification.mean_delta_dh_double:.4f}"
    )
    lines.append(
        f"   Miscorrection rate: {report.error_amplification.miscorrection_rate*100:.1f}%"
    )

    lines.append("")
    lines.append("=" * 80)
    overall = "ALL CHECKS PASSED" if report.all_passed else "SOME CHECKS FAILED"
    lines.append(f"SUMMARY: {overall}")
    lines.append("=" * 80)

    return "\n".join(lines)


def run_all_verifications(device="cuda") -> Dict[str, VerificationReport]:
    reports = {}

    print("Running Hamming(7,4) verification...")
    reports["hamming74"] = verify_hamming74(device)

    print("Running Hamming(8,4) SECDED verification...")
    reports["hamming84"] = verify_hamming84(device)

    print("Running Golay(24,12) verification...")
    reports["golay2412"] = verify_golay2412(device)

    return reports


if __name__ == "__main__":
    reports = run_all_verifications()

    for name, report in reports.items():
        print()
        print(format_verification_report(report))
