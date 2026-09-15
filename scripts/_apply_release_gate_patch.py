from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"patch anchor not found: {label}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "services/api/app/quality/contracts.py",
    '''@dataclass(frozen=True)
class PromotionPolicy:
    max_recall_drop: float = 0.0
    max_mrr_drop: float = 0.0
    max_ndcg_drop: float = 0.0
    max_p95_latency_increase_ratio: float = 0.15
    max_cost_increase_ratio: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return {
            "max_recall_drop": float(self.max_recall_drop),
            "max_mrr_drop": float(self.max_mrr_drop),
            "max_ndcg_drop": float(self.max_ndcg_drop),
            "max_p95_latency_increase_ratio": float(self.max_p95_latency_increase_ratio),
            "max_cost_increase_ratio": float(self.max_cost_increase_ratio),
        }
''',
    '''@dataclass(frozen=True)
class PromotionPolicy:
    max_recall_drop: float = 0.0
    max_mrr_drop: float = 0.0
    max_ndcg_drop: float = 0.0
    max_p95_latency_increase_ratio: float = 0.15
    max_cost_increase_ratio: float = 0.25
    # Optional release-readiness gates. They are disabled by default so the
    # established relative baseline-vs-candidate policy remains compatible.
    min_recall: Optional[float] = None
    min_mrr: Optional[float] = None
    min_ndcg: Optional[float] = None
    critical_case_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("min_recall", "min_mrr", "min_ndcg"):
            value = getattr(self, name)
            if value is None:
                continue
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
            object.__setattr__(self, name, numeric)

        normalized: list[str] = []
        seen: set[str] = set()
        for raw in self.critical_case_ids or ():
            case_id = str(raw).strip()
            if not case_id:
                raise ValueError("critical_case_ids must not contain blank IDs")
            if case_id in seen:
                continue
            seen.add(case_id)
            normalized.append(case_id)
        object.__setattr__(self, "critical_case_ids", tuple(normalized))

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_recall_drop": float(self.max_recall_drop),
            "max_mrr_drop": float(self.max_mrr_drop),
            "max_ndcg_drop": float(self.max_ndcg_drop),
            "max_p95_latency_increase_ratio": float(self.max_p95_latency_increase_ratio),
            "max_cost_increase_ratio": float(self.max_cost_increase_ratio),
            "min_recall": self.min_recall,
            "min_mrr": self.min_mrr,
            "min_ndcg": self.min_ndcg,
            "critical_case_ids": list(self.critical_case_ids),
        }
''',
    "PromotionPolicy",
)

replace_once(
    "services/api/app/quality/promotion.py",
    '''    _ratio_gate(
        "cost_usd",
        baseline.cost,
        candidate.cost,
        max_increase_ratio=policy.max_cost_increase_ratio,
        reasons=reasons,
        deltas=deltas,
        zero_baseline_is_missing=False,
    )

''',
    '''    _ratio_gate(
        "cost_usd",
        baseline.cost,
        candidate.cost,
        max_increase_ratio=policy.max_cost_increase_ratio,
        reasons=reasons,
        deltas=deltas,
        zero_baseline_is_missing=False,
    )

    _absolute_quality_gates(candidate, policy, reasons=reasons, deltas=deltas)
    _critical_case_gate(candidate, policy, reasons=reasons, deltas=deltas)

''',
    "evaluate_promotion release calls",
)
replace_once(
    "services/api/app/quality/promotion.py",
    "def _quality_gate(\n",
    '''def _absolute_quality_gates(
    candidate: EvaluationRun,
    policy: PromotionPolicy,
    *,
    reasons: list[str],
    deltas: dict[str, Any],
) -> None:
    configured = {
        "recall": policy.min_recall,
        "mrr": policy.min_mrr,
        "ndcg": policy.min_ndcg,
    }
    if not any(value is not None for value in configured.values()):
        return

    evidence: dict[str, Any] = {}
    for key, minimum in configured.items():
        if minimum is None:
            continue
        candidate_value = _number(candidate.metrics.get(key))
        passed = (
            candidate_value is not None
            and 0.0 <= candidate_value <= 1.0
            and candidate_value >= float(minimum)
        )
        evidence[key] = {
            "candidate": candidate_value,
            "minimum": float(minimum),
            "passed": passed,
        }
        if candidate_value is None:
            reasons.append(f"missing candidate absolute quality metric: {key}")
        elif not 0.0 <= candidate_value <= 1.0:
            reasons.append(
                f"invalid candidate absolute quality metric: {key}={candidate_value}; expected [0,1]"
            )
        elif candidate_value < float(minimum):
            reasons.append(
                f"{key} absolute floor {candidate_value:.6f} is below required {float(minimum):.6f}"
            )
    deltas["absolute_quality"] = evidence


def _critical_case_gate(
    candidate: EvaluationRun,
    policy: PromotionPolicy,
    *,
    reasons: list[str],
    deltas: dict[str, Any],
) -> None:
    required = list(policy.critical_case_ids)
    if not required:
        return

    raw_cases = candidate.artifacts.get("cases")
    case_items = raw_cases if isinstance(raw_cases, list) else []
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for item in case_items:
        if not isinstance(item, Mapping):
            continue
        case_id = str(item.get("case_id") or "").strip()
        if not case_id:
            continue
        if case_id in by_id:
            duplicate_ids.add(case_id)
            continue
        by_id[case_id] = item

    passed: list[str] = []
    failed: list[dict[str, Any]] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for case_id in required:
        if case_id in duplicate_ids:
            ambiguous.append(case_id)
            reasons.append(f"critical case evidence ambiguous: {case_id}")
            continue
        item = by_id.get(case_id)
        if item is None:
            missing.append(case_id)
            reasons.append(f"critical case evidence missing: {case_id}")
            continue
        if item.get("retrieval_pass") is True:
            passed.append(case_id)
            continue
        failure = {
            "case_id": case_id,
            "first_relevant_rank": item.get("first_relevant_rank"),
        }
        failed.append(failure)
        reasons.append(
            f"critical case failed: {case_id} rank={item.get('first_relevant_rank')}"
        )

    deltas["critical_cases"] = {
        "required": required,
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "ambiguous": ambiguous,
    }


def _quality_gate(
''',
    "promotion helpers",
)

replace_once(
    "services/api/app/routes/quality.py",
    '''class PromotionPolicyRequest(BaseModel):
    max_recall_drop: float = Field(default=0.0, ge=0.0)
    max_mrr_drop: float = Field(default=0.0, ge=0.0)
    max_ndcg_drop: float = Field(default=0.0, ge=0.0)
    max_p95_latency_increase_ratio: float = Field(default=0.15, ge=0.0)
    max_cost_increase_ratio: float = Field(default=0.25, ge=0.0)
''',
    '''class PromotionPolicyRequest(BaseModel):
    max_recall_drop: float = Field(default=0.0, ge=0.0)
    max_mrr_drop: float = Field(default=0.0, ge=0.0)
    max_ndcg_drop: float = Field(default=0.0, ge=0.0)
    max_p95_latency_increase_ratio: float = Field(default=0.15, ge=0.0)
    max_cost_increase_ratio: float = Field(default=0.25, ge=0.0)
    min_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_mrr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_ndcg: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    critical_case_ids: list[str] = Field(default_factory=list)
''',
    "PromotionPolicyRequest",
)

args_old = '''    parser.add_argument("--max-cost-increase-ratio", type=float, default=0.25)
    return parser.parse_args(argv)
'''
args_new = '''    parser.add_argument("--max-cost-increase-ratio", type=float, default=0.25)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--min-mrr", type=float)
    parser.add_argument("--min-ndcg", type=float)
    parser.add_argument("--critical-case", action="append", default=[])
    return parser.parse_args(argv)
'''
policy_old = '''                max_p95_latency_increase_ratio=args.max_p95_latency_increase_ratio,
                max_cost_increase_ratio=args.max_cost_increase_ratio,
            ),
'''
policy_new = '''                max_p95_latency_increase_ratio=args.max_p95_latency_increase_ratio,
                max_cost_increase_ratio=args.max_cost_increase_ratio,
                min_recall=args.min_recall,
                min_mrr=args.min_mrr,
                min_ndcg=args.min_ndcg,
                critical_case_ids=tuple(args.critical_case),
            ),
'''
for filename in (
    "benchmarks/weighted_rrf_promotion.py",
    "benchmarks/retrieval_plan_promotion_legacy.py",
):
    replace_once(filename, args_old, args_new, f"{filename} CLI args")
    replace_once(filename, policy_old, policy_new, f"{filename} policy construction")
