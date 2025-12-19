import random
import copy
from typing import Dict, List, Tuple, Optional, Any, Union

# ============================================================
# RepoGenerator v3
# ------------------------------------------------------------
# 相比 RepoGenerator_v2.py：
# - side_effects 支持 ensure / force_high / force_low / pin
# - high_version_conflicts 注入部分“范围冲突”（例如 >= next_solution_version）
# 以匹配 RepoEnv_v5.py 的最新规则，并提升难度（非单调/更强冲突）。
# ============================================================

# =========================
# 基础类型与工具函数
# =========================

Version = Tuple[int, int]  # (major, minor)
Constraint = Tuple[str, Version]  # (op, version)
ConstraintList = List[Constraint]


def compare_versions(
    a: Union[int, float, Version], b: Union[int, float, Version]
) -> int:
    """
    返回:
    -1 if a < b
     0 if a == b
     1 if a > b
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        if a[0] != b[0]:
            return -1 if a[0] < b[0] else 1
        if a[1] != b[1]:
            return -1 if a[1] < b[1] else 1
        return 0

    af = float(a)
    bf = float(b)
    if af < bf:
        return -1
    elif af > bf:
        return 1
    else:
        return 0


def check_version_constraint(ver: Version, op: str, target: Version) -> bool:
    cmp = compare_versions(ver, target)
    if op == "==":
        return cmp == 0
    if op == "!=":
        return cmp != 0
    if op == ">":
        return cmp == 1
    if op == ">=":
        return cmp in (0, 1)
    if op == "<":
        return cmp == -1
    if op == "<=":
        return cmp in (0, -1)
    raise ValueError(f"Unknown operator: {op}")


def version_satisfies_constraints(ver: Version, constraints: ConstraintList) -> bool:
    return all(check_version_constraint(ver, op, tgt) for op, tgt in constraints)


def parse_semver_to_tuple(ver_str: str) -> Version:
    """
    "3.10" -> (3,10)
    "2"    -> (2,0)
    """
    s = ver_str.strip()
    if "." in s:
        parts = s.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid semantic version: {ver_str}")
        return (int(parts[0]), int(parts[1]))
    return (int(s), 0)


def format_python_version(v: Version) -> str:
    return f"{v[0]}.{v[1]}"


def format_pkg_version(v: Version) -> str:
    return f"{v[0]}.{v[1]}"


def format_constraint_list(constraints: ConstraintList) -> str:
    if not constraints:
        return "<none>"
    return ",".join(f"{op}{format_pkg_version(v)}" for op, v in constraints)


def normalize_constraints(
    constraints: ConstraintList, all_versions: List[Version]
) -> ConstraintList:
    """
    将多个可能冗余的约束化简成规范区间：
      - 合并所有 lower bounds 为最大的下界
      - 合并所有 upper bounds 为最小的上界
      - 若出现多重 "=="，必须完全一致，否则返回 []
    若最终没有任何版本满足，则返回 []（表示冲突）。
    """
    lowers: List[Tuple[Version, bool]] = []
    uppers: List[Tuple[Version, bool]] = []
    equals: List[Version] = []

    for op, tgt in constraints:
        if op == "==":
            equals.append(tgt)
        elif op == ">=":
            lowers.append((tgt, True))
        elif op == ">":
            lowers.append((tgt, False))
        elif op == "<=":
            uppers.append((tgt, True))
        elif op == "<":
            uppers.append((tgt, False))

    # 有 "=="，则收敛到一个固定版本
    if equals:
        base = equals[0]
        for e in equals[1:]:
            if compare_versions(e, base) != 0:
                return []
        if base not in all_versions:
            return []
        return [("==", base)]

    # 合并下界
    low_ver: Optional[Version] = None
    low_inc = True
    if lowers:
        low_ver, low_inc = max(lowers, key=lambda x: x[0])

    # 合并上界
    up_ver: Optional[Version] = None
    up_inc = True
    if uppers:
        up_ver, up_inc = min(uppers, key=lambda x: x[0])

    def ok(v: Version) -> bool:
        if low_ver is not None:
            cmp = compare_versions(v, low_ver)
            if cmp < 0 or (cmp == 0 and not low_inc):
                return False
        if up_ver is not None:
            cmp = compare_versions(v, up_ver)
            if cmp > 0 or (cmp == 0 and not up_inc):
                return False
        return True

    valid = [v for v in all_versions if ok(v)]
    if not valid:
        return []

    out: ConstraintList = []
    if low_ver is not None:
        out.append((">=" if low_inc else ">", low_ver))
    if up_ver is not None:
        out.append(("<=" if up_inc else "<", up_ver))
    return out


def intersect_constraints(
    a: ConstraintList, b: ConstraintList, all_versions: List[Version]
) -> ConstraintList:
    """
    a ∧ b 的交集，并自动 normalize。
    """
    return normalize_constraints(a + b, all_versions)


# =========================
# JSON-safe World Generator v7.5 (RepoGenerator v3)
# =========================


class RandomWorldSpecGeneratorV7_5:
    """
    v7.5: world_spec 结构完全 JSON-safe（所有 dict 的 key 都是字符串），
    但内部生成仍然使用 tuple 版本，然后在结尾统一转换为 JSON 友好的格式。
    """

    def __init__(
        self,
        num_packages: int = 8,
        min_versions: int = 3,
        max_versions: int = 7,
        python_versions: Optional[List[Version]] = None,
        rng_seed: Optional[int] = None,
        # 难度/风格控制参数
        project_range_strict_prob: float = 0.5,
        implicit_range_strict_prob: float = 0.4,
        dep_range_strict_prob: float = 0.6,
        dep_edge_prob: float = 0.8,
        max_deps_per_version: int = 3,
        high_version_conflict_ratio: float = 0.5,
        fork_point_ratio: float = 0.4,
    ):
        self.rng_seed = rng_seed
        self.rng = random.Random(rng_seed)
        self.num_packages = num_packages
        self.min_versions = min_versions
        self.max_versions = max_versions
        self.project_range_strict_prob = project_range_strict_prob
        self.implicit_range_strict_prob = implicit_range_strict_prob
        self.dep_range_strict_prob = dep_range_strict_prob
        self.dep_edge_prob = dep_edge_prob
        self.max_deps_per_version = max_deps_per_version
        self.high_version_conflict_ratio = high_version_conflict_ratio
        self.fork_point_ratio = fork_point_ratio

        if python_versions is None:
            python_versions = [(3, 8), (3, 9), (3, 10), (3, 11)]
        self.python_versions = python_versions

        # -------------------------
        # Synthetic "API surface"
        # -------------------------
        # 用于生成 module/symbol/kwargs，从而在 runtime 只暴露对象，不暴露版本区间。
        self._symbol_pool = [
            "load",
            "dump",
            "fit",
            "transform",
            "predict",
            "compile",
            "Backend",
            "Session",
            "Engine",
            "Tokenizer",
            "Model",
            "Dataset",
            "Pipeline",
            "Config",
            "Factory",
            "Registry",
            "Adapter",
            "Resolver",
        ]
        self._kw_pool = [
            "axis",
            "dtype",
            "device",
            "backend",
            "strict",
            "mode",
            "seed",
            "timeout",
            "cache",
            "format",
            "verbose",
            "strategy",
            "precision",
            "batch_size",
        ]

        # base-lib 注入强度（目前作为常量固定；写入 meta 便于人工校验）
        self.base_dep_prob = 0.85
        self.base_strict_prob = 0.95
        self.base_slice_extra_prob = 0.6

    def _generator_params(self) -> Dict[str, Any]:
        """
        返回本次 generator 的关键参数（用于写入 spec 的 _gen 字段，便于人工校验）。
        """
        return {
            "generator": "RepoGenerator_v5.RandomWorldSpecGeneratorV7_5",
            "rng_seed": self.rng_seed,
            "num_packages": self.num_packages,
            "min_versions": self.min_versions,
            "max_versions": self.max_versions,
            "python_versions": [format_python_version(v) for v in self.python_versions],
            "project_range_strict_prob": self.project_range_strict_prob,
            "implicit_range_strict_prob": self.implicit_range_strict_prob,
            "dep_range_strict_prob": self.dep_range_strict_prob,
            "dep_edge_prob": self.dep_edge_prob,
            "max_deps_per_version": self.max_deps_per_version,
            "high_version_conflict_ratio": self.high_version_conflict_ratio,
            "fork_point_ratio": self.fork_point_ratio,
            "base_dep_prob": self.base_dep_prob,
            "base_strict_prob": self.base_strict_prob,
            "base_slice_extra_prob": self.base_slice_extra_prob,
        }

    def _build_rules_nl(self, spec: Dict[str, Any]) -> str:
        """
        把 spec 的“规则”整理成自然语言可读文本，便于放进 prompt 作为 rulebook。
        注意：这里描述的是规则本身，不包含 ground-truth 解。
        """
        lines: List[str] = []
        lines.append("## Rulebook (ground-truth environment rules)")

        # -------- Overview --------
        pyvers = spec.get("python_versions", []) or []
        base_libs = spec.get("base_libs", []) or []
        cons_rules = spec.get("consistency_rules", []) or []
        entrypoints = spec.get("entrypoints", []) or []
        scripts = spec.get("scripts", {}) or {}

        lines.append("### Overview")
        lines.append(f"- Available Python versions: {', '.join(pyvers) if pyvers else '<unknown>'}")
        lines.append(f"- Base libraries (shared deps): {', '.join(base_libs) if base_libs else '<none>'}")
        if entrypoints:
            lines.append("- Full project command `python run.py` executes scripts in order:")
            for p in entrypoints:
                lines.append(f"  - {p}")
        else:
            lines.append("- Full project command `python run.py` executes a single entry (no scripts/entrypoints specified).")

        # -------- Consistency rules --------
        lines.append("### Global consistency rules")
        if not cons_rules:
            lines.append("- <none>")
        else:
            for r in cons_rules:
                if not isinstance(r, dict):
                    continue
                rtype = r.get("type")
                anchor = r.get("anchor")
                pkgs = r.get("packages", []) or []
                if rtype == "same_major_with_anchor":
                    lines.append(
                        f"- ABI-major lockstep: for packages {pkgs}, their MAJOR must equal anchor {anchor}'s MAJOR."
                    )
                elif rtype == "same_minor_with_anchor":
                    lines.append(
                        f"- Tight lockstep: for packages {pkgs}, their (MAJOR,MINOR) must equal anchor {anchor}'s (MAJOR,MINOR)."
                    )
                else:
                    lines.append(
                        f"- {rtype}: anchor={anchor}, packages={pkgs}"
                    )

        # -------- Scripts --------
        lines.append("### Repo scripts (debuggable entrypoints)")
        if isinstance(scripts, dict) and scripts:
            for path in sorted(scripts.keys()):
                info = scripts.get(path, {}) or {}
                calls = info.get("calls", []) or []
                imports = info.get("imports", []) or []
                providers = []
                for c in calls:
                    if isinstance(c, dict) and isinstance(c.get("provider"), str):
                        providers.append(c["provider"])
                providers = sorted(set(providers))
                lines.append(f"- {path}")
                if providers:
                    lines.append(f"  - Providers touched by calls: {', '.join(providers)}")
                if imports:
                    lines.append(f"  - Extra imports/callers: {', '.join(imports)}")
                if calls:
                    lines.append("  - Calls:")
                    for c in calls:
                        if not isinstance(c, dict):
                            continue
                        p = c.get("provider")
                        m = c.get("module")
                        s = c.get("symbol")
                        kws = c.get("required_kwargs", []) or []
                        if isinstance(p, str) and isinstance(m, str) and isinstance(s, str):
                            if kws:
                                lines.append(f"    - {p}: {m}.{s}(kwargs: {', '.join(kws)})")
                            else:
                                lines.append(f"    - {p}: {m}.{s}()")
        else:
            lines.append("- <none>")

        # Helpers to format constraints
        def fmt_cons(cons_json: List[List[str]]) -> str:
            if not cons_json:
                return "<none>"
            return ",".join(f"{op}{vs}" for op, vs in cons_json)

        # -------- Package rules --------
        pkgs = spec.get("packages", {}) or {}
        vdeps = spec.get("version_dependencies", {}) or {}
        vconfs = spec.get("version_conflicts", {}) or {}
        vpy = spec.get("version_requires_python", {}) or {}
        se = spec.get("side_effects", {}) or {}
        symreq = spec.get("symbol_requires", {}) or {}

        lines.append("### Package rules (per version)")
        for pkg in sorted(pkgs.keys()):
            pinfo = pkgs.get(pkg, {}) or {}
            vers = pinfo.get("versions", []) or []
            prio = pinfo.get("priority", "?")
            lines.append(f"\n#### {pkg} (priority={prio})")
            lines.append(f"- Versions: {', '.join(vers) if vers else '<none>'}")

            # per version: requires-python
            vpy_map = vpy.get(pkg, {}) if isinstance(vpy, dict) else {}
            vdep_map = vdeps.get(pkg, {}) if isinstance(vdeps, dict) else {}
            vconf_map = vconfs.get(pkg, {}) if isinstance(vconfs, dict) else {}
            se_map = se.get(pkg, {}) if isinstance(se, dict) else {}
            sym_map = symreq.get(pkg, {}) if isinstance(symreq, dict) else {}

            for ver in vers:
                lines.append(f"- {pkg}=={ver}")

                # requires python
                if isinstance(vpy_map, dict) and ver in vpy_map:
                    pair = vpy_map.get(ver, None)
                    if isinstance(pair, list) and len(pair) == 2:
                        lines.append(f"  - Requires-Python: {pair[0]} {pair[1]}")

                # dependencies
                dep_list = []
                if isinstance(vdep_map, dict):
                    dep_list = vdep_map.get(ver, []) or []
                if dep_list:
                    lines.append("  - Requires:")
                    for dep_pkg, cons_json in dep_list:
                        lines.append(f"    - {dep_pkg} {fmt_cons(cons_json)}")
                else:
                    lines.append("  - Requires: <none>")

                # conflicts
                conf_list = []
                if isinstance(vconf_map, dict):
                    conf_list = vconf_map.get(ver, []) or []
                if conf_list:
                    lines.append("  - Conflicts:")
                    for item in conf_list:
                        # [conf_pkg, op, ver_str, msg]
                        if isinstance(item, list) and len(item) >= 3:
                            conf_pkg, op, conf_ver = item[0], item[1], item[2]
                            lines.append(f"    - {conf_pkg} {op} {conf_ver}")
                else:
                    lines.append("  - Conflicts: <none>")

                # side-effects
                eff_list = []
                if isinstance(se_map, dict):
                    eff_list = se_map.get(ver, []) or []
                if eff_list:
                    lines.append("  - Side-Effects (auto changes when installing this exact version):")
                    for e in eff_list:
                        # [eff_type, dep_pkg, [[op, ver], ...]]
                        if isinstance(e, list) and len(e) == 3:
                            eff_type, dep_pkg, cons_json = e
                            lines.append(f"    - {eff_type}: {dep_pkg} {fmt_cons(cons_json)}")
                else:
                    lines.append("  - Side-Effects: <none>")

                # symbol requires (runtime import/call requirements)
                sreqs = []
                if isinstance(sym_map, dict):
                    sreqs = sym_map.get(ver, []) or []
                if sreqs:
                    lines.append("  - Runtime symbol-requires (while importing this package version):")
                    for r in sreqs:
                        if not isinstance(r, dict):
                            continue
                        prov = r.get("provider")
                        mod = r.get("module")
                        sym = r.get("symbol")
                        kws = r.get("required_kwargs", []) or []
                        if isinstance(prov, str) and isinstance(mod, str) and isinstance(sym, str):
                            if kws:
                                lines.append(f"    - needs {prov}: {mod}.{sym}(kwargs: {', '.join(kws)})")
                            else:
                                lines.append(f"    - needs {prov}: {mod}.{sym}()")
                else:
                    lines.append("  - Runtime symbol-requires: <none>")

        # -------- Hidden rules --------
        lines.append("\n### Hidden rules (conditional failures)")
        hidden_rules = spec.get("hidden_rules", []) or []
        if not hidden_rules:
            lines.append("- <none>")
        else:
            for r in hidden_rules:
                if not isinstance(r, dict):
                    continue
                scopes = r.get("scope", ["run.py"])
                when = r.get("when", []) or []
                err = r.get("error", "RuntimeError: unknown.")
                lines.append(f"- scope={scopes}: when={when} => error='{err}'")

        return "\n".join(lines)

    def _build_rules_nl_compact(self, spec: Dict[str, Any]) -> str:
        """
        更紧凑的 rulebook：尽量“全”，但通过去重与合并减少 token。
        主要策略：
        - 对每个 pkg，把版本按“同一组规则签名”分组，合并成 versions 列表
        - 用短标签：RP(Requires-Python) / REQ / CON / SE / SYM
        - 省略重复的说明句，保留结构化要点
        """
        lines: List[str] = []
        lines.append("## Rulebook (compact)")

        pyvers = spec.get("python_versions", []) or []
        base_libs = spec.get("base_libs", []) or []
        cons_rules = spec.get("consistency_rules", []) or []
        entrypoints = spec.get("entrypoints", []) or []
        scripts = spec.get("scripts", {}) or {}

        lines.append(f"- PY: {', '.join(pyvers) if pyvers else '<unknown>'}")
        lines.append(f"- BASE: {', '.join(base_libs) if base_libs else '<none>'}")
        if entrypoints:
            lines.append("- RUN: python run.py executes:")
            lines.append("  - " + " -> ".join(entrypoints))
        else:
            lines.append("- RUN: python run.py executes a single entry.")

        # consistency
        lines.append("- CONSISTENCY:")
        if not cons_rules:
            lines.append("  - <none>")
        else:
            for r in cons_rules:
                if not isinstance(r, dict):
                    continue
                rtype = r.get("type")
                anchor = r.get("anchor")
                pkgs = r.get("packages", []) or []
                if rtype == "same_major_with_anchor":
                    lines.append(f"  - same_major(anchor={anchor}): pkgs={pkgs}")
                elif rtype == "same_minor_with_anchor":
                    lines.append(f"  - same_minor(anchor={anchor}): pkgs={pkgs}")
                else:
                    lines.append(f"  - {rtype}(anchor={anchor}): pkgs={pkgs}")

        # scripts (compact)
        lines.append("- SCRIPTS:")
        if isinstance(scripts, dict) and scripts:
            for path in sorted(scripts.keys()):
                info = scripts.get(path, {}) or {}
                calls = info.get("calls", []) or []
                imports = info.get("imports", []) or []
                providers = []
                for c in calls:
                    if isinstance(c, dict) and isinstance(c.get("provider"), str):
                        providers.append(c["provider"])
                providers = sorted(set(providers))
                lines.append(f"  - {path}: providers={providers} imports={len(imports)} calls={len(calls)}")
        else:
            lines.append("  - <none>")

        def fmt_cons(cons_json: List[List[str]]) -> str:
            if not cons_json:
                return "<none>"
            return ",".join(f"{op}{vs}" for op, vs in cons_json)

        # Package-level compaction
        pkgs = spec.get("packages", {}) or {}
        vdeps = spec.get("version_dependencies", {}) or {}
        vconfs = spec.get("version_conflicts", {}) or {}
        vpy = spec.get("version_requires_python", {}) or {}
        se = spec.get("side_effects", {}) or {}
        symreq = spec.get("symbol_requires", {}) or {}

        lines.append("\n## Packages (grouped by identical rules)")
        for pkg in sorted(pkgs.keys()):
            pinfo = pkgs.get(pkg, {}) or {}
            vers = pinfo.get("versions", []) or []
            prio = pinfo.get("priority", "?")
            lines.append(f"\n### {pkg} (prio={prio})")

            vpy_map = vpy.get(pkg, {}) if isinstance(vpy, dict) else {}
            vdep_map = vdeps.get(pkg, {}) if isinstance(vdeps, dict) else {}
            vconf_map = vconfs.get(pkg, {}) if isinstance(vconfs, dict) else {}
            se_map = se.get(pkg, {}) if isinstance(se, dict) else {}
            sym_map = symreq.get(pkg, {}) if isinstance(symreq, dict) else {}

            # build rule signature per version
            groups: Dict[str, List[str]] = {}
            payloads: Dict[str, List[str]] = {}

            for ver in vers:
                rp = ""
                if isinstance(vpy_map, dict) and ver in vpy_map:
                    pair = vpy_map.get(ver, None)
                    if isinstance(pair, list) and len(pair) == 2:
                        rp = f"{pair[0]} {pair[1]}"

                dep_list = vdep_map.get(ver, []) if isinstance(vdep_map, dict) else []
                dep_list = dep_list or []
                dep_norm = []
                for dep_pkg, cons_json in dep_list:
                    dep_norm.append((dep_pkg, fmt_cons(cons_json)))
                dep_norm.sort()

                conf_list = vconf_map.get(ver, []) if isinstance(vconf_map, dict) else []
                conf_list = conf_list or []
                conf_norm = []
                for item in conf_list:
                    if isinstance(item, list) and len(item) >= 3:
                        conf_norm.append((item[0], item[1], item[2]))
                conf_norm.sort()

                eff_list = se_map.get(ver, []) if isinstance(se_map, dict) else []
                eff_list = eff_list or []
                eff_norm = []
                for e in eff_list:
                    if isinstance(e, list) and len(e) == 3:
                        eff_norm.append((e[0], e[1], fmt_cons(e[2])))
                eff_norm.sort()

                sreqs = sym_map.get(ver, []) if isinstance(sym_map, dict) else []
                sreqs = sreqs or []
                sreq_norm = []
                for r in sreqs:
                    if not isinstance(r, dict):
                        continue
                    prov = r.get("provider")
                    mod = r.get("module")
                    sym = r.get("symbol")
                    kws = r.get("required_kwargs", []) or []
                    if isinstance(prov, str) and isinstance(mod, str) and isinstance(sym, str):
                        kwtxt = ",".join([k for k in kws if isinstance(k, str)])
                        sreq_norm.append((prov, mod, sym, kwtxt))
                sreq_norm.sort()

                sig = repr((rp, dep_norm, conf_norm, eff_norm, sreq_norm))
                groups.setdefault(sig, []).append(ver)

            # render groups
            for sig, vlist in groups.items():
                vlist_sorted = vlist[:]  # keep deterministic order
                # payload: rebuild human text from sig by recomputing for first version in group
                v0 = vlist_sorted[0]

                parts: List[str] = []
                # RP
                rp = ""
                if isinstance(vpy_map, dict) and v0 in vpy_map:
                    pair = vpy_map.get(v0, None)
                    if isinstance(pair, list) and len(pair) == 2:
                        rp = f"{pair[0]} {pair[1]}"
                if rp:
                    parts.append(f"RP={rp}")

                # REQ
                dep_list = vdep_map.get(v0, []) if isinstance(vdep_map, dict) else []
                dep_list = dep_list or []
                if dep_list:
                    dep_txt = "; ".join([f"{d} {fmt_cons(c)}" for d, c in dep_list])
                    parts.append(f"REQ=[{dep_txt}]")
                else:
                    parts.append("REQ=[]")

                # CON
                conf_list = vconf_map.get(v0, []) if isinstance(vconf_map, dict) else []
                conf_list = conf_list or []
                if conf_list:
                    items = []
                    for it in conf_list:
                        if isinstance(it, list) and len(it) >= 3:
                            items.append(f"{it[0]} {it[1]} {it[2]}")
                    parts.append(f"CON=[{'; '.join(items)}]")
                else:
                    parts.append("CON=[]")

                # SE
                eff_list = se_map.get(v0, []) if isinstance(se_map, dict) else []
                eff_list = eff_list or []
                if eff_list:
                    items = []
                    for e in eff_list:
                        if isinstance(e, list) and len(e) == 3:
                                items.append(f"{e[0]}:{e[1]} {fmt_cons(e[2])}")
                    parts.append(f"SE=[{'; '.join(items)}]")
                else:
                    parts.append("SE=[]")

                # SYM
                sreqs = sym_map.get(v0, []) if isinstance(sym_map, dict) else []
                sreqs = sreqs or []
                if sreqs:
                    items = []
                    for r in sreqs:
                        if not isinstance(r, dict):
                            continue
                        prov = r.get("provider")
                        mod = r.get("module")
                        sym = r.get("symbol")
                        kws = r.get("required_kwargs", []) or []
                        if isinstance(prov, str) and isinstance(mod, str) and isinstance(sym, str):
                            kwtxt = ",".join([k for k in kws if isinstance(k, str)])
                            if kwtxt:
                                items.append(f"{prov}:{mod}.{sym}({kwtxt})")
                            else:
                                items.append(f"{prov}:{mod}.{sym}()")
                    parts.append(f"SYM=[{'; '.join(items)}]")
                else:
                    parts.append("SYM=[]")

                lines.append(f"- versions: {', '.join(vlist_sorted)}")
                lines.append(f"  - " + " | ".join(parts))

        # hidden rules compact
        lines.append("\n## Hidden rules")
        hidden_rules = spec.get("hidden_rules", []) or []
        if not hidden_rules:
            lines.append("- <none>")
        else:
            for r in hidden_rules:
                if not isinstance(r, dict):
                    continue
                scopes = r.get("scope", ["run.py"])
                when = r.get("when", []) or []
                err = r.get("error", "RuntimeError: unknown.")
                lines.append(f"- scope={scopes} when={when} -> {err}")

        return "\n".join(lines)

    def _build_rules_nl_deps_only(self, spec: Dict[str, Any]) -> str:
        """
        依赖规则的“简化自然语言”：
        - 不包含脚本/py 文件调用路径
        - 只描述 base libs（含目标版本）与每个包每个版本的依赖/冲突
        """
        lines: List[str] = []
        lines.append("## Dependency rules (deps-only)")

        base_libs = spec.get("base_libs", []) or []
        base_targets = spec.get("base_lib_target_versions", {}) or {}
        base_avail = spec.get("base_lib_available_versions", {}) or {}

        lines.append("### Base libraries")
        if not base_libs:
            lines.append("- <none>")
        else:
            for bl in base_libs:
                tv = base_targets.get(bl, "<unknown>")
                av = base_avail.get(bl, [])
                av_txt = ", ".join(av) if isinstance(av, list) else "<unknown>"
                lines.append(f"- {bl} target={tv} (available: {av_txt})")

        def fmt_cons(cons_json: List[List[str]]) -> str:
            if not cons_json:
                return "<none>"
            return ",".join(f"{op}{vs}" for op, vs in cons_json)

        pkgs = spec.get("packages", {}) or {}
        vdeps = spec.get("version_dependencies", {}) or {}
        vconfs = spec.get("version_conflicts", {}) or {}

        lines.append("\n### Packages")
        for pkg in sorted(pkgs.keys()):
            pinfo = pkgs.get(pkg, {}) or {}
            vers = pinfo.get("versions", []) or []
            prio = pinfo.get("priority", "?")
            lines.append(f"\n- {pkg} (priority={prio}) versions: {', '.join(vers) if vers else '<none>'}")

            vdep_map = vdeps.get(pkg, {}) if isinstance(vdeps, dict) else {}
            vconf_map = vconfs.get(pkg, {}) if isinstance(vconfs, dict) else {}

            for ver in vers:
                lines.append(f"  - {pkg}=={ver}")

                dep_list = vdep_map.get(ver, []) if isinstance(vdep_map, dict) else []
                dep_list = dep_list or []
                if dep_list:
                    lines.append("    deps:")
                    for dep_pkg, cons_json in dep_list:
                        lines.append(f"      - {dep_pkg} {fmt_cons(cons_json)}")
                else:
                    lines.append("    deps: <none>")

                conf_list = vconf_map.get(ver, []) if isinstance(vconf_map, dict) else []
                conf_list = conf_list or []
                if conf_list:
                    lines.append("    conflicts:")
                    for item in conf_list:
                        if isinstance(item, list) and len(item) >= 3:
                            conf_pkg, op, conf_ver = item[0], item[1], item[2]
                            lines.append(f"      - {conf_pkg} {op} {conf_ver}")
                # conflicts 为空则不写（更紧凑）

        return "\n".join(lines)

    # ---------- 内部：版本生成 ----------

    def _generate_continuous_versions_for_package(self) -> List[Version]:
        """
        为单个包生成“连续”的 semantic 版本：
        例如：
          major=0: 0.0, 0.1, 0.2
          major=1: 1.0, 1.1
        """
        while True:
            versions: List[Version] = []
            num_major = self.rng.randint(1, 3)
            major_start = self.rng.choice([0, 1])

            for i in range(num_major):
                major = major_start + i
                minor_count = self.rng.randint(1, 4)
                for mn in range(minor_count):
                    versions.append((major, mn))

            if self.min_versions <= len(versions) <= self.max_versions:
                versions.sort()
                return versions

    def _sample_packages(self) -> Dict[str, Dict[str, Any]]:
        pkgs: Dict[str, Dict[str, Any]] = {}
        for i in range(self.num_packages):
            name = f"pkg{i}"
            versions = self._generate_continuous_versions_for_package()
            r = self.rng.random()
            if r < 0.25:
                priority = "high"
            elif r < 0.75:
                priority = "medium"
            else:
                priority = "low"
            pkgs[name] = {
                "versions": versions,
                "priority": priority,
            }
        return pkgs

    # ---------- 内部：ground-truth 解 ----------

    def _sample_solution(self, packages: Dict[str, Any]) -> Dict[str, Any]:
        py = self.rng.choice(self.python_versions)
        installed: Dict[str, Version] = {}
        for pkg, info in packages.items():
            vers = info["versions"]
            if len(vers) > 1 and self.rng.random() < 0.6:
                candidates = vers[:-1]  # 偏向非最高版本
                installed[pkg] = self.rng.choice(candidates)
            else:
                installed[pkg] = self.rng.choice(vers)
        return {"python_version": py, "installed": installed}

    # ---------- 内部：项目级 Python 约束 ----------

    def _derive_project_python_constraint(self, solution: Dict[str, Any]) -> Tuple[str, Version]:
        py = solution["python_version"]
        candidates = [v for v in self.python_versions if compare_versions(v, py) <= 0]
        if not candidates:
            return (">=", py)
        target = self.rng.choice(candidates)
        return (">=", target)

    # ---------- 内部：范围生成工具 ----------

    def _make_range_around_solution(
        self,
        sol_ver: Version,
        all_versions: List[Version],
        strict_prob: float,
    ) -> ConstraintList:
        """
        以 sol_ver 为中心，构造一个包含 sol_ver 的版本范围。
        strict_prob 越高，双边范围 (>=x,<=y) 越多。
        """
        idx = all_versions.index(sol_ver)
        n = len(all_versions)
        use_strict = self.rng.random() < strict_prob

        if n == 1:
            return [(">=", sol_ver)]

        # 单边约束
        if not use_strict:
            if self.rng.random() < 0.5:
                low_idx = self.rng.randint(0, idx)
                low = all_versions[low_idx]
                c = [(">=", low)]
            else:
                high_idx = self.rng.randint(idx, n - 1)
                high = all_versions[high_idx]
                if self.rng.random() < 0.5:
                    c = [("<=", high)]
                else:
                    if high_idx + 1 < n:
                        next_v = all_versions[high_idx + 1]
                        c = [("<", next_v)]
                    else:
                        c = [("<=", high)]
            return normalize_constraints(c, all_versions)

        # 双边约束
        low_idx = self.rng.randint(0, idx)
        high_idx = self.rng.randint(idx, n - 1)
        low = all_versions[low_idx]
        high = all_versions[high_idx]
        constraints: ConstraintList = [(">=", low)]

        if compare_versions(high, sol_ver) == 0:
            constraints.append(("<=", high))
        elif compare_versions(high, sol_ver) > 0:
            if self.rng.random() < 0.5:
                constraints.append(("<", high))
            else:
                constraints.append(("<=", high))
        else:
            constraints = [(">=", sol_ver)]

        return normalize_constraints(constraints, all_versions)

    # ---------- 内部：项目级包需求（范围） ----------

    def _derive_project_package_requirements(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
    ) -> List[Tuple[str, ConstraintList]]:
        installed = solution["installed"]
        proj: List[Tuple[str, ConstraintList]] = []

        pkg_list = list(installed.keys())
        self.rng.shuffle(pkg_list)

        k = max(1, len(pkg_list) // 2)
        chosen = pkg_list[:k]

        for pkg in chosen:
            sol_ver = installed[pkg]
            all_versions = packages[pkg]["versions"]
            constraints = self._make_range_around_solution(
                sol_ver, all_versions, self.project_range_strict_prob
            )
            proj.append((pkg, constraints))

        return proj

    def _sample_range_for_dependency(
        self,
        sol_ver: Version,
        all_versions: List[Version],
        strict_prob: float,
    ) -> ConstraintList:
        return self._make_range_around_solution(sol_ver, all_versions, strict_prob)

    # ---------- 内部：依赖 + 包级 Python 要求 ----------

    def _generate_dependencies_and_pyreqs(
        self,
        packages: Dict[str, Any],
        topo_order: List[str],
        solution: Dict[str, Any],
        version_dependencies: Dict[Tuple[str, Version], List[Tuple[str, ConstraintList]]],
        version_requires_python: Dict[Tuple[str, Version], Tuple[str, Version]],
    ) -> None:
        installed = solution["installed"]
        py = solution["python_version"]
        pos = {p: i for i, p in enumerate(topo_order)}

        for pkg, info in packages.items():
            for v in info["versions"]:
                key = (pkg, v)
                deps: List[Tuple[str, ConstraintList]] = []

                earlier = [p for p in topo_order if pos[p] < pos[pkg]]
                if earlier and self.rng.random() < self.dep_edge_prob:
                    # 高/中优先级包更“粘连”，依赖更多（强组合）
                    priority = packages[pkg]["priority"]
                    cap = self.max_deps_per_version
                    if priority == "medium":
                        cap += 1
                    elif priority == "high":
                        cap += 2
                    cap = min(cap, len(earlier))
                    if cap <= 0:
                        k_dep = 0
                    else:
                        k_dep = self.rng.randint(1, cap)
                    dep_candidates = self.rng.sample(earlier, k=k_dep)
                    for dep_pkg in dep_candidates:
                        dep_versions = packages[dep_pkg]["versions"]
                        sol_ver = installed[dep_pkg]
                        c = self._sample_range_for_dependency(
                            sol_ver, dep_versions, self.dep_range_strict_prob
                        )
                        deps.append((dep_pkg, c))

                version_dependencies[key] = deps

                # 包级 Python 约束
                if self.rng.random() < 0.3:
                    py_candidates = [
                        v_py
                        for v_py in self.python_versions
                        if compare_versions(v_py, py) <= 0
                    ]
                    if py_candidates:
                        target_py = self.rng.choice(py_candidates)
                        version_requires_python[key] = (">=", target_py)

    # ---------- v5: 底座库（强共享依赖）+ 一致性约束 ----------
    def _choose_base_libs(self, packages: Dict[str, Any], topo_order: List[str]) -> List[str]:
        """
        选择 1~2 个“底座库”（类似 numpy/openssl/torch），让大量包强依赖它们。

        v5 修正：base libs 必须选在 topo_order 的最前面（根节点），否则会出现
        “普通依赖 + base 注入”共同作用下的循环依赖：
          A(作为 base) 依赖 B（普通依赖允许，因为 B 在 A 之前）
          同时 B 被注入依赖 A（base 注入不看 topo）
        这会导致安装时出现 A<->B 互相要求。
        """
        order = [p for p in topo_order if p in packages]
        if not order:
            order = list(packages.keys())
        # 选前 k 个，保证它们没有“普通依赖”（root），从而避免环
        k = 2 if len(order) >= 6 else 1
        return order[:k]

    def _inject_base_lib_dependencies(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        base_libs: List[str],
        version_dependencies: Dict[Tuple[str, Version], List[Tuple[str, ConstraintList]]],
    ) -> None:
        """
        强化组合依赖：让绝大多数包/版本都依赖 base_libs，
        且每个 base_lib 的约束不同，最终形成交集（组合型规则）。

        生成原则：永远不破坏 ground-truth solution（约束必须包含 solution 版本）。
        """
        if not base_libs:
            return
        installed = solution["installed"]

        for pkg, info in packages.items():
            # 底座库自己不依赖自己
            if pkg in base_libs:
                continue

            for v in info["versions"]:
                key = (pkg, v)
                deps = version_dependencies.get(key, [])

                for bl in base_libs:
                    bl_vers = packages[bl]["versions"]
                    bl_sol = installed[bl]

                    # 对多数包注入强依赖（高概率）
                    if self.rng.random() < self.base_dep_prob:
                        # 用“围绕 solution 的区间”制造组合交集
                        cons = self._make_range_around_solution(
                            bl_sol, bl_vers, strict_prob=self.base_strict_prob
                        )
                        # 再随机加一个“同侧”约束，让不同包对同一底座库形成不同切片
                        # 但必须仍包含 bl_sol
                        if self.rng.random() < self.base_slice_extra_prob and len(bl_vers) >= 3:
                            idx = bl_vers.index(bl_sol)
                            if self.rng.random() < 0.5 and idx + 1 < len(bl_vers):
                                # 禁止过高版本（常见：ABI/行为变更）
                                upper = bl_vers[idx + 1]
                                cons2 = normalize_constraints(cons + [("<", upper)], bl_vers)
                                if cons2 and version_satisfies_constraints(bl_sol, cons2):
                                    cons = cons2
                            elif idx - 1 >= 0:
                                # 禁止过低版本（缺特性）
                                lower = bl_vers[idx - 1]
                                cons2 = normalize_constraints(cons + [(">", lower)], bl_vers)
                                if cons2 and version_satisfies_constraints(bl_sol, cons2):
                                    cons = cons2

                        # 如果 deps 里已经存在 bl（例如之前普通依赖就采样到了 base lib），
                        # 就把约束做交集合并成一条，避免重复边。
                        merged = False
                        for i, (dep_pkg, existing) in enumerate(deps):
                            if dep_pkg != bl:
                                continue
                            merged_cons = intersect_constraints(existing, cons, bl_vers)
                            # 理论上 merged_cons 一定包含 bl_sol；若为空则保守不覆盖
                            if merged_cons and version_satisfies_constraints(bl_sol, merged_cons):
                                deps[i] = (bl, merged_cons)
                            else:
                                # 保底：用 normalize 合并（仍尽量保持可解）
                                tmp = normalize_constraints(existing + cons, bl_vers)
                                if tmp and version_satisfies_constraints(bl_sol, tmp):
                                    deps[i] = (bl, tmp)
                            merged = True
                            break
                        if not merged:
                            deps.append((bl, cons))

                version_dependencies[key] = deps

    def _generate_consistency_rules(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        base_libs: List[str],
    ) -> List[Dict[str, Any]]:
        """
        生成跨包一致性规则（强组合）：
        - same_major_with_anchor: 一组包必须与某个底座库同主版本（模拟 ABI 断代）
        - same_minor_with_anchor: 少量包要求同 minor（更强耦合）
        """
        if not base_libs:
            return []

        installed = solution["installed"]
        # 只从“在 ground-truth 解里已经满足一致性”的包里抽组，
        # 保证至少存在一个可行解（避免出现你观察到的无解 case）。
        # 另外：即使包的全部版本都只有 major=0，也不会被拉进 major=1 的组里。
        anchor = base_libs[0]
        a_ver = installed[anchor]
        candidates_major = [
            p
            for p in packages.keys()
            if p not in base_libs
            and installed[p][0] == a_ver[0]
        ]
        self.rng.shuffle(candidates_major)
        if not candidates_major:
            return []

        rules: List[Dict[str, Any]] = []

        # 规则 1：大组 same_major（覆盖面大）
        group_size = max(2, min(len(candidates_major), max(3, len(packages) // 3)))
        group = candidates_major[:group_size]
        rules.append(
            {
                "type": "same_major_with_anchor",
                "anchor": anchor,
                "packages": group,
                "error": f"RuntimeError: ABI mismatch detected between '{anchor}' and dependent packages.",
            }
        )

        # 规则 2：小组 same_minor（更强，但覆盖面小）
        if len(base_libs) > 1:
            anchor2 = base_libs[1]
            a2_ver = installed[anchor2]
        else:
            anchor2 = anchor
            a2_ver = a_ver

        candidates_minor = [
            p
            for p in packages.keys()
            if p not in base_libs
            and installed[p][0] == a2_ver[0]
            and installed[p][1] == a2_ver[1]
        ]
        self.rng.shuffle(candidates_minor)
        if len(candidates_minor) >= 2:
            small = candidates_minor[: min(3, len(candidates_minor))]
            rules.append(
                {
                    "type": "same_minor_with_anchor",
                    "anchor": anchor2,
                    "packages": small,
                    "error": f"RuntimeError: tightly-coupled components are out of sync with '{anchor2}'.",
                }
            )

        return rules

    # ---------- 内部：基础冲突（非解版本） ----------

    def _generate_base_conflicts(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        version_conflicts: Dict[Tuple[str, Version], List[Tuple[str, str, Version, str]]],
    ) -> None:
        installed = solution["installed"]
        pkg_list = list(packages.keys())

        for pkg, info in packages.items():
            for v in info["versions"]:
                key = (pkg, v)
                version_conflicts[key] = []
                if installed[pkg] == v:
                    continue

                if self.rng.random() < 0.25:
                    other_pkg = self.rng.choice(pkg_list)
                    if other_pkg == pkg:
                        continue
                    other_versions = packages[other_pkg]["versions"]
                    bad_versions = [vv for vv in other_versions if vv != installed[other_pkg]]
                    if not bad_versions:
                        continue
                    conf_ver = self.rng.choice(bad_versions)
                    msg = (
                        f"{pkg}=={format_pkg_version(v)} is not compatible with "
                        f"{other_pkg}=={format_pkg_version(conf_ver)}"
                    )
                    version_conflicts[key].append((other_pkg, "==", conf_ver, msg))

    # ---------- 内部：隐式项目依赖 ----------

    def _generate_implicit_project_deps(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        proj_pkg_reqs: List[Tuple[str, ConstraintList]],
    ) -> List[Tuple[str, ConstraintList]]:
        installed = solution["installed"]
        pkg_list = list(packages.keys())
        implicit: List[Tuple[str, ConstraintList]] = []

        proj_pkgs = {p for (p, _) in proj_pkg_reqs}
        candidates = [p for p in pkg_list if p not in proj_pkgs]
        self.rng.shuffle(candidates)

        if not candidates:
            return implicit

        k = max(1, len(candidates) // 3)
        chosen = candidates[:k]
        for p in chosen:
            sol_ver = installed[p]
            all_versions = packages[p]["versions"]
            constraints = self._make_range_around_solution(
                sol_ver, all_versions, self.implicit_range_strict_prob
            )
            implicit.append((p, constraints))

        return implicit

    # ---------- 内部：side-effects（v3: 非单调） ----------

    def _generate_side_effects(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        version_dependencies: Dict[Tuple[str, Version], List[Tuple[str, ConstraintList]]],
    ) -> Dict[Tuple[str, Version], List[Tuple[str, str, ConstraintList]]]:
        """
        side_effects 用于制造“非单调”的环境变化：
        - ensure：仅在不满足约束时自动补齐到最高可行版本（与旧逻辑一致）
        - force_high：无论当前是否满足约束，都强制到最高可行版本（可能破坏其它包的期望）
        - force_low：无论当前是否满足约束，都强制到最低可行版本（制造降级陷阱）
        - pin：强制钉死到 ground-truth 的精确版本（==solution）

        设计原则：不破坏 ground-truth 解（solution），但会让“默认装最新/局部修补”
        的策略更容易翻车，从而提高规划/回溯难度。
        """
        side_effects: Dict[Tuple[str, Version], List[Tuple[str, str, ConstraintList]]] = {}

        installed_sol: Dict[str, Version] = solution["installed"]

        for pkg, info in packages.items():
            if info["priority"] != "high":
                continue
            for v in info["versions"]:
                key = (pkg, v)
                effects: List[Tuple[str, str, ConstraintList]] = []
                deps = version_dependencies.get(key, [])
                for dep_pkg, constr in deps:
                    # 只对依赖边制造 side-effect，避免无脑扩散导致不可解
                    if self.rng.random() >= 0.75:
                        continue

                    r = self.rng.random()
                    if r < 0.40:
                        eff_type = "ensure"
                        eff_cons = constr
                    elif r < 0.65:
                        eff_type = "force_high"
                        eff_cons = constr
                    elif r < 0.85:
                        eff_type = "force_low"
                        eff_cons = constr
                    else:
                        # pin 到 ground-truth 精确版本（确保可解）
                        eff_type = "pin"
                        eff_cons = [("==", installed_sol[dep_pkg])]

                    effects.append((eff_type, dep_pkg, eff_cons))
                if effects:
                    side_effects[key] = effects

        return side_effects

    # ---------- 内部：高版本冲突（不破坏解，v3: 更范围化） ----------

    def _inject_high_version_conflicts(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        version_conflicts: Dict[Tuple[str, Version], List[Tuple[str, str, Version, str]]],
    ) -> None:
        installed = solution["installed"]
        pkg_list = list(packages.keys())
        self.rng.shuffle(pkg_list)
        num_target = max(1, int(len(pkg_list) * self.high_version_conflict_ratio))
        target_pkgs = pkg_list[:num_target]

        anchor_pkgs = list(installed.keys())
        if not anchor_pkgs:
            return

        for pkg in target_pkgs:
            vers = packages[pkg]["versions"]
            v_max = max(vers)
            sol_ver = installed[pkg]
            if v_max == sol_ver:
                continue

            key = (pkg, v_max)
            if key not in version_conflicts:
                version_conflicts[key] = []

            # 修复：禁止与自身产生冲突（否则会出现 pkg7 与 pkg7 的冲突）
            anchor_candidates = [p for p in anchor_pkgs if p != pkg]
            if not anchor_candidates:
                continue
            anchor_pkg = self.rng.choice(anchor_candidates)
            anchor_ver = installed[anchor_pkg]

            # 让冲突更“范围化”：惩罚 anchor_pkg 的高版本（常见默认装最新陷阱），
            # 但不影响 ground-truth（anchor_ver 一定不触发）。
            anchor_vers = sorted(packages[anchor_pkg]["versions"])
            op = "=="
            boundary = anchor_ver
            if anchor_ver in anchor_vers:
                idx = anchor_vers.index(anchor_ver)
                if idx + 1 < len(anchor_vers) and self.rng.random() < 0.7:
                    op = ">="
                    boundary = anchor_vers[idx + 1]
                else:
                    op = "=="
                    boundary = anchor_ver

            msg = (
                f"{pkg}=={format_pkg_version(v_max)} is not compatible with "
                f"{anchor_pkg} {op} {format_pkg_version(boundary)} (high-version penalty)"
            )
            version_conflicts[key].append((anchor_pkg, op, boundary, msg))

    # ---------- 内部：合成 API surface（模块/符号/签名） ----------
    def _generate_api_surface(
        self,
        packages: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        生成 JSON-safe 的 api_surface：
        api_surface[pkg][ver_str] = {
          "modules": [module, ...],
          "exports": {module: [symbol, ...]},
          "signatures": {"module:symbol": [kw, ...]}
        }

        版本演化：
        - exports 随版本递增（新增 symbol）
        - signatures 大多随版本递增（新增 kw），少量在最高版本“破坏性变化”（删除 kw）
        """
        api_surface: Dict[str, Dict[str, Any]] = {}

        for pkg, info in packages.items():
            vers: List[Version] = sorted(info["versions"])
            if not vers:
                continue

            modules = [f"{pkg}.core", f"{pkg}.io"]
            # 为每个 module 准备一个“基础符号序列”
            base_symbols_by_module: Dict[str, List[str]] = {}
            for m in modules:
                # 稍微打散，保证不同包的符号组合不同
                pool = self._symbol_pool[:]
                self.rng.shuffle(pool)
                base_symbols_by_module[m] = pool[: self.rng.randint(4, 7)]

            pkg_map: Dict[str, Any] = {}
            for idx, v in enumerate(vers):
                v_str = format_pkg_version(v)
                exports: Dict[str, List[str]] = {}
                signatures: Dict[str, List[str]] = {}

                for m in modules:
                    base_syms = base_symbols_by_module[m]
                    # exports 递增：低版本少，高版本多
                    k = min(len(base_syms), 2 + (idx % 3))
                    exp_syms = base_syms[:k]
                    exports[m] = exp_syms

                    for s in exp_syms:
                        # 生成签名 kw（默认 1~3 个）
                        kw_pool = self._kw_pool[:]
                        self.rng.shuffle(kw_pool)
                        kws = kw_pool[: self.rng.randint(1, 3)]

                        # 版本递增：中高版本倾向增加 1 个 kw
                        if idx >= 1 and self.rng.random() < 0.7:
                            extra = kw_pool[self.rng.randint(3, min(6, len(kw_pool) - 1))]
                            if extra not in kws:
                                kws.append(extra)

                        # 最高版本：少量破坏性变化（删除一个 kw），制造“装太新也会炸”
                        if idx == len(vers) - 1 and len(kws) >= 2 and self.rng.random() < 0.35:
                            kws.pop(0)

                        signatures[f"{m}:{s}"] = kws

                pkg_map[v_str] = {
                    "modules": modules,
                    "exports": exports,
                    "signatures": signatures,
                }

            api_surface[pkg] = pkg_map

        return api_surface

    def _generate_project_calls(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        api_surface: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        project_calls: run.py 会执行的一组“调用”（合成），用于决定成功与否。
        每条 call 都是 JSON-safe：
          {"provider": pkg, "module": module, "symbol": symbol, "required_kwargs": [kw, ...]}
        """
        installed = solution["installed"]
        pkg_list = list(packages.keys())
        self.rng.shuffle(pkg_list)
        k = max(2, len(pkg_list) // 3)
        chosen = pkg_list[:k]

        calls: List[Dict[str, Any]] = []
        for p in chosen:
            sol_v = installed[p]
            v_str = format_pkg_version(sol_v)
            p_api = api_surface.get(p, {}).get(v_str, {})
            exports = p_api.get("exports", {})
            if not exports:
                continue

            module = self.rng.choice(list(exports.keys()))
            syms = exports.get(module, [])
            if not syms:
                continue
            symbol = self.rng.choice(syms)

            sig = p_api.get("signatures", {}).get(f"{module}:{symbol}", [])
            required_kwargs: List[str] = []
            # 选择 0~1 个 kw 作为调用参数（不泄露版本，但可产生 TypeError 线索）
            if sig and self.rng.random() < 0.75:
                required_kwargs = [sig[-1]]

            calls.append(
                {
                    "provider": p,
                    "module": module,
                    "symbol": symbol,
                    "required_kwargs": required_kwargs,
                }
            )

        # 保底：至少 2 条
        if len(calls) < 2 and pkg_list:
            p = pkg_list[0]
            sol_v = installed[p]
            v_str = format_pkg_version(sol_v)
            p_api = api_surface.get(p, {}).get(v_str, {})
            exports = p_api.get("exports", {})
            if exports:
                module = self.rng.choice(list(exports.keys()))
                syms = exports.get(module, [])
                if syms:
                    symbol = self.rng.choice(syms)
                    calls.append(
                        {
                            "provider": p,
                            "module": module,
                            "symbol": symbol,
                            "required_kwargs": [],
                        }
                    )

        return calls

    # ---------- v5: 多脚本/层级 repo（calls 分配到不同入口） ----------
    def _generate_repo_scripts(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        api_surface: Dict[str, Dict[str, Any]],
        symbol_requires: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        生成 scripts + entrypoints（JSON-safe）：
        - scripts[path] = {"calls": [...], "imports": [pkg,...]}
        - entrypoints 是 run.py 执行的脚本顺序
        """
        script_paths = [
            "core/smoke.py",
            "data/check_io.py",
            "model/smoke_model.py",
            "train/run_train.py",
            "eval/run_eval.py",
        ]
        entrypoints = script_paths[:]

        installed = solution["installed"]
        pkg_list = list(packages.keys())
        self.rng.shuffle(pkg_list)

        # 生成一组“全项目 calls”，再切分到各脚本
        all_calls: List[Dict[str, Any]] = []
        target_n = max(8, self.num_packages // 2 + 4)
        guard = 0
        while len(all_calls) < target_n and guard < target_n * 10:
            guard += 1
            p = self.rng.choice(pkg_list)
            sol_v = installed[p]
            v_str = format_pkg_version(sol_v)
            p_api = api_surface.get(p, {}).get(v_str, {})
            exports = p_api.get("exports", {})
            if not exports:
                continue
            module = self.rng.choice(list(exports.keys()))
            syms = exports.get(module, [])
            if not syms:
                continue
            symbol = self.rng.choice(syms)
            sig = p_api.get("signatures", {}).get(f"{module}:{symbol}", [])
            required_kwargs: List[str] = []
            if sig and self.rng.random() < 0.6:
                required_kwargs = [sig[-1]]
            all_calls.append(
                {
                    "provider": p,
                    "module": module,
                    "symbol": symbol,
                    "required_kwargs": required_kwargs,
                }
            )

        splits: Dict[str, List[Dict[str, Any]]] = {
            "core/smoke.py": all_calls[:2],
            "data/check_io.py": all_calls[2:4],
            "model/smoke_model.py": all_calls[4:6],
            "train/run_train.py": all_calls[6:8],
            "eval/run_eval.py": all_calls[8:],
        }

        callers = list(symbol_requires.keys())
        self.rng.shuffle(callers)

        scripts: Dict[str, Any] = {}
        for sp in script_paths:
            calls = splits.get(sp, [])
            imports = set()
            for c in calls:
                p = c.get("provider")
                if isinstance(p, str):
                    imports.add(p)

            extra_n = 1
            if sp.startswith("train/"):
                extra_n = 3
            elif sp.startswith("eval/"):
                extra_n = 2
            elif sp.startswith("data/") or sp.startswith("model/"):
                extra_n = 2

            for _ in range(extra_n):
                if callers:
                    imports.add(self.rng.choice(callers))

            scripts[sp] = {
                "calls": calls,
                "imports": sorted(list(imports)),
            }

        return scripts, entrypoints

    def _generate_symbol_requires(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        api_surface: Dict[str, Dict[str, Any]],
        version_dependencies: Dict[Tuple[str, Version], List[Tuple[str, ConstraintList]]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        symbol_requires[pkg][ver_str] = [
          {"provider": dep_pkg, "module": "...", "symbol": "...", "required_kwargs": [...]},
          ...
        ]

        生成策略：沿用 version_dependencies 的依赖边，但把“兼容性”编码成
        provider 的 module/symbol/kwargs 是否满足。
        """
        installed = solution["installed"]
        out: Dict[str, Dict[str, Any]] = {}

        for (pkg, v), deps in version_dependencies.items():
            if not deps:
                continue
            v_str = format_pkg_version(v)
            reqs: List[Dict[str, Any]] = []

            for dep_pkg, _constraints in deps:
                # 选择 provider 在 ground-truth 版本下的一个 module/symbol 作为运行时需求
                dep_sol = installed[dep_pkg]
                dep_sol_str = format_pkg_version(dep_sol)
                dep_api = api_surface.get(dep_pkg, {}).get(dep_sol_str, {})
                exports = dep_api.get("exports", {})
                if not exports:
                    continue
                module = self.rng.choice(list(exports.keys()))
                syms = exports.get(module, [])
                if not syms:
                    continue
                symbol = self.rng.choice(syms)
                sig = dep_api.get("signatures", {}).get(f"{module}:{symbol}", [])

                required_kwargs: List[str] = []
                if sig and self.rng.random() < 0.7:
                    required_kwargs = [sig[-1]]

                reqs.append(
                    {
                        "provider": dep_pkg,
                        "module": module,
                        "symbol": symbol,
                        "required_kwargs": required_kwargs,
                    }
                )

            if reqs:
                pkg_map = out.setdefault(pkg, {})
                pkg_map[v_str] = reqs

        return out

    # ---------- 内部：隐藏规则（组合触发的黑盒故障，不破坏解） ----------
    def _inject_hidden_rules(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        api_surface: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        hidden_rules 用于模拟现实里“只有症状，没有版本提示”的情况：
        - 规则只在特定组合/精确版本同时出现时触发
        - 报错不包含任何版本信息
        - 必须保证 ground-truth 解不会触发
        """
        installed = solution["installed"]
        pkg_list = list(packages.keys())
        if len(pkg_list) < 3:
            return []

        rules: List[Dict[str, Any]] = []

        # 尝试注入 1~2 条规则，避免过度黑盒导致不可解
        num_rules = 1 if self.rng.random() < 0.7 else 2
        tries = 0
        while len(rules) < num_rules and tries < 20:
            tries += 1
            a, b = self.rng.sample(pkg_list, 2)
            a_vers = packages[a]["versions"]
            b_vers = packages[b]["versions"]

            # v4：把“精确点触发”升级为“范围触发”，让 agent 更难从症状直接定位。
            # 对每个包选择一个非解版本 a_bad，并构造单边约束来包含 a_bad 但排除 solution。
            a_sol = installed[a]
            b_sol = installed[b]
            a_choices = [v for v in a_vers if v != a_sol]
            b_choices = [v for v in b_vers if v != b_sol]
            if not a_choices or not b_choices:
                continue

            a_bad = self.rng.choice(a_choices)
            b_bad = self.rng.choice(b_choices)

            def make_one_sided_range(sol: Version, bad: Version) -> List[List[str]]:
                # 输出 JSON-safe constraints: [[op, "x.y"]]
                if compare_versions(bad, sol) > 0:
                    return [[">=", format_pkg_version(bad)]]
                else:
                    return [["<=", format_pkg_version(bad)]]

            when = [
                [a, make_one_sided_range(a_sol, a_bad)],
                [b, make_one_sided_range(b_sol, b_bad)],
            ]

            # 生成“可推理”的对象级错误，不包含版本数字
            # 尝试从 a 的 ground-truth API 里抽一个 module/symbol
            a_api = api_surface.get(a, {}).get(format_pkg_version(a_sol), {})
            exports = a_api.get("exports", {}) or {f"{a}.core": ["load"]}
            mod = self.rng.choice(list(exports.keys()))
            syms = exports.get(mod, []) or ["load"]
            sym = self.rng.choice(syms)

            rules.append(
                {
                    "when": when,
                    # v5: 让 hidden rule 更像“后期才触发”的坑（默认偏 eval/train）
                    "scope": self.rng.choice(
                        [["eval/run_eval.py"], ["train/run_train.py"], ["run.py"]]
                    ),
                    "error": self.rng.choice(
                        [
                            f"ImportError: cannot import name '{sym}' from '{mod}'.",
                            f"AttributeError: module '{mod}' has no attribute '{sym}'.",
                            f"TypeError: {sym}() got an unexpected keyword argument 'axis'.",
                            "RuntimeError: extension module initialization failed.",
                            "RuntimeError: ABI mismatch detected at runtime.",
                        ]
                    ),
                }
            )

        return rules

    # ---------- 内部：fork-point 依赖 ----------

    def _inject_fork_point_dependencies(
        self,
        packages: Dict[str, Any],
        topo_order: List[str],
        solution: Dict[str, Any],
        version_dependencies: Dict[Tuple[str, Version], List[Tuple[str, ConstraintList]]],
    ) -> None:
        # v5 修复：fork-point 注入可能造成依赖环（A->B 且 B->A）。
        # 这里强制所有注入的“dep_pkg -> core_pkg”边都从 topo_order 的后置包指向前置包，
        # 与 _generate_dependencies_and_pyreqs 保持同向（后 -> 前），从而保证整体无环。
        pkg_list = list(packages.keys())
        pos = {p: i for i, p in enumerate(topo_order)}
        self.rng.shuffle(pkg_list)
        num_core = max(1, int(len(pkg_list) * self.fork_point_ratio))
        core_candidates = pkg_list[:num_core]

        for core_pkg in core_candidates:
            vers = sorted(packages[core_pkg]["versions"])
            if len(vers) < 3:
                continue

            sol_ver = solution["installed"][core_pkg]
            if sol_ver not in vers:
                continue

            idx = vers.index(sol_ver)
            lower_candidates = vers[:idx] if idx > 0 else []
            higher_candidates = vers[idx + 1 :] if idx + 1 < len(vers) else []

            if not lower_candidates or not higher_candidates:
                continue

            v_low = self.rng.choice(lower_candidates)
            v_high = self.rng.choice(higher_candidates)

            other_pkgs_after = [
                p for p in pkg_list
                if p != core_pkg and pos.get(p, -1) > pos.get(core_pkg, -1)
            ]
            if len(other_pkgs_after) < 2:
                continue

            depA_pkg, depB_pkg = self.rng.sample(other_pkgs_after, 2)

            def add_or_merge_dep(
                dep_pkg_name: str,
                dep_ver: Version,
                new_constraints: ConstraintList,
            ):
                key = (dep_pkg_name, dep_ver)
                deps = version_dependencies.get(key, [])
                all_vers = packages[core_pkg]["versions"]

                new_norm = normalize_constraints(new_constraints, all_vers)
                if not new_norm:
                    return

                for i, (existing_dep_pkg, existing_constraints) in enumerate(deps):
                    if existing_dep_pkg == core_pkg:
                        merged = intersect_constraints(existing_constraints, new_norm, all_vers)
                        if not merged:
                            return
                        deps[i] = (core_pkg, merged)
                        version_dependencies[key] = deps
                        return

                merged = normalize_constraints(new_norm, all_vers)
                if not merged:
                    return
                deps.append((core_pkg, merged))
                version_dependencies[key] = deps

            # depA_pkg 一些非解版本 -> core_pkg <= v_low
            dep_vers_A = packages[depA_pkg]["versions"]
            sol_depA = solution["installed"][depA_pkg]
            altA = [vv for vv in dep_vers_A if vv != sol_depA]
            if altA:
                chosenA = self.rng.choice(altA)
                add_or_merge_dep(depA_pkg, chosenA, [("<=", v_low)])

            # depB_pkg 一些非解版本 -> core_pkg >= v_high
            dep_vers_B = packages[depB_pkg]["versions"]
            sol_depB = solution["installed"][depB_pkg]
            altB = [vv for vv in dep_vers_B if vv != sol_depB]
            if altB:
                chosenB = self.rng.choice(altB)
                add_or_merge_dep(depB_pkg, chosenB, ([(">=", v_high)]))

    # ---------- 内部：生成 raw spec（tuple key） ----------

    def _generate_raw(self) -> Dict[str, Any]:
        packages = self._sample_packages()
        pkg_names = list(packages.keys())
        topo_order = pkg_names[:]
        self.rng.shuffle(topo_order)

        solution = self._sample_solution(packages)
        proj_py_req = self._derive_project_python_constraint(solution)
        proj_pkg_reqs = self._derive_project_package_requirements(packages, solution)

        version_requires_python: Dict[Tuple[str, Version], Tuple[str, Version]] = {}
        version_dependencies: Dict[Tuple[str, Version], List[Tuple[str, ConstraintList]]] = {}

        self._generate_dependencies_and_pyreqs(
            packages,
            topo_order,
            solution,
            version_dependencies,
            version_requires_python,
        )

        # v5: 选择底座库，并注入强共享依赖（组合型）
        base_libs = self._choose_base_libs(packages, topo_order)
        base_lib_target_versions = {
            bl: solution["installed"][bl] for bl in base_libs if bl in solution["installed"]
        }
        self._inject_base_lib_dependencies(
            packages, solution, base_libs, version_dependencies
        )

        # API surface & symbol rules（纯符号层，JSON-safe）
        api_surface = self._generate_api_surface(packages)
        symbol_requires = self._generate_symbol_requires(
            packages, solution, api_surface, version_dependencies
        )
        scripts, entrypoints = self._generate_repo_scripts(
            packages, solution, api_surface, symbol_requires
        )
        # 兼容旧字段：把全项目调用聚合起来当作 project_calls
        project_calls: List[Dict[str, Any]] = []
        for sp in entrypoints:
            project_calls.extend(scripts.get(sp, {}).get("calls", []) or [])

        version_conflicts: Dict[Tuple[str, Version], List[Tuple[str, str, Version, str]]] = {}
        self._generate_base_conflicts(packages, solution, version_conflicts)

        implicit_deps = self._generate_implicit_project_deps(packages, solution, proj_pkg_reqs)

        side_effects = self._generate_side_effects(packages, solution, version_dependencies)

        self._inject_high_version_conflicts(packages, solution, version_conflicts)
        self._inject_fork_point_dependencies(packages, topo_order, solution, version_dependencies)

        hidden_rules = self._inject_hidden_rules(packages, solution, api_surface)
        consistency_rules = self._generate_consistency_rules(
            packages, solution, base_libs
        )

        return {
            "python_versions": self.python_versions,
            "packages": packages,
            "version_requires_python": version_requires_python,
            "version_dependencies": version_dependencies,
            "version_conflicts": version_conflicts,
            "base_libs": base_libs,
            "base_lib_target_versions": base_lib_target_versions,
            "consistency_rules": consistency_rules,
            "project_requires_python": proj_py_req,
            "project_requires_packages": proj_pkg_reqs,
            "implicit_project_dependencies": implicit_deps,
            "side_effects": side_effects,
            # ---- v4: object-level runtime rules ----
            "api_surface": api_surface,
            "project_calls": project_calls,
            "symbol_requires": symbol_requires,
            # ---- v5: repo hierarchy ----
            "scripts": scripts,
            "entrypoints": entrypoints,
            # 默认关闭诊断命令：更难、更贴近真实
            "enable_diagnostics": False,
            "hidden_rules": hidden_rules,
        }

    # ---------- 内部：raw → JSON-safe spec ----------

    def _to_json_friendly(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        spec: Dict[str, Any] = {}

        # python_versions: List[str]
        spec["python_versions"] = [format_python_version(v) for v in raw["python_versions"]]

        # packages: {pkg: {"versions": [str], "priority": str}}
        pkgs: Dict[str, Any] = {}
        for name, info in raw["packages"].items():
            pkgs[name] = {
                "versions": [format_pkg_version(v) for v in info["versions"]],
                "priority": info["priority"],
            }
        spec["packages"] = pkgs

        # version_requires_python: {pkg: {ver_str: [op, py_str]}}
        vpy_json: Dict[str, Dict[str, Any]] = {}
        for (pkg, ver), (op, pyv) in raw["version_requires_python"].items():
            vmap = vpy_json.setdefault(pkg, {})
            vmap[format_pkg_version(ver)] = [op, format_python_version(pyv)]
        spec["version_requires_python"] = vpy_json

        # version_dependencies: {pkg: {ver_str: [[dep_pkg, [[op, ver_str], ...]], ...]}}
        vdep_json: Dict[str, Dict[str, Any]] = {}
        for (pkg, ver), deps in raw["version_dependencies"].items():
            vmap = vdep_json.setdefault(pkg, {})
            dep_list = []
            for dep_pkg, constraints in deps:
                cons_json = [[op, format_pkg_version(vv)] for op, vv in constraints]
                dep_list.append([dep_pkg, cons_json])
            vmap[format_pkg_version(ver)] = dep_list
        spec["version_dependencies"] = vdep_json

        # version_conflicts: {pkg: {ver_str: [[conf_pkg, op, conf_ver_str, msg], ...]}}
        vconf_json: Dict[str, Dict[str, Any]] = {}
        for (pkg, ver), confs in raw["version_conflicts"].items():
            vmap = vconf_json.setdefault(pkg, {})
            clist = []
            for conf_pkg, op, conf_ver, msg in confs:
                clist.append([conf_pkg, op, format_pkg_version(conf_ver), msg])
            vmap[format_pkg_version(ver)] = clist
        spec["version_conflicts"] = vconf_json

        # base libs
        spec["base_libs"] = raw.get("base_libs", [])
        # base_lib_target_versions: {pkg: "x.y"}
        bltv = {}
        for k, v in (raw.get("base_lib_target_versions", {}) or {}).items():
            if isinstance(k, str) and isinstance(v, tuple):
                bltv[k] = format_pkg_version(v)
        spec["base_lib_target_versions"] = bltv
        # available versions of base libs (for readability)
        blav = {}
        for bl in spec["base_libs"]:
            if bl in raw.get("packages", {}):
                vers = raw["packages"][bl]["versions"]
                blav[bl] = [format_pkg_version(x) for x in vers]
        spec["base_lib_available_versions"] = blav

        # project_requires_python: [op, py_str]
        op_py, pyv = raw["project_requires_python"]
        spec["project_requires_python"] = [op_py, format_python_version(pyv)]

        # project_requires_packages: [[pkg, [[op, ver_str], ...]], ...]
        prj_pkgs = []
        for pkg, constraints in raw["project_requires_packages"]:
            cons_json = [[op, format_pkg_version(vv)] for op, vv in constraints]
            prj_pkgs.append([pkg, cons_json])
        spec["project_requires_packages"] = prj_pkgs

        # implicit_project_dependencies: [[pkg, [[op, ver_str], ...]], ...]
        impl = []
        for pkg, constraints in raw["implicit_project_dependencies"]:
            cons_json = [[op, format_pkg_version(vv)] for op, vv in constraints]
            impl.append([pkg, cons_json])
        spec["implicit_project_dependencies"] = impl

        # side_effects: {pkg: {ver_str: [[eff_type, dep_pkg, [[op, ver_str], ...]], ...]}}
        se_json: Dict[str, Dict[str, Any]] = {}
        for (pkg, ver), effects in raw["side_effects"].items():
            vmap = se_json.setdefault(pkg, {})
            elist = []
            for eff_type, dep_pkg, constraints in effects:
                cons_json = [[op, format_pkg_version(vv)] for op, vv in constraints]
                elist.append([eff_type, dep_pkg, cons_json])
            vmap[format_pkg_version(ver)] = elist
        spec["side_effects"] = se_json

        # v4: already JSON-safe
        spec["api_surface"] = raw.get("api_surface", {})
        spec["project_calls"] = raw.get("project_calls", [])
        spec["symbol_requires"] = raw.get("symbol_requires", {})
        spec["scripts"] = raw.get("scripts", {})
        spec["entrypoints"] = raw.get("entrypoints", [])
        spec["consistency_rules"] = raw.get("consistency_rules", [])
        spec["enable_diagnostics"] = bool(raw.get("enable_diagnostics", False))
        spec["hidden_rules"] = raw.get("hidden_rules", [])

        return spec

    # ---------- 对外接口：生成 JSON-safe world_spec ----------

    def generate(self) -> Dict[str, Any]:
        raw = self._generate_raw()
        spec = self._to_json_friendly(raw)
        rules_nl = self._build_rules_nl(spec)
        rules_nl_compact = self._build_rules_nl_compact(spec)
        rules_nl_deps_only = self._build_rules_nl_deps_only(spec)
        # 让生成参数与 rulebook 出现在 dict 最前面（Python 3.7+ 保持插入顺序；json.dump 会保序）
        return {
            "_gen": self._generator_params(),
            "rules_nl_deps_only": rules_nl_deps_only,
            # "rules_nl_compact": rules_nl_compact,
            # "rules_nl": rules_nl,
            **spec,
        }


if __name__ == "__main__":
    import json
    from collections import defaultdict

    def _fmt_cons(cons_json: List[List[str]]) -> str:
        if not cons_json:
            return "<none>"
        return ",".join(f"{op}{vs}" for op, vs in cons_json)

    def print_world_spec_readable(
        spec: Dict[str, Any],
        *,
        max_packages: int = 12,
        max_versions_per_pkg: int = 2,
        show_non_base_deps: bool = False,
    ) -> None:
        """
        人工校验用：以结构化可读形式打印依赖与规则（不输出到文件）。
        - 默认重点展示 base_libs 相关依赖与一致性规则
        - 依赖图很大时做截断
        """
        pkgs = spec.get("packages", {}) or {}
        vdeps = spec.get("version_dependencies", {}) or {}
        vconfs = spec.get("version_conflicts", {}) or {}
        base_libs = spec.get("base_libs", []) or []
        cons_rules = spec.get("consistency_rules", []) or []
        scripts = spec.get("scripts", {}) or {}
        entrypoints = spec.get("entrypoints", []) or []

        print("========== WORLD SPEC SUMMARY ==========")
        print(f"- python_versions: {spec.get('python_versions', [])}")
        print(f"- num_packages: {len(pkgs)}")
        print(f"- base_libs: {base_libs}")
        print(f"- consistency_rules: {len(cons_rules)}")
        for i, r in enumerate(cons_rules[:5]):
            if not isinstance(r, dict):
                continue
            print(
                f"  [{i}] {r.get('type')} anchor={r.get('anchor')} "
                f"packages={len(r.get('packages', []) or [])}"
            )
        if len(cons_rules) > 5:
            print(f"  ... {len(cons_rules) - 5} more")

        # 统计依赖边数量
        edge_cnt = 0
        base_edge_cnt = 0
        for pkg, ver_map in vdeps.items():
            if not isinstance(ver_map, dict):
                continue
            for _ver, dep_list in ver_map.items():
                if not isinstance(dep_list, list):
                    continue
                edge_cnt += len(dep_list)
                for dep in dep_list:
                    if isinstance(dep, list) and dep and dep[0] in base_libs:
                        base_edge_cnt += 1
        print(f"- dependency_edges: total={edge_cnt}, to_base_libs={base_edge_cnt}")
        print(f"- conflicts_entries: {len(vconfs)} (per-pkg maps)")

        # 打印脚本结构
        print("\n--- Repo entrypoints (run.py executes in order) ---")
        print("entrypoints:")
        for p in entrypoints:
            print(f"  - {p}")

        print("\n--- Repo scripts (calls/providers/imports) ---")
        for path, info in sorted(scripts.items()):
            if not isinstance(info, dict):
                continue
            calls = info.get("calls", []) or []
            imports = info.get("imports", []) or []
            providers = []
            for c in calls:
                if isinstance(c, dict) and isinstance(c.get("provider"), str):
                    providers.append(c["provider"])
            providers = sorted(set(providers))
            print(
                f"- {path}: calls={len(calls)} providers={providers} imports={len(imports)}"
            )

        # 打印依赖（重点展示 base libs 相关）
        print("\n--- Package dependencies (truncated) ---")
        shown = 0
        for pkg_name in sorted(pkgs.keys()):
            if shown >= max_packages:
                break
            info = pkgs[pkg_name] or {}
            vers = info.get("versions", []) or []
            prio = info.get("priority", "?")
            print(f"\n[{pkg_name}] priority={prio} versions={vers[:max_versions_per_pkg]}{'...' if len(vers)>max_versions_per_pkg else ''}")

            ver_map = vdeps.get(pkg_name, {}) or {}
            if not isinstance(ver_map, dict) or not ver_map:
                print("  deps: <none>")
                shown += 1
                continue

            # 取前 max_versions_per_pkg 个版本打印
            for ver_str in vers[:max_versions_per_pkg]:
                dep_list = ver_map.get(ver_str, []) or []
                if not isinstance(dep_list, list):
                    dep_list = []
                if not dep_list:
                    print(f"  - {ver_str}: deps=<none>")
                    continue

                base_parts = []
                other_parts = []
                for dep_pkg, cons_json in dep_list:
                    if dep_pkg in base_libs:
                        base_parts.append(f"{dep_pkg}({_fmt_cons(cons_json)})")
                    else:
                        other_parts.append(f"{dep_pkg}({_fmt_cons(cons_json)})")

                if base_parts:
                    print(f"  - {ver_str}: base_deps: " + ", ".join(base_parts[:10]) + (" ..." if len(base_parts) > 10 else ""))
                if show_non_base_deps and other_parts:
                    print(f"            other_deps: " + ", ".join(other_parts[:10]) + (" ..." if len(other_parts) > 10 else ""))

            shown += 1

        if len(pkgs) > max_packages:
            print(f"\n... {len(pkgs) - max_packages} more packages not shown")
        print("========================================\n")

    test_data = []
    idx = 1

    count_list = [6,9,9,6]
    # 默认生成一组更难的 repo-lite 数据
    for i, num_packages in enumerate([9,10,11,12]):
        for _ in range(count_list[i]):
            # 不依赖 numpy：用可复现的 python RNG 采样超参
            local_rng = random.Random(100000 + idx)
            gen = RandomWorldSpecGeneratorV7_5(
                num_packages=num_packages,
                min_versions=3,
                max_versions=local_rng.choice([8, 9, 10]),
                python_versions=None,
                rng_seed=42 + idx,
                project_range_strict_prob=0.6,
                implicit_range_strict_prob=0.6,
                dep_range_strict_prob=local_rng.uniform(0.75, 0.80),
                high_version_conflict_ratio=local_rng.uniform(0.75, 0.8),
                fork_point_ratio=local_rng.uniform(0.75, 0.90),
                max_deps_per_version=local_rng.choice([8,9,10]),
            )
            spec = gen.generate()
            test_data.append(spec)
            # 只打印第一条样本，避免刷屏；需要更多就自行改这里
            if idx == 1:
                print_world_spec_readable(
                    spec,
                    max_packages=12,
                    max_versions_per_pkg=4,
                    show_non_base_deps=True,
                )
            idx += 1

    # 避免覆盖旧文件：默认输出 v5 后缀
    with open("test_data/repo/test_repo_lite_251217.json", "w") as file:
        json.dump(test_data, file, indent=4)

    # -----------------------------
    # 自动检查：依赖图是否存在环
    # -----------------------------
    def _build_pkg_edges(world: Dict[str, Any]):
        vdeps = world.get("version_dependencies", {}) or {}
        edges = set()
        self_loops = set()
        for pkg, ver_map in vdeps.items():
            if not isinstance(ver_map, dict):
                continue
            for _ver, dep_list in ver_map.items():
                if not isinstance(dep_list, list):
                    continue
                for dep in dep_list:
                    if not isinstance(dep, list) or len(dep) != 2:
                        continue
                    dep_pkg = dep[0]
                    if not isinstance(dep_pkg, str):
                        continue
                    if dep_pkg == pkg:
                        self_loops.add(pkg)
                    edges.add((pkg, dep_pkg))
        return edges, self_loops

    def _find_any_cycle(edges):
        g = defaultdict(list)
        nodes = set()
        for a, b in edges:
            g[a].append(b)
            nodes.add(a)
            nodes.add(b)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}
        parent = {}

        def dfs(u):
            color[u] = GRAY
            for v in g.get(u, []):
                if color.get(v, WHITE) == WHITE:
                    parent[v] = u
                    cyc = dfs(v)
                    if cyc:
                        return cyc
                elif color.get(v) == GRAY:
                    # back-edge u->v，回溯出一个环
                    path = [v]
                    cur = u
                    while cur != v and cur in parent:
                        path.append(cur)
                        cur = parent[cur]
                    path.append(v)
                    path.reverse()
                    return path
            color[u] = BLACK
            return None

        for n in list(nodes):
            if color[n] == WHITE:
                parent[n] = None
                cyc = dfs(n)
                if cyc:
                    return cyc
        return None

    has_cycle = False
    for i, world in enumerate(test_data):
        edges, self_loops = _build_pkg_edges(world)
        cycle_path = _find_any_cycle(edges)
        if self_loops or cycle_path:
            has_cycle = True
            print("❌ Dependency cycle detected!")
            print(f"- sample_idx: {i}")
            print(f"- base_libs: {world.get('base_libs')}")
            if self_loops:
                print(f"- self_loops: {sorted(self_loops)}")
            if cycle_path:
                print("- cycle_path: " + " -> ".join(cycle_path))
            break

    if not has_cycle:
        print("✅ Dependency graph check passed: no cycles found.")
