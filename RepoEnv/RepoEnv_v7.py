import random
import copy
from typing import Dict, List, Tuple, Optional, Any, Union
import re

# =========================
# 基础类型与工具函数
# =========================

Version = Tuple[int, int]          # (major, minor)
Constraint = Tuple[str, Version]   # (op, version)
ConstraintList = List[Constraint]


def compare_versions(a: Union[int, float, Version],
                     b: Union[int, float, Version]) -> int:
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


def version_satisfies_constraints(ver: Version,
                                  constraints: ConstraintList) -> bool:
    return all(check_version_constraint(ver, op, tgt)
               for op, tgt in constraints)


def parse_semver_to_tuple(ver_str: str) -> Version:
    """
    "3.10" -> (3,10)
    "2"    -> (2,0)
    "1.2.3" -> (1,2)   # v6: 容忍 patch 版本，忽略第三段
    """
    # 容错：从字符串中提取第一个形如 x、x.y 或 x.y.z 的数字版本，
    # 忽略尾部脏字符（例如模型输出污染：'3<action'、'1.2.1</action>'）。
    s = (ver_str or "").strip()
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", s)
    if not m:
        raise ValueError(f"Invalid semantic version: {ver_str}")
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    return (major, minor)


def format_python_version(v: Version) -> str:
    return f"{v[0]}.{v[1]}"


def format_pkg_version(v: Version) -> str:
    return f"{v[0]}.{v[1]}"


def format_constraint_list(constraints: ConstraintList) -> str:
    if not constraints:
        return "<none>"
    return ",".join(f"{op}{format_pkg_version(v)}" for op, v in constraints)


def normalize_constraints(constraints: ConstraintList,
                          all_versions: List[Version]) -> ConstraintList:
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


def intersect_constraints(a: ConstraintList,
                          b: ConstraintList,
                          all_versions: List[Version]) -> ConstraintList:
    """
    a ∧ b 的交集，并自动 normalize。
    """
    return normalize_constraints(a + b, all_versions)


# =========================
# JSON-safe World Generator v7.5
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
        high_version_conflict_ratio: float = 0.5,
        fork_point_ratio: float = 0.4,
    ):
        self.rng = random.Random(rng_seed)
        self.num_packages = num_packages
        self.min_versions = min_versions
        self.max_versions = max_versions
        self.project_range_strict_prob = project_range_strict_prob
        self.implicit_range_strict_prob = implicit_range_strict_prob
        self.dep_range_strict_prob = dep_range_strict_prob
        self.high_version_conflict_ratio = high_version_conflict_ratio
        self.fork_point_ratio = fork_point_ratio

        if python_versions is None:
            python_versions = [(3, 8), (3, 9), (3, 10), (3, 11)]
        self.python_versions = python_versions

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

    def _derive_project_python_constraint(
        self, solution: Dict[str, Any]
    ) -> Tuple[str, Version]:
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
                if earlier and self.rng.random() < 0.8:
                    k_dep = self.rng.randint(1, min(3, len(earlier)))
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
                    bad_versions = [
                        vv for vv in other_versions if vv != installed[other_pkg]
                    ]
                    if not bad_versions:
                        continue
                    conf_ver = self.rng.choice(bad_versions)
                    msg = (
                        f"{pkg}=={format_pkg_version(v)} is not compatible with "
                        f"{other_pkg}=={format_pkg_version(conf_ver)}"
                    )
                    version_conflicts[key].append(
                        (other_pkg, "==", conf_ver, msg)
                    )

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

    # ---------- 内部：side-effects ----------

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
        side_effects: Dict[
            Tuple[str, Version], List[Tuple[str, str, ConstraintList]]
        ] = {}

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

    # ---------- 内部：高版本冲突（不破坏解） ----------

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

            anchor_pkg = self.rng.choice(anchor_pkgs)
            anchor_ver = installed[anchor_pkg]
            # 让冲突更“范围化”：惩罚 anchor_pkg 的高版本（常见默认装最新陷阱），
            # 但不影响 ground-truth（anchor_ver 一定不触发）。
            anchor_vers = sorted(packages[anchor_pkg]["versions"])
            op = "=="
            boundary = anchor_ver

            if anchor_ver in anchor_vers:
                idx = anchor_vers.index(anchor_ver)
                # 若 solution 不是最高版本，则用 >= next_solution 来卡“装更高版本”
                if idx + 1 < len(anchor_vers) and self.rng.random() < 0.7:
                    op = ">="
                    boundary = anchor_vers[idx + 1]
                # 否则（solution 是最高版本或随机没选到），退化为等值冲突
                else:
                    op = "=="
                    boundary = anchor_ver

            msg = (
                f"{pkg}=={format_pkg_version(v_max)} is not compatible with "
                f"{anchor_pkg} {op} {format_pkg_version(boundary)} (high-version penalty)"
            )
            version_conflicts[key].append((anchor_pkg, op, boundary, msg))

    # ---------- 内部：fork-point 依赖 ----------

    def _inject_fork_point_dependencies(
        self,
        packages: Dict[str, Any],
        solution: Dict[str, Any],
        version_dependencies: Dict[Tuple[str, Version], List[Tuple[str, ConstraintList]]],
    ) -> None:
        pkg_list = list(packages.keys())
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

            other_pkgs = [p for p in pkg_list if p != core_pkg]
            if len(other_pkgs) < 2:
                continue

            depA_pkg, depB_pkg = self.rng.sample(other_pkgs, 2)

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
                        merged = intersect_constraints(
                            existing_constraints, new_norm, all_vers
                        )
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
        proj_pkg_reqs = self._derive_project_package_requirements(
            packages, solution
        )

        version_requires_python: Dict[Tuple[str, Version], Tuple[str, Version]] = {}
        version_dependencies: Dict[
            Tuple[str, Version], List[Tuple[str, ConstraintList]]
        ] = {}

        self._generate_dependencies_and_pyreqs(
            packages,
            topo_order,
            solution,
            version_dependencies,
            version_requires_python,
        )

        version_conflicts: Dict[
            Tuple[str, Version], List[Tuple[str, str, Version, str]]
        ] = {}
        self._generate_base_conflicts(
            packages, solution, version_conflicts
        )

        implicit_deps = self._generate_implicit_project_deps(
            packages, solution, proj_pkg_reqs
        )

        side_effects = self._generate_side_effects(
            packages, solution, version_dependencies
        )

        self._inject_high_version_conflicts(
            packages, solution, version_conflicts
        )
        self._inject_fork_point_dependencies(
            packages, solution, version_dependencies
        )

        return {
            "python_versions": self.python_versions,
            "packages": packages,
            "version_requires_python": version_requires_python,
            "version_dependencies": version_dependencies,
            "version_conflicts": version_conflicts,
            "project_requires_python": proj_py_req,
            "project_requires_packages": proj_pkg_reqs,
            "implicit_project_dependencies": implicit_deps,
            "side_effects": side_effects,
        }

    # ---------- 内部：raw → JSON-safe spec ----------

    def _to_json_friendly(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        spec: Dict[str, Any] = {}

        # python_versions: List[str]
        spec["python_versions"] = [
            format_python_version(v) for v in raw["python_versions"]
        ]

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
                cons_json = [
                    [op, format_pkg_version(vv)] for op, vv in constraints
                ]
                dep_list.append([dep_pkg, cons_json])
            vmap[format_pkg_version(ver)] = dep_list
        spec["version_dependencies"] = vdep_json

        # version_conflicts: {pkg: {ver_str: [[conf_pkg, op, conf_ver_str, msg], ...]}}
        vconf_json: Dict[str, Dict[str, Any]] = {}
        for (pkg, ver), confs in raw["version_conflicts"].items():
            vmap = vconf_json.setdefault(pkg, {})
            clist = []
            for conf_pkg, op, conf_ver, msg in confs:
                clist.append(
                    [conf_pkg, op, format_pkg_version(conf_ver), msg]
                )
            vmap[format_pkg_version(ver)] = clist
        spec["version_conflicts"] = vconf_json

        # project_requires_python: [op, py_str]
        op_py, pyv = raw["project_requires_python"]
        spec["project_requires_python"] = [op_py, format_python_version(pyv)]

        # project_requires_packages: [[pkg, [[op, ver_str], ...]], ...]
        prj_pkgs = []
        for pkg, constraints in raw["project_requires_packages"]:
            cons_json = [
                [op, format_pkg_version(vv)] for op, vv in constraints
            ]
            prj_pkgs.append([pkg, cons_json])
        spec["project_requires_packages"] = prj_pkgs

        # implicit_project_dependencies: [[pkg, [[op, ver_str], ...]], ...]
        impl = []
        for pkg, constraints in raw["implicit_project_dependencies"]:
            cons_json = [
                [op, format_pkg_version(vv)] for op, vv in constraints
            ]
            impl.append([pkg, cons_json])
        spec["implicit_project_dependencies"] = impl

        # side_effects: {pkg: {ver_str: [[eff_type, dep_pkg, [[op, ver_str], ...]], ...]}}
        se_json: Dict[str, Dict[str, Any]] = {}
        for (pkg, ver), effects in raw["side_effects"].items():
            vmap = se_json.setdefault(pkg, {})
            elist = []
            for eff_type, dep_pkg, constraints in effects:
                cons_json = [
                    [op, format_pkg_version(vv)] for op, vv in constraints
                ]
                elist.append([eff_type, dep_pkg, cons_json])
            vmap[format_pkg_version(ver)] = elist
        spec["side_effects"] = se_json

        return spec

    # ---------- 对外接口：生成 JSON-safe world_spec ----------

    def generate(self) -> Dict[str, Any]:
        raw = self._generate_raw()
        return self._to_json_friendly(raw)


class ComputerEnvSetupInductionEnvV7_5:
    """
    v7.5.2 环境（基于 v7.5 原始结构）：
    - world_spec 为 JSON-safe（字符串 key）
    - 支持 pip install/uninstall/python run.py/pip list
    - v7.5.2 修复：pip install 对 package 安装引入“原子事务回滚”，避免失败安装污染环境状态
    """

    def __init__(
        self,
        world_spec: Dict[str, Any],
        max_steps: int = 120,
        seed: Optional[int] = None,
    ):
        self.world_spec_json = world_spec
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self._load_world_spec(world_spec)

        self.step_count: int = 0
        self.python_version: Optional[Version] = None
        self.installed: Dict[str, Version] = {}
        self.last_message = ""
        self.last_action: Optional[str] = None
        self.done: bool = False
        self.success: bool = False

    # ---------- world_spec 解析 ----------

    def _load_world_spec(self, spec: Dict[str, Any]) -> None:
        # 默认关闭诊断命令（更贴近真实：元数据不一定可得，且容易形成捷径）
        self.enable_diagnostics: bool = bool(spec.get("enable_diagnostics", False))

        self.python_versions: List[Version] = [
            parse_semver_to_tuple(s) for s in spec["python_versions"]
        ]

        self.packages: Dict[str, Dict[str, Any]] = {}
        for pkg, info in spec["packages"].items():
            vers = [parse_semver_to_tuple(s) for s in info["versions"]]
            self.packages[pkg] = {
                "versions": vers,
                "priority": info["priority"],
            }

        self.version_requires_python: Dict[
            Tuple[str, Version], Tuple[str, Version]
        ] = {}
        for pkg, ver_map in spec.get("version_requires_python", {}).items():
            for ver_str, pair in ver_map.items():
                op, py_str = pair
                self.version_requires_python[
                    (pkg, parse_semver_to_tuple(ver_str))
                ] = (op, parse_semver_to_tuple(py_str))

        self.version_dependencies: Dict[
            Tuple[str, Version], List[Tuple[str, ConstraintList]]
        ] = {}
        for pkg, ver_map in spec.get("version_dependencies", {}).items():
            for ver_str, dep_list in ver_map.items():
                key = (pkg, parse_semver_to_tuple(ver_str))
                deps: List[Tuple[str, ConstraintList]] = []
                for dep_pkg, cons_json in dep_list:
                    cons = [
                        (op, parse_semver_to_tuple(vs)) for op, vs in cons_json
                    ]
                    deps.append((dep_pkg, cons))
                self.version_dependencies[key] = deps

        self.version_conflicts: Dict[
            Tuple[str, Version], List[Tuple[str, str, Version, str]]
        ] = {}
        for pkg, ver_map in spec.get("version_conflicts", {}).items():
            for ver_str, clist in ver_map.items():
                key = (pkg, parse_semver_to_tuple(ver_str))
                confs: List[Tuple[str, str, Version, str]] = []
                for conf_pkg, op, conf_ver_str, msg in clist:
                    confs.append(
                        (conf_pkg, op, parse_semver_to_tuple(conf_ver_str), msg)
                    )
                self.version_conflicts[key] = confs

        op_py, py_str = spec["project_requires_python"]
        self.project_requires_python: Tuple[str, Version] = (
            op_py,
            parse_semver_to_tuple(py_str),
        )

        self.project_requires_packages: List[Tuple[str, ConstraintList]] = []
        for pkg, cons_json in spec.get("project_requires_packages", []):
            cons = [
                (op, parse_semver_to_tuple(vs)) for op, vs in cons_json
            ]
            self.project_requires_packages.append((pkg, cons))

        self.implicit_project_dependencies: List[
            Tuple[str, ConstraintList]
        ] = []
        for pkg, cons_json in spec.get("implicit_project_dependencies", []):
            cons = [
                (op, parse_semver_to_tuple(vs)) for op, vs in cons_json
            ]
            self.implicit_project_dependencies.append((pkg, cons))

        self.side_effects: Dict[
            Tuple[str, Version], List[Tuple[str, str, ConstraintList]]
        ] = {}
        for pkg, ver_map in spec.get("side_effects", {}).items():
            for ver_str, eff_list in ver_map.items():
                key = (pkg, parse_semver_to_tuple(ver_str))
                effs: List[Tuple[str, str, ConstraintList]] = []
                for eff_type, dep_pkg, cons_json in eff_list:
                    cons = [
                        (op, parse_semver_to_tuple(vs))
                        for op, vs in cons_json
                    ]
                    effs.append((eff_type, dep_pkg, cons))
                self.side_effects[key] = effs

        # -----------------------------
        # API surface & symbol-level rules (optional, v6)
        # -----------------------------
        # api_surface[pkg][ver] = {"modules": [...], "exports": {m:[sym...]}, "signatures": {"m:sym":[kw...]}}
        self.api_modules: Dict[Tuple[str, Version], set] = {}
        self.api_exports: Dict[Tuple[str, Version, str], set] = {}
        self.api_signatures: Dict[Tuple[str, Version, str, str], set] = {}

        for pkg, ver_map in (spec.get("api_surface", {}) or {}).items():
            if not isinstance(ver_map, dict):
                continue
            for ver_str, entry in ver_map.items():
                try:
                    v = parse_semver_to_tuple(ver_str)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue

                modules = entry.get("modules", []) or []
                if isinstance(modules, list):
                    self.api_modules[(pkg, v)] = set(
                        [m for m in modules if isinstance(m, str)]
                    )

                exports = entry.get("exports", {}) or {}
                if isinstance(exports, dict):
                    for m, syms in exports.items():
                        if isinstance(m, str) and isinstance(syms, list):
                            self.api_exports[(pkg, v, m)] = set(
                                [s for s in syms if isinstance(s, str)]
                            )

                sigs = entry.get("signatures", {}) or {}
                if isinstance(sigs, dict):
                    for k, kws in sigs.items():
                        if not isinstance(k, str) or ":" not in k:
                            continue
                        if not isinstance(kws, list):
                            continue
                        m, s = k.split(":", 1)
                        self.api_signatures[(pkg, v, m, s)] = set(
                            [x for x in kws if isinstance(x, str)]
                        )

        # project_calls: run.py 的合成调用序列（不泄露版本）
        self.project_calls: List[Dict[str, Any]] = []
        for c in spec.get("project_calls", []) or []:
            if isinstance(c, dict):
                self.project_calls.append(c)

        # symbol_requires[pkg][ver] = list(req)
        # req: {"provider":..., "module":..., "symbol":..., "required_kwargs":[...]}
        self.symbol_requires: Dict[Tuple[str, Version], List[Dict[str, Any]]] = {}
        for pkg, ver_map in (spec.get("symbol_requires", {}) or {}).items():
            if not isinstance(ver_map, dict):
                continue
            for ver_str, reqs in ver_map.items():
                try:
                    v = parse_semver_to_tuple(ver_str)
                except Exception:
                    continue
                if isinstance(reqs, list):
                    self.symbol_requires[(pkg, v)] = [
                        r for r in reqs if isinstance(r, dict)
                    ]

        # -----------------------------
        # v7: Repo hierarchy (scripts + entrypoints)
        # -----------------------------
        # scripts[path] = {"calls": [...], "imports": [pkg,...]}
        self.scripts: Dict[str, Dict[str, Any]] = {}
        scripts = spec.get("scripts", {}) or {}
        if isinstance(scripts, dict):
            for path, entry in scripts.items():
                if not isinstance(path, str) or not isinstance(entry, dict):
                    continue
                calls = entry.get("calls", []) or []
                imports = entry.get("imports", []) or []
                if not isinstance(calls, list):
                    calls = []
                if not isinstance(imports, list):
                    imports = []
                self.scripts[path] = {
                    "calls": [c for c in calls if isinstance(c, dict)],
                    "imports": [p for p in imports if isinstance(p, str)],
                }

        entrypoints = spec.get("entrypoints", None)
        if isinstance(entrypoints, list):
            self.entrypoints: List[str] = [x for x in entrypoints if isinstance(x, str)]
        else:
            # 兼容旧 spec：没有 scripts/entrypoints 时，run.py 使用 project_calls
            self.entrypoints = []

        # 若没有 scripts，但存在 project_calls，则把它当成单脚本（run.py 的主体）
        if not self.scripts and self.project_calls:
            self.scripts["app/run_calls"] = {
                "calls": self.project_calls[:],
                "imports": [],
            }
            self.entrypoints = ["app/run_calls"]

        # -----------------------------
        # v7: base libs + consistency rules (optional)
        # -----------------------------
        self.base_libs: List[str] = [
            x for x in (spec.get("base_libs", []) or []) if isinstance(x, str)
        ]
        # consistency_rules:
        # - {"type": "same_major_with_anchor"|"same_minor_with_anchor",
        #    "anchor": "pkgX", "packages": ["pkgA",...], "error": "..."}
        self.consistency_rules: List[Dict[str, Any]] = []
        for r in spec.get("consistency_rules", []) or []:
            if not isinstance(r, dict):
                continue
            rtype = r.get("type")
            anchor = r.get("anchor")
            pkgs = r.get("packages", []) or []
            if not isinstance(rtype, str) or not isinstance(anchor, str) or not isinstance(pkgs, list):
                continue
            pkgs2 = [p for p in pkgs if isinstance(p, str)]
            if not pkgs2:
                continue
            self.consistency_rules.append(
                {
                    "type": rtype,
                    "anchor": anchor,
                    "packages": pkgs2,
                    "error": r.get(
                        "error",
                        "RuntimeError: consistency check failed.",
                    ),
                }
            )

        # -----------------------------
        # Hidden rules (optional)
        # -----------------------------
        # 这些规则用于模拟“只看症状，不透露版本/约束”的真实调试体验：
        # - v5 兼容格式：{"when_installed": {pkg: "1.2", ...}, "error": "..."}
        # - v6 推荐格式：{"when": [[pkg, [[op, "1.2"], ...]], ...], "error": "..."}
        # 规则触发后只返回症状型错误，不包含任何版本信息。
        self.hidden_rules: List[Dict[str, Any]] = []
        for r in spec.get("hidden_rules", []) or []:
            when: List[Tuple[str, ConstraintList]] = []

            # v6 格式：when = [[pkg, [[op, ver_str], ...]], ...]
            when_json = r.get("when", None)
            if isinstance(when_json, list):
                for item in when_json:
                    if (
                        isinstance(item, list)
                        and len(item) == 2
                        and isinstance(item[0], str)
                        and isinstance(item[1], list)
                    ):
                        pkg = item[0]
                        cons_json = item[1]
                        cons: ConstraintList = []
                        for pair in cons_json:
                            if (
                                isinstance(pair, list)
                                and len(pair) == 2
                                and isinstance(pair[0], str)
                                and isinstance(pair[1], str)
                            ):
                                cons.append(
                                    (pair[0], parse_semver_to_tuple(pair[1]))
                                )
                        if cons:
                            when.append((pkg, cons))

            # v5 兼容格式：when_installed = {pkg: "1.2", ...}
            if not when:
                when_installed_raw = r.get("when_installed", {}) or {}
                if isinstance(when_installed_raw, dict):
                    for pkg, ver_str in when_installed_raw.items():
                        if isinstance(pkg, str) and isinstance(ver_str, str):
                            when.append((pkg, [("==", parse_semver_to_tuple(ver_str))]))

            if not when:
                continue

            scope = r.get("scope", None)
            scopes: List[str] = []
            if isinstance(scope, list):
                scopes = [x for x in scope if isinstance(x, str)]
            elif isinstance(scope, str):
                scopes = [scope]
            # 默认只在 run.py（全项目）阶段触发，避免子脚本调试时被黑盒误伤
            if not scopes:
                scopes = ["run.py"]

            self.hidden_rules.append(
                {
                    "when": when,
                    "scope": scopes,
                    "error": r.get(
                        "error",
                        "RuntimeError: project failed due to an unknown issue.",
                    ),
                }
            )

    # ---------- Gym-like 接口 ----------

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.python_version = None
        self.installed = {}
        self.last_message = ""
        self.last_action = None
        self.done = False
        self.success = False
        return self._get_obs()

    def step(self, action: str):
        if self.done:
            self.last_message = (
                "ERROR: Environment already finished. Please reset()."
            )
            return self._get_obs(), 0.0, True, {}

        self.step_count += 1
        self.last_action = action
        self.last_message = ""
        reward = 0.0

        cmd = (action or "").strip()
        # 容错：有时 action 会带上 <action> 标签或其它残留，避免污染解析
        cmd = cmd.replace("<action>", "").replace("</action>", "").strip()
        # 只取第一行，避免把后续内容拼到版本串里
        if "\n" in cmd:
            cmd = cmd.split("\n", 1)[0].strip()
        if not cmd or cmd == "noop":
            pass
        elif cmd == "pip list":
            self._handle_pip_list()
        elif cmd in ("repo tree", "repo ls"):
            self._handle_repo_tree()
        elif cmd.startswith("pip install"):
            self._handle_pip_install(cmd)
        elif cmd.startswith("pip uninstall"):
            self._handle_pip_uninstall(cmd)
        elif cmd.startswith("python"):
            self._handle_python_command(cmd)
        else:
            self.last_message = f"ERROR: Unknown command: {cmd}"

        if cmd.startswith("python") and self.success:
            reward = 1.0

        # if self.step_count >= self.max_steps and not self.done:
        #     self.done = True

        return self._get_obs(), reward, self.done, {}

    # ---------- pip list ----------

    def _handle_pip_list(self) -> None:
        lines = ["Environment status:"]
        if self.python_version is None:
            lines.append("  Python: <not set>")
        else:
            lines.append(
                f"  Python: {format_python_version(self.python_version)}"
            )

        if not self.installed:
            lines.append("  Installed packages: <none>")
        else:
            lines.append("  Installed packages:")
            for pkg in sorted(self.installed.keys()):
                v = self.installed[pkg]
                lines.append(f"    - {pkg}=={format_pkg_version(v)}")

        self.last_message = "\n".join(lines)

    # ---------- pip show ----------
    def _handle_pip_show(self, cmd: str) -> None:
        """
        Lightweight diagnostic command.
        Example: pip show pkgA
        """
        tokens = cmd.split()
        if len(tokens) != 3 or tokens[0] != "pip" or tokens[1] != "show":
            self.last_message = f"ERROR: Invalid pip show command: {cmd}"
            return

        pkg = tokens[2].strip()
        if not pkg:
            self.last_message = f"ERROR: Invalid pip show command: {cmd}"
            return

        if pkg.lower() == "python":
            if self.python_version is None:
                self.last_message = "ERROR: Python is not configured."
            else:
                self.last_message = (
                    f"Name: python\n"
                    f"Version: {format_python_version(self.python_version)}"
                )
            return

        v = self.installed.get(pkg)
        if v is None:
            self.last_message = f"ERROR: Package '{pkg}' is not installed."
            return

        lines = [
            f"Name: {pkg}",
            f"Version: {format_pkg_version(v)}",
        ]

        # Requires-Python (package-level)
        if (pkg, v) in self.version_requires_python:
            op, target_py = self.version_requires_python[(pkg, v)]
            lines.append(
                f"Requires-Python: {op} {format_python_version(target_py)}"
            )

        # Requires (explicit deps) — show constraints here (by design),
        # but run.py errors will not reveal them automatically.
        deps = self.version_dependencies.get((pkg, v), [])
        if deps:
            req_items = []
            for dep_pkg, cons in deps:
                if cons:
                    req_items.append(
                        f"{dep_pkg}{format_constraint_list(cons)}"
                    )
                else:
                    req_items.append(dep_pkg)
            lines.append("Requires: " + ", ".join(req_items))
        else:
            lines.append("Requires: <none>")

        # Conflicts (redacted message, keep only structured part)
        confs = self.version_conflicts.get((pkg, v), [])
        if confs:
            conf_items = [
                f"{conf_pkg} {op} {format_pkg_version(conf_ver)}"
                for conf_pkg, op, conf_ver, _msg in confs
            ]
            lines.append("Conflicts: " + ", ".join(conf_items))
        else:
            lines.append("Conflicts: <none>")

        # Side-effects (for high-priority packages)
        effs = self.side_effects.get((pkg, v), [])
        if effs:
            items = []
            for eff_type, dep_pkg, cons in effs:
                if cons:
                    items.append(f"{eff_type} {dep_pkg}{format_constraint_list(cons)}")
                else:
                    items.append(f"{eff_type} {dep_pkg}")
            lines.append("Side-Effects: " + ", ".join(items))
        else:
            lines.append("Side-Effects: <none>")

        self.last_message = "\n".join(lines)

    # ---------- pip check ----------
    def _handle_pip_check(self) -> None:
        """
        Similar to 'pip check': report broken requirements/conflicts
        based on current environment state, without telling the exact
        constraint ranges for dependency mismatches.
        """
        issues: List[str] = []

        # Python configured?
        if self.python_version is None:
            issues.append("Python is not configured.")

        # Package-level python requirement
        for pkg, v in sorted(self.installed.items()):
            if (pkg, v) in self.version_requires_python:
                op, target_py = self.version_requires_python[(pkg, v)]
                if self.python_version is None or not check_version_constraint(
                    self.python_version, op, target_py
                ):
                    issues.append(
                        f"{pkg}=={format_pkg_version(v)} requires Python "
                        f"{op} {format_python_version(target_py)}."
                    )

        # Dependency satisfaction for installed packages
        for pkg, v in sorted(self.installed.items()):
            deps = self.version_dependencies.get((pkg, v), [])
            for dep_pkg, constraints in deps:
                inst = self.installed.get(dep_pkg)
                if inst is None:
                    issues.append(
                        f"{pkg}=={format_pkg_version(v)} requires {dep_pkg} "
                        f"but it is missing."
                    )
                else:
                    if constraints and not version_satisfies_constraints(
                        inst, constraints
                    ):
                        issues.append(
                            f"{pkg}=={format_pkg_version(v)} has incompatible "
                            f"dependency {dep_pkg}=={format_pkg_version(inst)}."
                        )

        # Conflicts (runtime-like)
        for (pkg, v), confs in self.version_conflicts.items():
            if self.installed.get(pkg) != v:
                continue
            for conf_pkg, op, conf_ver, _msg in confs:
                inst = self.installed.get(conf_pkg)
                if inst is not None and check_version_constraint(
                    inst, op, conf_ver
                ):
                    issues.append(
                        f"{pkg}=={format_pkg_version(v)} is incompatible with "
                        f"{conf_pkg}=={format_pkg_version(inst)}."
                    )

        if not issues:
            self.last_message = "No broken requirements found."
        else:
            lines = ["Found broken requirements:"]
            for x in issues:
                lines.append(f"  - {x}")
            self.last_message = "\n".join(lines)

    # ---------- 命令解析 ----------

    def _handle_pip_install(self, cmd: str) -> None:
        tokens = cmd.split()
        if len(tokens) < 3 or tokens[0] != "pip" or tokens[1] != "install":
            self.last_message = f"ERROR: Invalid pip install command: {cmd}"
            return

        specs = tokens[2:]

        # ===========================
        # v7.5.2 明确不支持 multi-spec
        # ===========================
        if len(specs) != 1:
            self.last_message = (
                "ERROR: Unsupported command.\n"
                "This environment only supports installing one package per command.\n"
                "Please install packages sequentially."
            )
            return

        spec = specs[0].strip()
        if not spec:
            self.last_message = (
                f"ERROR: No package specified in command: {cmd}"
            )
            return

        try:
            name, constraints = self._parse_name_and_constraints(spec)
        except ValueError as e:
            # 解析错误不应让环境崩溃（常见：==latest、==1.2.1</action> 等）
            self.last_message = f"ERROR: Invalid requirement spec '{spec}'. {e}"
            return

        if name.lower() == "python":
            self._handle_install_python(name, constraints)
        else:
            self._handle_install_package(name, constraints)

    def _handle_pip_uninstall(self, cmd: str) -> None:
        tokens = cmd.split()
        if len(tokens) < 3 or tokens[0] != "pip" or tokens[1] != "uninstall":
            self.last_message = f"ERROR: Invalid pip uninstall command: {cmd}"
            return
        pkg = tokens[2]
        self._handle_uninstall_package(pkg)

    def _handle_python_command(self, cmd: str) -> None:
        tokens = cmd.split()
        if len(tokens) < 2:
            self.last_message = f"ERROR: Unsupported python command: {cmd}"
            return

        # 修复：未安装/未配置 Python 时，任何 python 命令都应先报错
        if self.python_version is None:
            self.last_message = (
                "ERROR: Python is not configured. "
                "Please install an explicit version first, e.g., "
                "'pip install python==3.10'."
            )
            return

        target = tokens[1].strip()
        if target == "run.py":
            ok, msg = self._run_entrypoints()
            if ok:
                self.success = True
                self.done = True
                self.last_message = "Project executed successfully."
            else:
                self.last_message = self._format_run_errors(
                    [{"type": "script_error", "msg": msg}]
                )
            return

        # v7: 支持运行子脚本进行逐个击破
        if target in self.scripts:
            ok, msg = self._run_script(target)
            if ok:
                self.last_message = f"Script executed successfully: {target}"
            else:
                self.last_message = self._format_run_errors(
                    [{"type": "script_error", "msg": msg}]
                )
            return

        self.last_message = f"ERROR: Unknown script: {target}. Try 'repo tree'."

    # ---------- repo tree ----------
    def _handle_repo_tree(self) -> None:
        """
        输出 repo 的层级脚本结构（不泄露依赖/版本），用于让 agent 选择调试入口。
        """
        paths = sorted(self.scripts.keys())
        if not paths:
            self.last_message = "Repo tree is empty."
            return

        lines = ["Repo tree:"]
        lines.append("  run.py")
        for p in paths:
            lines.append(f"  {p}")
        self.last_message = "\n".join(lines)

    # ---------- v7: run entrypoints / single script ----------
    def _run_entrypoints(self) -> Tuple[bool, str]:
        # 若没有 entrypoints，回退到旧的 _check_run（兼容旧 spec）
        if not self.entrypoints:
            ok, errors = self._check_run()
            if ok:
                return True, ""
            return False, errors[0]["msg"] if errors else "RuntimeError: run failed."

        for script_path in self.entrypoints:
            ok, msg = self._run_script(script_path)
            if not ok:
                # 给出脚本上下文（更像真实 monorepo 调试）
                return False, f"while running '{script_path}': {msg}"
        return True, ""

    def _run_script(self, script_path: str) -> Tuple[bool, str]:
        entry = self.scripts.get(script_path, {})
        calls = entry.get("calls", []) or []
        imports = entry.get("imports", []) or []
        if not isinstance(calls, list):
            calls = []
        if not isinstance(imports, list):
            imports = []
        imported_pkgs = set([p for p in imports if isinstance(p, str)])
        # 默认把 call 的 provider 也视为该脚本会触达的包
        for c in calls:
            if isinstance(c, dict):
                p = c.get("provider")
                if isinstance(p, str):
                    imported_pkgs.add(p)

        # 0) 一致性规则：在该脚本触达相关包时才检查（避免无意义全局阻塞）
        ok, msg = self._run_consistency_rules(imported_pkgs, current_scope=script_path)
        if not ok:
            return False, msg

        # 1) 入口 calls
        ok, msg = self._run_calls(
            calls, context=f"in '{script_path}'"
        )
        if not ok:
            return False, msg

        # 2) 包间符号依赖（只对该脚本会 import 的 caller 生效）
        ok, msg = self._run_symbol_requires(
            imported_pkgs, context=f"in '{script_path}'"
        )
        if not ok:
            return False, msg

        # 3) hidden rules（按 scope 生效）
        ok, msg = self._run_hidden_rules(
            imported_pkgs, current_scope=script_path
        )
        if not ok:
            return False, msg

        return True, ""

    def _run_consistency_rules(self, imported_pkgs: set, current_scope: str) -> Tuple[bool, str]:
        """
        强组合规则：跨包一致性（类似 ABI/锁步组件）。
        仅当 anchor 与 group 中至少一个包被该脚本触达时才检查，
        且只在相关包都已安装时才触发（缺包优先走 ModuleNotFoundError）。
        """
        if not self.consistency_rules:
            return True, ""

        for r in self.consistency_rules:
            rtype = r.get("type")
            anchor = r.get("anchor")
            pkgs = r.get("packages", [])
            if not isinstance(rtype, str) or not isinstance(anchor, str) or not isinstance(pkgs, list):
                continue

            touched = False
            if anchor in imported_pkgs:
                touched = True
            else:
                for p in pkgs:
                    if p in imported_pkgs:
                        touched = True
                        break
            if not touched:
                continue

            a_ver = self.installed.get(anchor)
            if a_ver is None:
                # 让缺包错误自然发生在后续 checks
                continue

            # 只对已安装的成员做一致性检查（未安装的交给后续 missing）
            for p in pkgs:
                v = self.installed.get(p)
                if v is None:
                    continue

                if rtype == "same_major_with_anchor":
                    if v[0] != a_ver[0]:
                        return False, r.get(
                            "error",
                            f"RuntimeError: ABI mismatch detected between '{anchor}' and '{p}'.",
                        )
                elif rtype == "same_minor_with_anchor":
                    if v[0] != a_ver[0] or v[1] != a_ver[1]:
                        return False, r.get(
                            "error",
                            f"RuntimeError: components out of sync: '{anchor}' vs '{p}'.",
                        )
                else:
                    # 未知类型：忽略，保持兼容
                    continue

        return True, ""

    def _run_calls(self, calls: List[Dict[str, Any]], context: str) -> Tuple[bool, str]:
        for call in calls:
            provider = call.get("provider")
            module = call.get("module")
            symbol = call.get("symbol")
            required_kwargs = call.get("required_kwargs", []) or []
            if (
                not isinstance(provider, str)
                or not isinstance(module, str)
                or not isinstance(symbol, str)
                or not isinstance(required_kwargs, list)
            ):
                continue

            pv = self.installed.get(provider)
            if pv is None:
                return (
                    False,
                    f"{context}: ModuleNotFoundError: No module named '{provider}'.",
                )

            if module not in self.api_modules.get((provider, pv), set()):
                return (
                    False,
                    f"{context}: ModuleNotFoundError: No module named '{module}'.",
                )

            exported = self.api_exports.get((provider, pv, module), set())
            if symbol not in exported:
                return (
                    False,
                    f"{context}: ImportError: cannot import name '{symbol}' from '{module}'.",
                )

            sig = self.api_signatures.get((provider, pv, module, symbol), set())
            for kw in required_kwargs:
                if isinstance(kw, str) and kw not in sig:
                    return (
                        False,
                        f"{context}: TypeError: during project entry, "
                        f"{module}.{symbol}() got an unexpected keyword argument '{kw}'.",
                    )

        return True, ""

    def _run_symbol_requires(self, imported_pkgs: set, context: str) -> Tuple[bool, str]:
        for (pkg, v), reqs in self.symbol_requires.items():
            # 只检查该脚本会 import 的 caller
            if pkg not in imported_pkgs:
                continue
            if self.installed.get(pkg) != v:
                continue
            for r in reqs:
                provider = r.get("provider")
                module = r.get("module")
                symbol = r.get("symbol")
                required_kwargs = r.get("required_kwargs", []) or []
                if (
                    not isinstance(provider, str)
                    or not isinstance(module, str)
                    or not isinstance(symbol, str)
                    or not isinstance(required_kwargs, list)
                ):
                    continue

                pv = self.installed.get(provider)
                if pv is None:
                    return (
                        False,
                        f"{context}: ModuleNotFoundError: No module named '{provider}'.",
                    )

                if module not in self.api_modules.get((provider, pv), set()):
                    return (
                        False,
                        f"{context}: ModuleNotFoundError: No module named '{module}'.",
                    )

                exported = self.api_exports.get((provider, pv, module), set())
                if symbol not in exported:
                    return (
                        False,
                        f"{context}: ImportError: cannot import name '{symbol}' from '{module}'.",
                    )

                sig = self.api_signatures.get((provider, pv, module, symbol), set())
                for kw in required_kwargs:
                    if isinstance(kw, str) and kw not in sig:
                        return (
                            False,
                            f"{context}: TypeError: while importing '{pkg}', "
                            f"{module}.{symbol}() got an unexpected keyword argument '{kw}'.",
                        )

        return True, ""

    def _run_hidden_rules(self, imported_pkgs: set, current_scope: str) -> Tuple[bool, str]:
        for r in self.hidden_rules:
            scopes = r.get("scope", [])
            if isinstance(scopes, list):
                scopes_list = [x for x in scopes if isinstance(x, str)]
            elif isinstance(scopes, str):
                scopes_list = [scopes]
            else:
                scopes_list = ["run.py"]

            # run.py 视为全项目 scope
            allowed = (
                current_scope in scopes_list
                or "run.py" in scopes_list
                or "*" in scopes_list
            )
            if not allowed:
                continue

            when: List[Tuple[str, ConstraintList]] = r.get("when", [])
            ok = True
            for rpkg, cons in when:
                inst = self.installed.get(rpkg)
                if inst is None:
                    ok = False
                    break
                if cons and not version_satisfies_constraints(inst, cons):
                    ok = False
                    break
            if ok:
                return False, r.get(
                    "error",
                    "RuntimeError: project failed due to an unknown issue.",
                )
        return True, ""

    # ---------- 解析 name + constraints ----------

    def _parse_name_and_constraints(
        self, spec: str
    ) -> Tuple[str, ConstraintList]:
        """
        支持形式：
        - "pkg0"
        - "pkg0==1.2"
        - "pkg0>=1.2,<2.0"
        - "pkg0<1.5"
        """
        spec = spec.strip()
        ops = ["==", ">=", "<=", ">", "<"]

        min_pos = None
        for op in ops:
            pos = spec.find(op)
            if pos != -1 and (min_pos is None or pos < min_pos):
                min_pos = pos

        if min_pos is None:
            return spec, []

        name = spec[:min_pos].strip()
        tail = spec[min_pos:].strip()

        constraints: ConstraintList = []
        while tail:
            matched = False
            for op in ["==", ">=", "<=", ">", "<"]:
                if tail.startswith(op):
                    matched = True
                    tail_remain = tail[len(op):].strip()
                    if "," in tail_remain:
                        v_str, tail_remain = tail_remain.split(",", 1)
                        tail = tail_remain.strip()
                    else:
                        v_str = tail_remain
                        tail = ""
                    # 容错：允许 v_str 带脏字符（例如 action 残留、标点）
                    ver = parse_semver_to_tuple(v_str.strip())
                    constraints.append((op, ver))
                    break
            if not matched:
                break

        return name, constraints

    # ---------- 安装 Python ----------

    def _handle_install_python(
        self, name: str, constraints: ConstraintList
    ) -> None:
        if not constraints:
            self.last_message = (
                "ERROR: Installing Python requires an explicit version, e.g., "
                "'pip install python==3.10'."
            )
            return

        if len(constraints) != 1 or constraints[0][0] != "==":
            self.last_message = (
                "ERROR: Only exact Python version is supported, "
                "e.g., 'python==3.10'."
            )
            return

        py_tuple = constraints[0][1]
        available = self.python_versions
        if available and py_tuple not in available:
            avail_str = ", ".join(format_python_version(v) for v in available)
            self.last_message = (
                f"ERROR: Could not find a Python version that satisfies "
                f"python=={format_python_version(py_tuple)} "
                f"(available versions: {avail_str})"
            )
            return

        self.python_version = py_tuple
        self.last_message = (
            f"Successfully installed python=={format_python_version(py_tuple)}"
        )

    # ==========================================================
    # v7.5.2：关键修复 —— package 安装引入事务回滚（只加这一层）
    # ==========================================================

    def _handle_install_package(
        self, pkg: str, constraints: ConstraintList
    ) -> None:
        # v7.5.2: snapshot for atomic install
        snapshot_installed = copy.deepcopy(self.installed)
        snapshot_last_message = self.last_message

        # 清理一下 last_message，避免误判上一条 ERROR
        self.last_message = None

        # —— 下面逻辑保持 v7.5 原样 ——
        if pkg not in self.packages:
            self.last_message = (
                f"ERROR: Could not find a package named '{pkg}'."
            )
            # rollback
            self.installed = snapshot_installed
            self.last_message = self.last_message  # keep error
            return

        available = self.packages[pkg]["versions"]

        # 无版本约束：安装最新，如果已安装则提示
        if not constraints:
            if pkg in self.installed:
                current_ver = self.installed[pkg]
                self.last_message = (
                    f"{pkg}=={format_pkg_version(current_ver)} "
                    f"is already installed."
                )
                # success-like, no rollback needed
                return
            chosen_ver = max(available)
            self._install_concrete_version(pkg, chosen_ver)
            # v7.5.2: rollback on error
            if self.last_message and self.last_message.startswith("ERROR:"):
                self.installed = snapshot_installed
            return

        # 先对用户给的约束 normalize
        normalized = normalize_constraints(constraints, available)
        if not normalized:
            avail_str = ", ".join(format_pkg_version(x) for x in available)
            self.last_message = (
                f"ERROR: Could not find any version of {pkg} that satisfies "
                f"constraints {format_constraint_list(constraints)} "
                f"(available versions: {avail_str})"
            )
            # rollback
            self.installed = snapshot_installed
            return

        candidates = [
            v for v in available if version_satisfies_constraints(v, normalized)
        ]
        if not candidates:
            avail_str = ", ".join(format_pkg_version(x) for x in available)
            self.last_message = (
                f"ERROR: Could not find any version of {pkg} that satisfies "
                f"constraints {format_constraint_list(normalized)} "
                f"(available versions: {avail_str})"
            )
            # rollback
            self.installed = snapshot_installed
            return

        chosen_ver = max(candidates)

        if pkg in self.installed and self.installed[pkg] == chosen_ver:
            self.last_message = (
                f"{pkg}=={format_pkg_version(chosen_ver)} is already installed "
                f"and satisfies constraints "
                f"{format_constraint_list(normalized)}."
            )
            return

        self._install_concrete_version(pkg, chosen_ver)

        # v7.5.2: rollback on error (关键!)
        if self.last_message and self.last_message.startswith("ERROR:"):
            self.installed = snapshot_installed

        # 如果不是 error，就 commit（什么都不做）
        # snapshot_last_message 不需要恢复；last_message 已由 _install_concrete_version 设定

    # ---------- 实际写入安装（含 side-effects） ----------

    def _install_concrete_version(self, pkg: str, v: Version) -> None:
        available = self.packages[pkg]["versions"]
        if v not in available:
            avail_str = ", ".join(format_pkg_version(x) for x in available)
            self.last_message = (
                f"ERROR: Could not find a version that satisfies the "
                f"requirement {pkg}=={format_pkg_version(v)} "
                f"(available versions: {avail_str})"
            )
            return

        priority = self.packages[pkg]["priority"]

        installed_changes: List[Tuple[str, Version]] = []
        side_effect_changes: List[Tuple[str, Version]] = []

        # 包级 Python 约束
        if (pkg, v) in self.version_requires_python:
            op, target_py = self.version_requires_python[(pkg, v)]
            if self.python_version is None:
                self.last_message = (
                    f"ERROR: {pkg}=={format_pkg_version(v)} requires Python "
                    f"{op} {format_python_version(target_py)}, "
                    f"but no Python version is configured."
                )
                return
            if not check_version_constraint(self.python_version, op, target_py):
                self.last_message = (
                    f"ERROR: {pkg}=={format_pkg_version(v)} requires Python "
                    f"{op} {format_python_version(target_py)}, "
                    f"but you have Python "
                    f"{format_python_version(self.python_version)}."
                )
                return

        # high priority: 自动 ensure + side-effects
        if priority == "high":
            ok, msg, new_installs = self._ensure_dependencies_with_side_effects_verbose(
                pkg, v
            )
            if not ok:
                self.last_message = msg
                return
            side_effect_changes.extend(new_installs)
        # medium priority: 严格检查依赖，必要时自动 upgrade
        elif priority == "medium":
            ok, msg = self._check_dependencies_for_install_strict(pkg, v)
            if not ok:
                self.last_message = msg
                return
        # low priority: 保持原行为（不强制修复依赖）

        # 冲突检查
        conflicts = self.version_conflicts.get((pkg, v), [])
        for conf_pkg, op, conf_ver, msg in conflicts:
            inst_ver = self.installed.get(conf_pkg)
            if inst_ver is not None and check_version_constraint(inst_ver, op, conf_ver):
                self.last_message = (
                    f"ERROR: Cannot install {pkg}=={format_pkg_version(v)} "
                    f"because it conflicts with "
                    f"{conf_pkg}=={format_pkg_version(inst_ver)}. {msg}"
                )
                return

        # 真正写入安装
        self.installed[pkg] = v
        installed_changes.append((pkg, v))

        # 去重：同一个包可能在同一次安装过程中被多次升级/降级，
        # 环境最终只保留最后一次写入的版本；message 也应只显示最终版本。
        se_final: Dict[str, Version] = {}
        for spkg, sv in side_effect_changes:
            se_final[spkg] = sv
        for spkg, sv in se_final.items():
            self.installed[spkg] = sv

        # success message
        lines: List[str] = []
        if installed_changes:
            lines.append(
                "Successfully installed "
                + " ".join(
                    f"{p}=={format_pkg_version(ver)}"
                    for p, ver in installed_changes
                )
            )
        if se_final:
            lines.append(
                "Also installed/updated due to dependencies: "
                + ", ".join(
                    f"{p}=={format_pkg_version(ver)}"
                    for p, ver in sorted(se_final.items(), key=lambda x: x[0])
                )
            )
        self.last_message = "\n".join(lines)

    # ---------- medium priority 依赖 ----------

    def _check_dependencies_for_install_strict(
        self, pkg: str, v: Version
    ) -> Tuple[bool, str]:
        deps = self.version_dependencies.get((pkg, v), [])
        if not deps:
            return True, ""

        problems = []
        for dep_pkg, constraints in deps:
            inst = self.installed.get(dep_pkg)

            if inst is None:
                problems.append(
                    f"{dep_pkg} (not installed)"
                    # f"requires {format_constraint_list(constraints)})"
                )
                continue

            if not version_satisfies_constraints(inst, constraints):
                all_vers = self.packages[dep_pkg]["versions"]
                candidates = [
                    vv
                    for vv in all_vers
                    if version_satisfies_constraints(vv, constraints)
                ]
                if not candidates:
                    problems.append(
                        f"{dep_pkg} (installed {format_pkg_version(inst)}, "
                        f"requires {format_constraint_list(constraints)}, "
                        f"but no compatible version exists)"
                    )
                    continue

                chosen = max(candidates)
                self.installed[dep_pkg] = chosen  # 自动升级

        if not problems:
            return True, ""

        lines = [
            f"ERROR: Cannot install {pkg}=={format_pkg_version(v)} because "
            f"the following required dependencies are missing:",
        ]
        for p in problems:
            lines.append(f"  - {p}")

        return False, "\n".join(lines)

    # ---------- high priority 依赖 + side-effects ----------

    def _ensure_dependencies_with_side_effects_verbose(
        self, pkg: str, v: Version
    ) -> Tuple[bool, str, List[Tuple[str, Version]]]:
        """
        high priority 包：
        - 显式依赖 & side-effect 都会自动安装或升级，
          并返回发生的变更列表用于 message。
        """
        # 记录“最终变更”（同一包可能被多次改版本，只保留最后一次）
        changed: Dict[str, Version] = {}

        # 显式依赖
        deps = self.version_dependencies.get((pkg, v), [])
        for dep_pkg, constraints in deps:
            if dep_pkg not in self.packages:
                return (
                    False,
                    f"ERROR: Internal world error: dependency {dep_pkg} not defined.",
                    [],
                )
            all_vers = self.packages[dep_pkg]["versions"]
            candidates = [
                x for x in all_vers if version_satisfies_constraints(x, constraints)
            ]
            if not candidates:
                return (
                    False,
                    f"ERROR: Cannot satisfy dependency {dep_pkg} with "
                    f"constraints {format_constraint_list(constraints)} "
                    f"when installing {pkg}=={format_pkg_version(v)} "
                    f"(no compatible version available).",
                    [],
                )
            chosen = max(candidates)
            current = self.installed.get(dep_pkg)
            if current is None or not version_satisfies_constraints(current, constraints):
                self.installed[dep_pkg] = chosen
                changed[dep_pkg] = chosen

        # side-effects
        effects = self.side_effects.get((pkg, v), [])
        for eff_type, dep_pkg, constraints in effects:
            if dep_pkg not in self.packages:
                continue
            all_vers = self.packages[dep_pkg]["versions"]
            candidates = [
                x for x in all_vers if version_satisfies_constraints(x, constraints)
            ]
            if not candidates:
                return (
                    False,
                    f"ERROR: Side-effect for {pkg}=={format_pkg_version(v)} "
                    f"cannot be satisfied: {dep_pkg} with constraints "
                    f"{format_constraint_list(constraints)} has no valid version.",
                    [],
                )
            current = self.installed.get(dep_pkg)

            # 语义：更难的非单调 side-effect
            # - ensure：仅在不满足时补齐到 max(candidates)
            # - force_high：无论如何强制到 max(candidates)
            # - force_low：无论如何强制到 min(candidates)
            # - pin：通常 constraints 为 ==x，强制到唯一候选（等价 max/min）
            if eff_type == "ensure":
                chosen = max(candidates)
                if current is None or not version_satisfies_constraints(
                    current, constraints
                ):
                    self.installed[dep_pkg] = chosen
                    changed[dep_pkg] = chosen
            elif eff_type == "force_high":
                chosen = max(candidates)
                if current != chosen:
                    self.installed[dep_pkg] = chosen
                    changed[dep_pkg] = chosen
            elif eff_type == "force_low":
                chosen = min(candidates)
                if current != chosen:
                    self.installed[dep_pkg] = chosen
                    changed[dep_pkg] = chosen
            elif eff_type == "pin":
                chosen = max(candidates)
                if current != chosen:
                    self.installed[dep_pkg] = chosen
                    changed[dep_pkg] = chosen
            else:
                # 未知类型：忽略（保持兼容）
                continue

        # 转回 list，保持输出稳定（按包名排序）
        return True, "", sorted(changed.items(), key=lambda x: x[0])

    # ---------- uninstall ----------

    def _handle_uninstall_package(self, pkg: str) -> None:
        if pkg not in self.installed:
            self.last_message = f"ERROR: Package '{pkg}' is not installed."
            return
        old_ver = self.installed[pkg]
        del self.installed[pkg]
        self.last_message = (
            f"Successfully uninstalled {pkg}=={format_pkg_version(old_ver)}"
        )

    # ---------- python run.py 检查 ----------

    def _check_run(self) -> Tuple[bool, List[Dict[str, Any]]]:
        errors: List[Dict[str, Any]] = []

        # 项目级 Python
        op_py, target_py = self.project_requires_python
        if self.python_version is None:
            errors.append(
                {
                    "type": "python_mismatch",
                    "msg": (
                        f"Python version is not set."
                        # f"Python version is not set. The project requires "
                        # f"Python {op_py} {format_python_version(target_py)}."
                    ),
                }
            )
            return False, errors[:1]
        else:
            if not check_version_constraint(self.python_version, op_py, target_py):
                errors.append(
                    {
                        "type": "python_mismatch",
                        "msg": (
                            f"Python {format_python_version(self.python_version)} "
                            f"does not satisfy the project requirement: "
                            f"Python {op_py} {format_python_version(target_py)}."
                        ),
                    }
                )
                return False, errors[:1]

        # ==========================================================
        # v6: object-level checks (preferred when project_calls exists)
        # ==========================================================
        if self.project_calls:
            # 先检查 project_calls（项目入口）
            for call in self.project_calls:
                provider = call.get("provider")
                module = call.get("module")
                symbol = call.get("symbol")
                required_kwargs = call.get("required_kwargs", []) or []
                if (
                    not isinstance(provider, str)
                    or not isinstance(module, str)
                    or not isinstance(symbol, str)
                    or not isinstance(required_kwargs, list)
                ):
                    continue

                pv = self.installed.get(provider)
                if pv is None:
                    errors.append(
                        {
                            "type": "module_missing",
                            "msg": f"ModuleNotFoundError: No module named '{provider}'.",
                        }
                    )
                    return False, errors[:1]

                if module not in self.api_modules.get((provider, pv), set()):
                    errors.append(
                        {
                            "type": "module_missing",
                            "msg": f"ModuleNotFoundError: No module named '{module}'.",
                        }
                    )
                    return False, errors[:1]

                exported = self.api_exports.get((provider, pv, module), set())
                if symbol not in exported:
                    errors.append(
                        {
                            "type": "symbol_missing",
                            "msg": f"ImportError: cannot import name '{symbol}' from '{module}'.",
                        }
                    )
                    return False, errors[:1]

                sig = self.api_signatures.get((provider, pv, module, symbol), set())
                for kw in required_kwargs:
                    if isinstance(kw, str) and kw not in sig:
                        errors.append(
                            {
                                "type": "signature_mismatch",
                                "msg": (
                                    f"TypeError: during project entry, "
                                    f"{module}.{symbol}() got an unexpected keyword argument '{kw}'."
                                ),
                            }
                        )
                        return False, errors[:1]

            # 再检查运行时依赖（symbol_requires），用于模拟“包 A import 了 B 的某个符号”
            for (pkg, v), reqs in self.symbol_requires.items():
                if self.installed.get(pkg) != v:
                    continue
                for r in reqs:
                    provider = r.get("provider")
                    module = r.get("module")
                    symbol = r.get("symbol")
                    required_kwargs = r.get("required_kwargs", []) or []
                    if (
                        not isinstance(provider, str)
                        or not isinstance(module, str)
                        or not isinstance(symbol, str)
                        or not isinstance(required_kwargs, list)
                    ):
                        continue

                    pv = self.installed.get(provider)
                    if pv is None:
                        errors.append(
                            {
                                "type": "runtime_provider_missing",
                                "msg": f"ModuleNotFoundError: No module named '{provider}'.",
                            }
                        )
                        return False, errors[:1]

                    if module not in self.api_modules.get((provider, pv), set()):
                        errors.append(
                            {
                                "type": "runtime_module_missing",
                                "msg": f"ModuleNotFoundError: No module named '{module}'.",
                            }
                        )
                        return False, errors[:1]

                    exported = self.api_exports.get((provider, pv, module), set())
                    if symbol not in exported:
                        errors.append(
                            {
                                "type": "runtime_symbol_missing",
                                "msg": f"ImportError: cannot import name '{symbol}' from '{module}'.",
                            }
                        )
                        return False, errors[:1]

                    sig = self.api_signatures.get((provider, pv, module, symbol), set())
                    for kw in required_kwargs:
                        if isinstance(kw, str) and kw not in sig:
                            errors.append(
                                {
                                    "type": "runtime_signature_mismatch",
                                    "msg": (
                                        f"TypeError: while importing '{pkg}', "
                                        f"{module}.{symbol}() got an unexpected keyword argument '{kw}'."
                                    ),
                                }
                            )
                            return False, errors[:1]

            # 最后检查 hidden_rules（黑盒故障），其报错应当是对象级的
            for r in self.hidden_rules:
                when: List[Tuple[str, ConstraintList]] = r.get("when", [])
                ok = True
                for rpkg, cons in when:
                    inst = self.installed.get(rpkg)
                    if inst is None:
                        ok = False
                        break
                    if cons and not version_satisfies_constraints(inst, cons):
                        ok = False
                        break
                if ok:
                    errors.append(
                        {
                            "type": "hidden_rule_triggered",
                            "msg": r.get(
                                "error",
                                "RuntimeError: project failed due to an unknown issue.",
                            ),
                        }
                    )
                    return False, errors[:1]

            return True, []

        # 项目必需包（范围）
        for pkg, constraints in self.project_requires_packages:
            inst = self.installed.get(pkg)
            if inst is None:
                errors.append(
                    {
                        "type": "missing_package",
                        "msg": (
                            f"Project requires {pkg} "
                            # f"{format_constraint_list(constraints)}, "
                            f"but it is not installed."
                        ),
                    }
                )
                return False, errors[:1]
            elif not version_satisfies_constraints(inst, constraints):
                errors.append(
                    {
                        "type": "version_mismatch",
                        "msg": (
                            # 不暴露版本/约束：只给“症状”
                            f"ImportError: cannot import name 'Config' from '{pkg}.core'."
                        ),
                    }
                )
                return False, errors[:1]

        # 包级 Python 约束
        for (pkg, v), (op, target_py2) in self.version_requires_python.items():
            if self.installed.get(pkg) == v:
                if self.python_version is None or not check_version_constraint(
                    self.python_version, op, target_py2
                ):
                    errors.append(
                        {
                            "type": "package_python_mismatch",
                            "msg": (
                                f"{pkg}=={format_pkg_version(v)} requires "
                                f"Python {op} "
                                f"{format_python_version(target_py2)}, "
                                f"but you have Python "
                                f"{format_python_version(self.python_version) if self.python_version else 'None'}."
                            ),
                        }
                    )
                    return False, errors[:1]

        # 显式依赖
        for (pkg, v), deps in self.version_dependencies.items():
            if self.installed.get(pkg) != v:
                continue
            for dep_pkg, constraints in deps:
                inst = self.installed.get(dep_pkg)
                if inst is None:
                    errors.append(
                        {
                            "type": "runtime_missing_dep",
                            "msg": (
                                f"Runtime error: {pkg}=={format_pkg_version(v)} "
                                f"requires dependency {dep_pkg}, "
                                f"which is not installed."
                            ),
                        }
                    )
                    return False, errors[:1]
                elif not version_satisfies_constraints(inst, constraints):
                    errors.append(
                        {
                            "type": "runtime_dep_version_mismatch",
                            "msg": (
                                # 不暴露具体是哪个版本不兼容
                                f"ImportError: cannot import name 'Backend' from '{dep_pkg}.core'."
                            ),
                        }
                    )
                    return False, errors[:1]

        # 隐式项目依赖
        for pkg, constraints in self.implicit_project_dependencies:
            inst = self.installed.get(pkg)
            if inst is None:
                errors.append(
                    {
                        "type": "implicit_missing_dep",
                        "msg": (
                            f"Runtime import error: project implicitly imports "
                            f"{pkg}, but it is not installed."
                        ),
                    }
                )
                return False, errors[:1]
            elif not version_satisfies_constraints(inst, constraints):
                errors.append(
                    {
                        "type": "implicit_dep_version_mismatch",
                        "msg": (
                            f"ModuleNotFoundError: No module named '{pkg}.core'."
                        ),
                    }
                )
                return False, errors[:1]

        # 冲突检查
        for (pkg, v), confs in self.version_conflicts.items():
            if self.installed.get(pkg) != v:
                continue
            for conf_pkg, op, conf_ver, msg in confs:
                inst = self.installed.get(conf_pkg)
                if inst is not None and check_version_constraint(inst, op, conf_ver):
                    errors.append(
                        {
                            "type": "conflict",
                            "msg": (
                                # 不暴露冲突双方与版本细节
                                f"RuntimeError: ABI mismatch detected while importing '{conf_pkg}'."
                            ),
                        }
                    )
                    return False, errors[:1]

        # hidden rules（组合触发的“黑盒故障”）
        for r in self.hidden_rules:
            when: List[Tuple[str, ConstraintList]] = r.get("when", [])
            ok = True
            for rpkg, cons in when:
                inst = self.installed.get(rpkg)
                # hidden rule 只在“包已安装且满足条件”时触发；
                # 如果包未安装，不触发（避免覆盖真实 missing 报错）。
                if inst is None:
                    ok = False
                    break
                if cons and not version_satisfies_constraints(inst, cons):
                    ok = False
                    break
            if ok:
                errors.append(
                    {
                        "type": "hidden_rule_triggered",
                        "msg": r.get(
                            "error",
                            "RuntimeError: project failed due to an unknown issue.",
                        ),
                    }
                )
                return False, errors[:1]

        ok = len(errors) == 0
        return ok, errors

    def _format_run_errors(self, errors: List[Dict[str, Any]]) -> str:
        if not errors:
            return ""
        # Make run.py non-oracle: only show one (redacted) issue.
        e = errors[0]
        return "\n".join(
            [
                "ERROR: Project execution failed.",
                f"  - {e['msg']}",
            ]
        )

    # ---------- 观测 ----------

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "python_version": self.python_version,
            "installed": copy.deepcopy(self.installed),
            "last_message": self.last_message,
            "last_action": self.last_action,
            "success": self.success,
        }


    def return_obs(self):
        return self.last_message



if __name__ == "__main__":
    import json

    # gen = RandomWorldSpecGeneratorV7_5(num_packages=5, rng_seed=0)
    # spec = gen.generate()

    # ✅ 直接写 JSON
    # with open("world_case_v7_5.json", "w") as f:
    #     json.dump(spec, f, indent=2)

    # print("Saved JSON-safe world to world_case_v7_5.json")

    # 创建环境并跑几步

    with open(f"test_data/repo/test_repo_lite_251214.json", "r") as file:
        test_data = json.load(file)

    for i in range(len(test_data)):
        spec = test_data[i]
        env = ComputerEnvSetupInductionEnvV7_5(spec, max_steps=120, seed=0)
        # obs = env.reset()
        # print("Initial obs:", obs)
        print(f"====== Test Sample {i+1} ======")
        done = False
        step = 1
        while not done:
            print(f"=== Step {step} ===")
            a = input(">>> COMMAND:")
            obs, reward, done, info = env.step(a)
            # print("Reward:", reward, "Done:", done)
            print("Last message:\n", obs["last_message"])
            # print("Current installed:", {k: format_pkg_version(v) for k, v in obs["installed"].items()})
            print("-----")
            if done:
                break
            step += 1
