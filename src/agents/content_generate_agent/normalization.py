from __future__ import annotations

from typing import Any, Callable

from loguru import logger



def _summarize_for_log(v: Any, *, max_len: int = 200) -> str:
    try:
        s = repr(v)
    except Exception:
        s = f"<unreprable:{type(v).__name__}>"
    if len(s) > max_len:
        s = s[:max_len] + "...(truncated)"
    return s


def is_noneish(value: Any) -> bool:
    """
    判定“等价于 None”的值：
    - None
    - 空白字符串（含 ""、"   "）
    - 空数组 []
    - 空对象 {}

    使用场景：
    - 业务逻辑里想把 Java 侧传来的“空值”（""/[]/{})当成无值时，统一用这个判定，
      避免各处自行写 `x is None` / `x == ""` / `len(x)==0` 造成契约不一致。

    （注：暂未被使用，暂时保留/考虑哪里可以使用）
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    return False


def normalize_noneish(
    value: Any,
    *,
    path: str = "$",
    warn: Callable[[str], None] | None = None,
) -> Any:
    """
    递归归一化“None-ish”值：
    - 空白字符串 -> None（并 strip 非空字符串）
    - [] -> None
    - {} -> None
    - list/dict 递归处理，移除归一化为 None 的元素/键值对

    注意：
    - 这是“激进归一化”，会改变输入结构；所有归一化都会 warning（可传 warn 关闭/接管）。

    什么时候会被使用：
    - 在 Pydantic `@model_validator(mode="before")` 中作为“入口归一化”，
      在字段解析与类型转换之前，把 Java 侧的空字符串/空数组/空对象统一清洗为 None，
      从而让后续业务逻辑只需要处理 `None` 这一种“无值”表达。
    """
    warn_fn = warn or logger.warning

    if value is None:
        return None

    if isinstance(value, str):
        if value.strip() == "":
            warn_fn(f"[noneish.normalize] {path}: empty/blank string -> None (was={_summarize_for_log(value)})")
            return None
        return value

    if isinstance(value, list):
        out: list[Any] = []
        changed = False
        for idx, item in enumerate(value):
            ni = normalize_noneish(item, path=f"{path}[{idx}]", warn=warn_fn)
            if ni is None:
                changed = True
                continue
            if ni is not item:
                changed = True
            out.append(ni)
        if len(out) == 0:
            warn_fn(f"[noneish.normalize] {path}: empty list -> None (was_len={len(value)})")
            return None
        if changed:
            warn_fn(f"[noneish.normalize] {path}: list normalized (len {len(value)} -> {len(out)})")
        return out

    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        changed = False
        for k, v in value.items():
            nv = normalize_noneish(v, path=f"{path}.{k}", warn=warn_fn)
            if nv is None:
                changed = True
                continue
            if nv is not v:
                changed = True
            out[k] = nv
        if len(out) == 0:
            warn_fn(f"[noneish.normalize] {path}: empty dict -> None (was_keys={len(value)})")
            return None
        if changed:
            warn_fn(f"[noneish.normalize] {path}: dict normalized (keys {len(value)} -> {len(out)})")
        return out

    return value

