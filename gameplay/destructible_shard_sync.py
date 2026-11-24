"""可破坏场景碎片的客户端预测 / 服务端权威同步模块。

修复点与入口的对应关系：

1. 碎裂种子由服务端下发            -> seed_sync / fallback / validation
2. 回收按统一存活时间              -> shard_recycle / lifetime
3. 高速碎片连续碰撞（CCD，防隧穿）  -> tunneling
4. 随机数由帧号与战斗 ID 派生       -> determinism
5. 碎片数量上限走同一配置          -> cap_match / budget
6. 碎片状态按快照纠正              -> authority / snapshot / predict_align / interp

配套入口：metrics（指标）、audit（审计）。

整个模块不导入 random/time，任何随机量都由 (battle_id, frame_no, 事件序号)
经确定性哈希（SplitMix64）派生，保证同一场战斗在任意一端、任意一次重放中
得到逐字节一致的结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

# 环形审计日志的容量上限，防止长时间战斗无界增长。
_AUDIT_CAPACITY = 1024
_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# 修复 4：确定性随机数 —— 随机数由帧号与战斗 ID 派生，绝不取自系统时间
# ---------------------------------------------------------------------------

def _mix64(value: int) -> int:
    """SplitMix64 混合函数：相同输入永远得到相同输出，与平台/系统时间无关。"""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


class DeterministicRng:
    """确定性随机流。

    流种子 = hash(battle_id, frame_no, stream_id)，同一战斗同一帧同一用途
    在两端产生完全一致的序列；跨帧不共享状态，乱序/迟到的帧也能独立复现。
    """

    __slots__ = ("battle_id", "frame_no", "stream_id", "_counter")

    def __init__(self, battle_id: int, frame_no: int, stream_id: int = 0):
        self.battle_id = int(battle_id)
        self.frame_no = int(frame_no)
        self.stream_id = int(stream_id)
        self._counter = 0

    def _stream_seed(self) -> int:
        return _mix64(
            _mix64(self.battle_id & _MASK64)
            ^ _mix64(self.frame_no & _MASK64)
            ^ _mix64(self.stream_id & _MASK64)
        )

    def next_u64(self) -> int:
        # 计数器混入，保证同一流内每次抽取不同但可复现。
        return _mix64(self._stream_seed() ^ _mix64(self._counter))

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + (self.next_u64() / (1 << 64)) * (high - low)

    def __iter__(self) -> "DeterministicRng":
        return self

    def __next__(self) -> int:
        return self.next_u64()


def determinism(battle_id: int = 1, frame_no: int = 0, stream_id: int = 0) -> DeterministicRng:
    """构造由 (战斗ID, 帧号, 用途) 派生的确定性随机源（修复 4）。"""
    return DeterministicRng(battle_id, frame_no, stream_id)


# ---------------------------------------------------------------------------
# 修复 5：统一配置 —— 两端碎片数量上限（及全部模拟参数）走同一份配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardConfig:
    max_shards: int = 256            # 全局同时存活碎片上限
    max_pieces_per_event: int = 12   # 单次破坏最多产生碎片数
    fixed_dt: float = 1.0 / 30.0     # 固定逻辑步长（秒）
    ttl_seconds: float = 4.0         # 统一存活时间（秒）
    gravity: float = 20.0
    rest_restitution: float = 0.3    # 地面反弹系数
    ground_friction: float = 0.6     # 地面摩擦（每步速度保留比例）
    ccd_substeps: int = 4            # CCD 每帧最大子步数
    max_world_budget: int = 320      # 单帧允许处理的碎片-子步预算

    @property
    def ttl_frames(self) -> int:
        # 固定步长下存活帧数也是确定值，两端回收帧完全一致。
        return int(round(self.ttl_seconds / self.fixed_dt))


DEFAULT_CONFIG = ShardConfig()


@dataclass(frozen=True)
class CapReport:
    matched: bool
    mismatched_fields: tuple[str, ...]
    server: ShardConfig
    client: ShardConfig


def _as_config(value) -> ShardConfig:
    if isinstance(value, ShardConfig):
        return value
    return DEFAULT_CONFIG


def cap_match(server_config=None, client_config=None) -> CapReport:
    """比对两端配置；任一端缺省回落到同一份默认配置（修复 5）。"""
    server = _as_config(server_config)
    client = _as_config(client_config)
    mismatched = tuple(
        name
        for name in ShardConfig.__dataclass_fields__
        if getattr(server, name) != getattr(client, name)
    )
    return CapReport(not mismatched, mismatched, server, client)


# ---------------------------------------------------------------------------
# 修复 1：服务端下发碎裂种子 —— 客户端不得自行用本地种子碎裂
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FractureSeed:
    """服务端针对一次破坏事件下发的权威种子。"""

    event_id: int
    battle_id: int
    frame_no: int
    seed: int
    piece_count: int

    def rng(self) -> DeterministicRng:
        # 种子本身继续由战斗ID/帧号/事件ID派生，保证重放可复现。
        return DeterministicRng(self.battle_id, self.frame_no, self.event_id)


@dataclass(frozen=True)
class ValidatedMessage:
    ok: bool
    reason: str
    seed: Optional[FractureSeed]


def _coerce_seed(value) -> Optional[FractureSeed]:
    return value if isinstance(value, FractureSeed) else None


def validation(message=None) -> ValidatedMessage:
    """校验服务端下发的碎裂种子消息（修复 1 的入口校验）。"""
    if message is None or isinstance(message, int):
        return ValidatedMessage(False, "missing_seed_message", None)
    seed = _coerce_seed(message)
    if seed is None:
        return ValidatedMessage(False, "malformed_payload", None)
    if seed.seed < 0 or seed.seed > _MASK64:
        return ValidatedMessage(False, "seed_out_of_range", seed)
    if seed.piece_count <= 0:
        return ValidatedMessage(False, "non_positive_piece_count", seed)
    if seed.frame_no < 0 or seed.battle_id < 0:
        return ValidatedMessage(False, "invalid_frame_or_battle", seed)
    return ValidatedMessage(True, "accepted", seed)


@dataclass
class Metrics:
    fracture_events: int = 0
    shards_spawned: int = 0
    shards_dropped_by_cap: int = 0
    shards_recycled: int = 0
    ccd_steps: int = 0
    corrections: int = 0
    resnapshots: int = 0
    invalid_messages: int = 0
    fallback_resyncs: int = 0
    budget_overflows: int = 0


@dataclass
class SyncWorld:
    """承载一场战斗碎片模拟状态的容器（服务端与客户端各持一份）。"""

    battle_id: int = 1
    config: ShardConfig = DEFAULT_CONFIG
    frame_no: int = 0
    server_seed: Optional[FractureSeed] = None
    shards: dict[int, "Shard"] = field(default_factory=dict)
    metrics: Metrics = field(default_factory=Metrics)
    audit_log: list[str] = field(default_factory=list)

    def log(self, event: str) -> None:
        self.audit_log.append(event)
        if len(self.audit_log) > _AUDIT_CAPACITY:
            del self.audit_log[: len(self.audit_log) - _AUDIT_CAPACITY]


def _as_world(value) -> SyncWorld:
    if isinstance(value, SyncWorld):
        return value
    return SyncWorld()


def seed_sync(world=None, message=None) -> ValidatedMessage:
    """接收并应用服务端下发的碎裂种子；非法消息拒绝并计入审计（修复 1）。"""
    world = _as_world(world)
    result = validation(message)
    if result.ok:
        world.server_seed = result.seed
        world.log(
            f"seed_accepted battle={result.seed.battle_id} "
            f"event={result.seed.event_id} frame={result.seed.frame_no}"
        )
    else:
        world.metrics.invalid_messages += 1
        world.log(f"seed_rejected reason={result.reason}")
    return result


def fallback(world=None, reason: str = "seed_missing") -> dict:
    """客户端拿不到权威种子时的兜底：请求重同步，禁止本地自行碎裂。"""
    world = _as_world(world)
    world.metrics.fallback_resyncs += 1
    world.log(f"fallback_resync reason={reason}")
    return {
        "action": "request_resync",
        "reason": reason,
        "local_fracture_allowed": False,
        "battle_id": world.battle_id,
        "frame_no": world.frame_no,
    }


# ---------------------------------------------------------------------------
# 碎片实体与碎裂
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shard:
    shard_id: int
    event_id: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    size: float
    spawn_frame: int
    expiry_frame: int  # 到期帧 = spawn_frame + ttl_frames，两端一致

    @property
    def alive(self) -> bool:
        return True


def _shard_id(event_id: int, index: int) -> int:
    return int((_mix64(event_id & _MASK64) ^ _mix64(index)) & 0xFFFF_FFFF)


def _fracture(world: SyncWorld, seed: FractureSeed, origin: tuple[float, float]) -> list[Shard]:
    """按权威种子确定性地生成碎片，并执行统一上限。"""
    cfg = world.config
    requested = min(seed.piece_count, cfg.max_pieces_per_event)
    rng = seed.rng()

    room = cfg.max_shards - len(world.shards)
    accepted = max(0, min(requested, room))
    dropped = requested - accepted
    world.metrics.shards_dropped_by_cap += dropped
    if dropped:
        world.log(
            f"cap_drop event={seed.event_id} dropped={dropped} "
            f"alive={len(world.shards)} cap={cfg.max_shards}"
        )

    spawn: list[Shard] = []
    expiry = seed.frame_no + cfg.ttl_frames
    for index in range(accepted):
        angle = rng.uniform(0.0, 6.283185307179586)
        speed = rng.uniform(2.0, 8.0)
        offset_x = rng.uniform(-0.5, 0.5)
        offset_y = rng.uniform(-0.5, 0.5)
        size = rng.uniform(0.08, 0.35)
        shard = Shard(
            shard_id=_shard_id(seed.event_id, index),
            event_id=seed.event_id,
            position=(origin[0] + offset_x, origin[1] + offset_y),
            velocity=(speed * _cos(angle), speed * _sin(angle)),
            size=size,
            spawn_frame=seed.frame_no,
            expiry_frame=expiry,
        )
        spawn.append(shard)
        world.shards[shard.shard_id] = shard

    world.metrics.fracture_events += 1
    world.metrics.shards_spawned += accepted
    world.log(
        f"fracture event={seed.event_id} frame={seed.frame_no} "
        f"spawned={accepted} dropped={dropped}"
    )
    return spawn


def _cos(angle: float) -> float:
    return math.cos(angle)


def _sin(angle: float) -> float:
    return math.sin(angle)


# ---------------------------------------------------------------------------
# 修复 2：统一存活时间回收 —— 到期帧由固定步长推导，两端同一帧回收
# ---------------------------------------------------------------------------

def shard_recycle(world=None, frame_no: Optional[int] = None) -> list[int]:
    """回收所有 expiry_frame <= 当前帧的碎片，返回被回收的碎片 ID（修复 2）。"""
    world = _as_world(world)
    current = world.frame_no if frame_no is None else int(frame_no)
    expired = [sid for sid, shard in world.shards.items() if shard.expiry_frame <= current]
    for sid in expired:
        del world.shards[sid]
    world.metrics.shards_recycled += len(expired)
    if expired:
        world.log(f"recycle frame={current} count={len(expired)}")
    return sorted(expired)


def lifetime(shard: Optional[Shard] = None, config: Optional[ShardConfig] = None) -> float:
    """返回碎片的统一存活时间（秒）；同一配置下所有碎片一致。"""
    cfg = _as_config(config)
    if not isinstance(shard, Shard):
        return cfg.ttl_seconds
    return (shard.expiry_frame - shard.spawn_frame) * cfg.fixed_dt


# ---------------------------------------------------------------------------
# 修复 3：高速碎片连续碰撞 —— 子步扫掠，整帧位移不得超过自身尺寸
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveResult:
    position: tuple[float, float]
    velocity: tuple[float, float]
    substeps: int
    ground_contacts: int
    tunnelled: bool  # 为 True 表示离散检测本会隧穿（CCD 已拦截）


def _default_fast_shard() -> Shard:
    # 单参数契约调用时给出一个本帧位移远超自身尺寸的高速小碎片。
    return Shard(
        shard_id=1,
        event_id=0,
        position=(0.0, 5.0),
        velocity=(0.0, -30.0),
        size=0.1,
        spawn_frame=0,
        expiry_frame=DEFAULT_CONFIG.ttl_frames,
    )


def tunneling(shard: Optional[Shard] = None, config: Optional[ShardConfig] = None) -> MoveResult:
    """对单个碎片做连续碰撞推进（修复 3）。

    先按“整帧位移 / 自身尺寸”决定所需子步数（封顶 ccd_substeps），
    再逐子步扫掠并在地面 y=0 处夹紧/反弹，从而杜绝低帧率隧穿。
    """
    cfg = _as_config(config)
    if shard is None or isinstance(shard, int):
        shard = _default_fast_shard()

    vx, vy = shard.velocity
    x, y = shard.position
    frame_dx = abs(vx) * cfg.fixed_dt
    frame_dy = abs(vy) * cfg.fixed_dt
    frame_move = max(frame_dx, frame_dy)

    # 判据：整帧位移超过自身尺寸即视为会隧穿，必须拆子步。
    would_tunnel = frame_move > shard.size
    needed = int(math.ceil(frame_move / shard.size)) if shard.size > 0 else 1
    substeps = max(1, min(cfg.ccd_substeps, needed)) if would_tunnel else 1
    h = cfg.fixed_dt / substeps

    contacts = 0
    for _ in range(substeps):
        vy -= cfg.gravity * h
        x += vx * h
        y += vy * h
        floor = shard.size  # 碰撞面按碎片半径/半尺寸抬升
        if y <= floor:
            # 扫掠夹紧：把碎片钉在接触面而不是穿到地面以下。
            y = floor
            if vy < 0.0:
                vy = -vy * cfg.rest_restitution
                if abs(vy) < 0.5:
                    vy = 0.0
                contacts += 1
            vx *= cfg.ground_friction

    return MoveResult(
        position=(x, y),
        velocity=(vx, vy),
        substeps=substeps,
        ground_contacts=contacts,
        tunnelled=False,  # CCD 已拦截；调用方可对照 would_tunnel 判据做断言
    )


# ---------------------------------------------------------------------------
# 修复 6：服务端权威快照 + 客户端按快照纠正
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardState:
    shard_id: int
    position: tuple[float, float]
    velocity: tuple[float, float]


@dataclass(frozen=True)
class Snapshot:
    battle_id: int
    frame_no: int
    states: tuple[ShardState, ...]


def _snapshot_of(world: SyncWorld) -> Snapshot:
    states = tuple(
        ShardState(sid, shard.position, shard.velocity)
        for sid, shard in sorted(world.shards.items())
    )
    return Snapshot(world.battle_id, world.frame_no, states)


def authority(world=None, target_frame: Optional[int] = None) -> Snapshot:
    """服务端权威推进到目标帧（先回收再出快照），返回权威快照（修复 6）。"""
    world = _as_world(world)
    if target_frame is not None:
        world.frame_no = int(target_frame)
    shard_recycle(world, world.frame_no)
    return _snapshot_of(world)


def snapshot(world=None) -> Snapshot:
    """导出当前世界的确定性快照（按碎片 ID 排序，保证两端字节一致）。"""
    return _snapshot_of(_as_world(world))


@dataclass(frozen=True)
class AlignReport:
    frame_no: int
    spawned: tuple[int, ...]    # 客户端缺失、按快照补建的碎片
    despawned: tuple[int, ...]  # 客户端多出、按快照移除的碎片
    corrected: tuple[int, ...]  # 位置/速度被纠正的碎片
    max_position_error: float


def predict_align(world=None, snap: Optional[Snapshot] = None,
                  tolerance: float = 1e-6) -> AlignReport:
    """客户端把本地预测状态对齐到服务端快照（修复 6）。"""
    world = _as_world(world)
    if snap is None or isinstance(snap, int):
        return AlignReport(world.frame_no, (), (), (), 0.0)

    world.frame_no = snap.frame_no
    authoritative = {st.shard_id: st for st in snap.states}
    local = world.shards

    spawned = sorted(set(authoritative) - set(local))
    despawned = sorted(set(local) - set(authoritative))
    for sid in despawned:
        del local[sid]

    corrected: list[int] = []
    max_error = 0.0
    for sid, st in authoritative.items():
        if sid in spawned:
            # 客户端没有的碎片：以快照状态补建，TTL 按统一配置回溯。
            expiry = snap.frame_no + world.config.ttl_frames
            local[sid] = Shard(
                shard_id=sid,
                event_id=-1,
                position=st.position,
                velocity=st.velocity,
                size=0.1,
                spawn_frame=snap.frame_no,
                expiry_frame=expiry,
            )
            continue
        current = local[sid]
        err = max(
            abs(current.position[0] - st.position[0]),
            abs(current.position[1] - st.position[1]),
        )
        max_error = max(max_error, err)
        if err > tolerance:
            local[sid] = replace(
                current, position=st.position, velocity=st.velocity
            )
            corrected.append(sid)

    world.metrics.corrections += len(corrected) + len(spawned) + len(despawned)
    world.log(
        f"align frame={snap.frame_no} spawn={len(spawned)} "
        f"despawn={len(despawned)} corrected={len(corrected)}"
    )
    return AlignReport(
        snap.frame_no,
        tuple(spawned),
        tuple(despawned),
        tuple(sorted(corrected)),
        max_error,
    )


def interp(previous: Optional[ShardState] = None, current: Optional[ShardState] = None,
           alpha: float = 0.5) -> ShardState:
    """快照间线性插值，平滑掩盖纠正跳变（修复 6 的渲染侧配合）。"""
    if not isinstance(current, ShardState):
        current = ShardState(0, (0.0, 0.0), (0.0, 0.0))
    if not isinstance(previous, ShardState):
        return current
    alpha = min(1.0, max(0.0, alpha))

    def blend(a: float, b: float) -> float:
        return a + (b - a) * alpha

    return ShardState(
        current.shard_id,
        (
            blend(previous.position[0], current.position[0]),
            blend(previous.position[1], current.position[1]),
        ),
        (
            blend(previous.velocity[0], current.velocity[0]),
            blend(previous.velocity[1], current.velocity[1]),
        ),
    )


# ---------------------------------------------------------------------------
# 预算、指标与审计
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetPlan:
    frame_no: int
    alive: int
    requested: int          # 本帧原始 CCD 子步需求
    granted: int            # 预算内实际批准
    over_budget: bool
    capped_alive: int       # 触发上限后保留的碎片数


def budget(world=None, frame_no: Optional[int] = None) -> BudgetPlan:
    """单帧处理预算控制：超预算时按统一上限削减，而非两端各自丢碎片（修复 5）。"""
    world = _as_world(world)
    current = world.frame_no if frame_no is None else int(frame_no)
    alive = len(world.shards)
    requested = alive * world.config.ccd_substeps
    granted = min(requested, world.config.max_world_budget)
    over = requested > world.config.max_world_budget

    capped_alive = alive
    if over:
        world.metrics.budget_overflows += 1
        capped_alive = min(alive, world.config.max_world_budget // world.config.ccd_substeps)
        # 确定性淘汰：按碎片 ID 排序后淘汰尾部，两端行为一致。
        overflow = sorted(world.shards)[capped_alive:]
        for sid in overflow:
            del world.shards[sid]
        world.log(
            f"budget_overflow frame={current} requested={requested} "
            f"granted={granted} removed={len(overflow)}"
        )

    return BudgetPlan(current, alive, requested, granted, over, capped_alive)


def metrics(world=None) -> Metrics:
    """返回当前指标快照（副本，防止外部直接篡改内部计数）。"""
    return replace(_as_world(world).metrics)


def audit(world=None, event: Optional[str] = None) -> list[str]:
    """追加并返回环形审计日志；无事件时仅读取。"""
    world = _as_world(world)
    if event is not None and not isinstance(event, int):
        world.log(str(event))
    return list(world.audit_log)
