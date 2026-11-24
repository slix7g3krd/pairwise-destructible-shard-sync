"""六个修复点的端到端行为验证（不替代原契约测试，只做语义佐证）。"""

import gameplay.destructible_shard_sync as mod
from gameplay.destructible_shard_sync import (
    DeterministicRng,
    FractureSeed,
    Shard,
    ShardConfig,
    ShardState,
    Snapshot,
    SyncWorld,
)


def make_seed(event_id=1, battle_id=77, frame_no=10, piece_count=8):
    rng = DeterministicRng(battle_id, frame_no, event_id)
    return FractureSeed(event_id, battle_id, frame_no, rng.next_u64(), piece_count)


# -- 修复 1：碎裂种子由服务端下发 ------------------------------------------

def test_seed_must_come_from_server_and_local_fracture_is_refused():
    world = SyncWorld(battle_id=77)

    # 没有权威种子时，客户端只能请求重同步，不允许本地自行碎裂。
    plan = mod.fallback(world, reason="seed_missing")
    assert plan["local_fracture_allowed"] is False
    assert world.metrics.fallback_resyncs == 1

    # 非法种子被拒，世界不接受。
    bad = mod.seed_sync(world, FractureSeed(2, 77, 10, -1, 4))
    assert bad.ok is False
    assert world.server_seed is None

    # 合法种子由服务端下发后才生效。
    seed = make_seed()
    ok = mod.seed_sync(world, seed)
    assert ok.ok is True
    assert world.server_seed == seed


# -- 修复 2：回收按统一存活时间 --------------------------------------------

def test_recycle_uses_unified_ttl_on_both_ends():
    cfg = ShardConfig(ttl_seconds=1.0, fixed_dt=1 / 10)  # ttl_frames = 10
    server = SyncWorld(battle_id=1, config=cfg)
    client = SyncWorld(battle_id=1, config=cfg)
    seed = make_seed(frame_no=5, piece_count=4)

    mod.seed_sync(server, seed)
    mod.seed_sync(client, seed)
    # 两端各自独立碎裂，但种子与配置相同。
    server_shards = mod._fracture(server, seed, (0.0, 0.0))
    client_shards = mod._fracture(client, seed, (0.0, 0.0))
    assert sorted(s.shard_id for s in server_shards) == sorted(
        s.shard_id for s in client_shards
    )

    # 到期帧之前两端都不回收；到期帧同一帧回收同一批碎片。
    assert mod.shard_recycle(server, frame_no=14) == []
    assert mod.shard_recycle(client, frame_no=14) == []
    assert mod.shard_recycle(server, frame_no=15) == mod.shard_recycle(client, frame_no=15)
    assert server.shards == {} and client.shards == {}
    assert mod.lifetime(server_shards[0], cfg) == 1.0


# -- 修复 3：高速碎片连续碰撞，禁止隧穿 ------------------------------------

def test_ccd_prevents_tunneling_while_naive_step_tunnels():
    cfg = ShardConfig(fixed_dt=1 / 30, ccd_substeps=8)
    fast = Shard(1, 0, (0.0, 0.5), (0.0, -100.0), 0.05, 0, cfg.ttl_frames)

    # 整帧位移 3.33m，远超自身 0.05m —— 离散单步必然穿过 y=0 地面。
    naive_y = fast.position[1] + fast.velocity[1] * cfg.fixed_dt
    assert naive_y < 0.0

    result = mod.tunneling(fast, cfg)
    assert result.substeps > 1                 # 触发了子步扫掠
    assert result.position[1] >= fast.size    # 被夹在接触面，没有穿地
    assert result.tunnelled is False
    assert result.ground_contacts >= 1


# -- 修复 4：随机数由帧号与战斗 ID 派生，重放确定 ---------------------------

def test_rng_is_derived_and_replay_is_identical():
    a = [DeterministicRng(77, 10, 1).next_u64() for _ in range(5)]
    b = [DeterministicRng(77, 10, 1).next_u64() for _ in range(5)]
    assert a == b  # 同战斗同帧同用途：重放逐值一致

    # 不同帧派生不同序列，杜绝“随机数取自系统时间”式的分叉。
    other_frame = [DeterministicRng(77, 11, 1).next_u64() for _ in range(5)]
    other_battle = [DeterministicRng(78, 10, 1).next_u64() for _ in range(5)]
    assert a != other_frame and a != other_battle

    # 同一场战斗两次碎裂重放，碎片 ID/位置/速度完全一致。
    seed = make_seed(event_id=3, frame_no=2, piece_count=6)
    run1 = mod._fracture(SyncWorld(battle_id=77), seed, (1.0, 2.0))
    run2 = mod._fracture(SyncWorld(battle_id=77), seed, (1.0, 2.0))
    assert run1 == run2


# -- 修复 5：上限走同一配置 ------------------------------------------------

def test_cap_mismatch_is_detected_and_overflow_is_deterministic():
    report = mod.cap_match(ShardConfig(max_shards=128), ShardConfig(max_shards=256))
    assert report.matched is False
    assert "max_shards" in report.mismatched_fields
    assert mod.cap_match(ShardConfig(), ShardConfig()).matched is True

    # 大批量破坏：两端同配置，超限后被丢弃的碎片完全一致。
    cfg = ShardConfig(max_shards=4, max_pieces_per_event=10)
    seed = make_seed(piece_count=10)
    left = mod._fracture(SyncWorld(config=cfg), seed, (0.0, 0.0))
    right = mod._fracture(SyncWorld(config=cfg), seed, (0.0, 0.0))
    assert len(left) == len(right) == 4
    assert [s.shard_id for s in left] == [s.shard_id for s in right]


# -- 修复 6：碎片状态按快照纠正 --------------------------------------------

def test_client_state_is_corrected_by_authoritative_snapshot():
    server = SyncWorld(battle_id=77)
    seed = make_seed(piece_count=3)
    mod.seed_sync(server, seed)
    mod._fracture(server, seed, (0.0, 0.0))
    snap = mod.authority(server, target_frame=12)
    assert snap.frame_no == 12

    # 客户端预测漂移：挪走一个、改错一个、多一个幽灵碎片。
    client = SyncWorld(battle_id=77)
    ids = [st.shard_id for st in snap.states]
    kept = server.shards[ids[0]]
    client.shards[ids[0]] = Shard(
        ids[0], kept.event_id, (99.0, 99.0), (0.0, 0.0),
        kept.size, kept.spawn_frame, kept.expiry_frame,
    )
    ghost = Shard(424242, -1, (1.0, 1.0), (0.0, 0.0), 0.1, 0, 100)
    client.shards[ghost.shard_id] = ghost

    report = mod.predict_align(client, snap)
    assert report.corrected == (ids[0],)
    assert report.spawned == tuple(sorted(ids[1:]))
    assert report.despawned == (ghost.shard_id,)
    assert report.max_position_error > 1.0
    # 对齐后客户端与服务端快照逐碎片一致。
    aligned = mod.snapshot(client)
    assert aligned.states == snap.states

    # 插值在两份快照间平滑过渡。
    p = ShardState(ids[0], (0.0, 0.0), (0.0, 0.0))
    c = ShardState(ids[0], (10.0, 20.0), (2.0, 4.0))
    mid = mod.interp(p, c, 0.5)
    assert mid.position == (5.0, 10.0) and mid.velocity == (1.0, 2.0)


# -- 预算与审计的确定性 -----------------------------------------------------

def test_budget_and_audit_behave_deterministically():
    cfg = ShardConfig(max_shards=8, ccd_substeps=4, max_world_budget=8)
    world = SyncWorld(config=cfg)
    seed = make_seed(piece_count=8)
    mod._fracture(world, seed, (0.0, 0.0))

    plan = mod.budget(world, frame_no=1)  # 需求 32 > 预算 8
    assert plan.over_budget is True
    assert plan.capped_alive == 2
    assert world.metrics.budget_overflows == 1
    assert any(e.startswith("budget_overflow") for e in mod.audit(world))

    m = mod.metrics(world)
    assert m.shards_spawned == 8 and m.shards_dropped_by_cap == 0
