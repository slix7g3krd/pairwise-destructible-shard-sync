# pairwise-destructible-shard-sync

## 领域

游戏、图形与多媒体

## 主题

可破坏碎片同步

## 说明

预测对齐、碎片回收与种子一致。

初始状态下 `gameplay/destructible_shard_sync.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
