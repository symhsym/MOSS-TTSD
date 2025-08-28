# parallel_preprocess.py
import os, json, argparse, pickle
import numpy as np
import glob
import torch, itertools, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context, Queue
from transformers import AutoTokenizer
from tqdm import tqdm         
import threading              
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from XY_Tokenizer.xy_tokenizer.model import XY_Tokenizer
# 复用常量/函数 -------------------------
from data_preprocess import (
    SYSTEM_PROMPT, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH,
    MAX_CHANNELS, load_tokenizer, process_inputs,
    normalize_text, load_audio_data
)
# -----------------------------------------------------

def shard_indices(num_items, world_size):
    # 轮询分片，保证各卡负载尽量均衡
    return [list(range(r, num_items, world_size)) for r in range(world_size)]

def worker_rank(rank, args, items, idx_list, progress_q):  # ★ 多传一个 progress_q
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # 每卡加载一次模型与分词器
    tokenizer, spt = load_tokenizer(args.model_path, SPT_CONFIG_PATH, SPT_CHECKPOINT_PATH)
    spt = spt.to(device).eval()

    # 分片输出
    shard_name = f"{args.data_name}.rank{rank}"
    pkl_path = os.path.join(args.output_dir, f"{shard_name}.pkl")
    meta_path = os.path.join(args.output_dir, f"{shard_name}_metas.npy")

    offsets, tokens_lengths, tims_lengths, order = [], [], [], []

    # 小线程池用于并行读盘
    io_workers = max(1, min(args.io_workers_per_gpu, 32))
    with open(pkl_path, "wb") as pf, ThreadPoolExecutor(max_workers=io_workers) as pool:
        # 预提交 IO 任务
        futures = {}
        for idx in idx_list:
            item = items[idx]
            futures[pool.submit(load_item_audio, item)] = (idx, item)

        for fut in as_completed(futures):
            idx, item = futures[fut]
            try:
                audio_kwargs, final_text = fut.result()
            except Exception as e:
                print(f"[rank{rank}] skip idx={idx} due to IO error: {e}")
                try: progress_q.put(1, block=False)    # ★ 失败也推进度
                except: pass
                continue

            try:
                # 复用你原来的 process_inputs，不改任何处理
                input_ids, labels, total_len, audio_len = process_inputs(
                    tokenizer, spt, SYSTEM_PROMPT, final_text, device, **audio_kwargs,
                    max_channels=MAX_CHANNELS
                )
            except Exception as e:
                print(f"[rank{rank}] skip idx={idx} due to encode error: {e}")
                try: progress_q.put(1, block=False)    # ★ 失败也推进度
                except: pass
                continue

            data_entry = {
                "input_ids": input_ids.tolist(),
                "labels": labels.tolist()
            }

            offsets.append(pf.tell())
            pickle.dump(data_entry, pf)
            tokens_lengths.append(total_len)
            tims_lengths.append(audio_len)

            # ★ 成功推进度
            try: progress_q.put(1, block=False)
            except: pass

    # 保存本片 metas 与顺序
    np.save(meta_path, np.stack([np.array(offsets), np.array(tokens_lengths), np.array(tims_lengths)]))
    print(f"[rank{rank}] done. {len(order)} items -> {pkl_path}")

def load_item_audio(item):
    """
    只做 IO 与文本规范化，完全复用你现有逻辑，返回给 worker 计算：
      - audio_kwargs: 传给 process_inputs 的命名参数（audio_data 或 reference_audio/main_audio）
      - final_text:   规范化+替换后的文本
    """
    if "file_path" in item and "full_transcript" in item:
        file_path = item["file_path"]
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"audio missing: {file_path}")
        audio = load_audio_data(file_path)
        full_text = item["full_transcript"]
        full_text = normalize_text(full_text)
        final_text = full_text.replace("[S1]", "<speaker1>").replace("[S2]", "<speaker2>")
        return {"audio_data": audio}, final_text

    else:
        raise ValueError("unsupported item schema")

# 新增：清理分片文件
def cleanup_shards(args):
    shard_names = [f"{args.data_name}.rank{r}" for r in range(args.gpus)]
    for n in shard_names:
        for suffix in (".pkl", "_metas.npy", "_order.npy"):
            path = os.path.join(args.output_dir, f"{n}{suffix}")
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

def merge_shards(args):
    """
    将 rank0..N-1 的分片随机交错合并为单个 pkl + metas.npy。
    不按原 JSONL 顺序；不依赖/不使用 *_order.npy。
    """
    rng = np.random.default_rng()

    shard_names = [f"{args.data_name}.rank{r}" for r in range(args.gpus)]
    metas  = [np.load(os.path.join(args.output_dir, f"{n}_metas.npy")) for n in shard_names]
    pkls   = [open(os.path.join(args.output_dir, f"{n}.pkl"), "rb") for n in shard_names]

    # 每片剩余条目数 & 已读计数（按分片写入顺序顺序读取）
    counts = [m.shape[1] for m in metas]   # metas 形状 (3, num_local)
    read_pos = [0]*args.gpus

    final_pkl = os.path.join(args.output_dir, f"{args.data_name}.pkl")
    final_meta = os.path.join(args.output_dir, f"{args.data_name}_metas.npy")
    pointers, tokens_all, tims_all = [], [], []

    # 仍有数据的分片集合
    live = [r for r, c in enumerate(counts) if c > 0]

    with open(final_pkl, "wb") as fout:
        while live:
            # 随机选择一个还有数据的分片
            r = int(rng.choice(live))
            fp = pkls[r]

            try:
                entry = pickle.load(fp)  # 顺序读一条
            except EOFError:
                # 分片异常结束：移出
                live.remove(r)
                continue

            local_pos = read_pos[r]
            pointers.append(fout.tell())
            pickle.dump(entry, fout)
            tokens_all.append(metas[r][1][local_pos])  # tokens_lengths
            tims_all.append(metas[r][2][local_pos])    # tims_lengths

            read_pos[r] += 1
            if read_pos[r] >= counts[r]:
                live.remove(r)

    np.save(final_meta, np.stack([np.array(pointers), np.array(tokens_all), np.array(tims_all)]))
    for f in pkls: f.close()
    print(f"[merge] merged into {final_pkl} & {final_meta} (random interleave)")
    # ★ 合并完成后清理分片（即使没有 *_order.npy 也不会报错）
    cleanup_shards(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_name", default="processed_data")
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--io_workers_per_gpu", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.jsonl, "r") as f:
        items = [json.loads(x) for x in f]

    parts = shard_indices(len(items), args.gpus)

    ctx = get_context("spawn")

    # ★ 进度队列与监听线程（总进度 + ETA）
    total_items = len(items)
    progress_q = ctx.Queue(maxsize=65536)

    def progress_monitor(q, total, num_workers):
        done_workers = 0
        with tqdm(total=total, desc="TOTAL", unit="it",
                  dynamic_ncols=True, smoothing=0.3) as pbar:
            while done_workers < num_workers:
                msg = q.get()
                if msg is None:
                    done_workers += 1
                else:
                    pbar.update(int(msg))

    mon = threading.Thread(target=progress_monitor,
                           args=(progress_q, total_items, args.gpus),
                           daemon=True)
    mon.start()

    # ★ 启动 worker，传入 progress_q
    procs = []
    for r in range(args.gpus):
        p = ctx.Process(target=worker_rank, args=(r, args, items, parts[r], progress_q))
        p.start(); procs.append(p)
    for p in procs: p.join()

    # ★ 告知进度线程所有 worker 已结束，并等待其退出
    for _ in range(args.gpus):
        progress_q.put(None)
    mon.join()

    merge_shards(args)

if __name__ == "__main__":
    # 一些建议的环境变量，可在外部 export 更灵活
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()


