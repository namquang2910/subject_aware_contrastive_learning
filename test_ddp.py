# test_ddp.py
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    
    print(f"[rank {rank}] checkpoint 1: init done", flush=True)
    dist.barrier()
    
    print(f"[rank {rank}] checkpoint 2: barrier passed", flush=True)
    dist.barrier()

    t = torch.tensor([rank * 1.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] checkpoint 3: all_reduce result = {t.item()}", flush=True)
    dist.barrier()
    
    print(f"[rank {rank}] ALL DONE", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()