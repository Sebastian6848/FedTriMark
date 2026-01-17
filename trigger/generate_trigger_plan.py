import random, os, json
from collections import defaultdict
import numpy as np, os
from torchvision import transforms
from trigger.generate_waffle_pattern import *
from trigger.generate_pattern import *
from fed.client import client_generate_trigger_subset
from fed.server import server_aggregate_trigger_sets

# 生成触发集分配方案
"""
    "0": [1, 7, 12, 18, 33, 47, 52, 60, 70, 85],
    "1": [0, 4, 9, 11, 22, 24, 45, 55, 64, 97],
"""
def generate_trigger_distribution_map(args, output_dir="./trigger/trigger_plan"):
    client_to_classes = dict()
    class_to_clients = defaultdict(list)

    for client_id in range(args.num_clients):
        assigned_classes = sorted(random.sample(range(args.num_classes), args.classes_per_client))
        client_to_classes[client_id] = assigned_classes
        for cls in assigned_classes:
            class_to_clients[cls].append(client_id)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "client_to_classes.json"), "w") as f:
        json.dump(client_to_classes, f, indent=2)
    with open(os.path.join(output_dir, "class_to_clients.json"), "w") as f:
        json.dump(class_to_clients, f, indent=2)

    print(f"[✓] 调度图已生成，输出到 {output_dir}")
    return client_to_classes, class_to_clients

def create_trigger(args, model):
    print("Step 1: Generating trigger plan...")
    generate_trigger_distribution_map(args)

    print("Step 2: Simulating client trigger generation...")
    for cid in range(args.num_clients):
        client_generate_trigger_subset(cid, args)

    print("Step 3: Aggregating on server...")
    trigger_dataset = server_aggregate_trigger_sets(upload_dir="./trigger/upload_buffer", adv=True, model=model, eps=0.03, device=args.device)
    print(f"[✓] Final trigger dataset size: {len(trigger_dataset)}")
    return trigger_dataset