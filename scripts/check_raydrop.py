"""Quick raydrop metrics check on a saved checkpoint."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from lib import dataloader
from lib.arguments import parse
from lib.gaussian_renderer import raytracing
from lib.utils.loss_utils import BinaryCrossEntropyLoss

def main():
    args = parse("configs/t4/exp_t4.yaml")
    args = parse("configs/t4/dynamic/example.yaml", args)
    args.model_path = "output/t4_tuned/test/scene_t4d1/models/ckpt_it_19000_good.pth"

    scene = dataloader.load_scene(args.source_dir, args, test=False)
    gaussians_assets = scene.gaussians_assets
    scene.training_setup(args.opt)

    (model_params, first_iter) = torch.load(args.model_path)
    scene.restore(model_params, args.opt)
    print(f"Loaded checkpoint: iter {first_iter}")

    background = torch.tensor([0, 0, 1], device="cuda").float()
    BCELoss = BinaryCrossEntropyLoss()

    results = {}
    for sensor_name, lidar in scene.train_lidars.items():
        bce_list, acc_list, prec_list, rec_list, f1_list = [], [], [], [], []
        for frame in lidar.eval_frames:
            render_pkg = raytracing(frame, gaussians_assets, lidar, background, args)
            raydrop_prob = render_pkg["raydrop"].detach()
            gt_mask = lidar.get_mask(frame).cuda()

            # BCE
            labels_idx = ~gt_mask
            labels = labels_idx.reshape(-1, 1)
            preds = raydrop_prob.reshape(-1, 1)
            bce = BCELoss(labels, preds=preds).item()

            # Classification metrics
            pred_drop = (raydrop_prob.reshape(-1) > 0.5)
            gt_drop = labels_idx.reshape(-1)
            tp = (pred_drop & gt_drop).sum().item()
            fp = (pred_drop & ~gt_drop).sum().item()
            fn = (~pred_drop & gt_drop).sum().item()
            tn = (~pred_drop & ~gt_drop).sum().item()
            acc = (tp + tn) / max(tp + fp + fn + tn, 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)

            bce_list.append(bce)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)

        # GT fill rate for context
        all_masks = [lidar.get_mask(f).float().mean().item() for f in lidar.eval_frames]
        fill_rate = np.mean(all_masks)

        results[sensor_name] = {
            "bce": np.mean(bce_list),
            "accuracy": np.mean(acc_list),
            "precision": np.mean(prec_list),
            "recall": np.mean(rec_list),
            "f1": np.mean(f1_list),
            "fill_rate": fill_rate,
            "n_frames": len(lidar.eval_frames),
        }

    print(f"\n{'sensor':<20s} {'BCE':>8s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'FillRate':>8s}")
    print("-" * 80)
    all_bce, all_acc, all_prec, all_rec, all_f1 = [], [], [], [], []
    for name, r in results.items():
        print(f"{name:<20s} {r['bce']:8.4f} {r['accuracy']:8.4f} {r['precision']:8.4f} {r['recall']:8.4f} {r['f1']:8.4f} {r['fill_rate']:8.4f}")
        all_bce.append(r['bce'])
        all_acc.append(r['accuracy'])
        all_prec.append(r['precision'])
        all_rec.append(r['recall'])
        all_f1.append(r['f1'])
    print("-" * 80)
    print(f"{'AVERAGE':<20s} {np.mean(all_bce):8.4f} {np.mean(all_acc):8.4f} {np.mean(all_prec):8.4f} {np.mean(all_rec):8.4f} {np.mean(all_f1):8.4f}")

if __name__ == "__main__":
    main()
