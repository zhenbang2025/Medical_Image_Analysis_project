"""
Generate/execute the experiment plan command list.

This script does not hardcode cluster paths and can be used locally or on server.
"""

import argparse
import subprocess


def cmd_join(parts):
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_all_dir", type=str, required=True)
    parser.add_argument("--embedding_image_dir", type=str, required=True)
    parser.add_argument("--execute", action="store_true", help="Execute commands. Otherwise print only.")
    parser.add_argument("--python", type=str, default="python")
    args = parser.parse_args()

    cmds = []
    # MLP architecture comparison (A1/A2) on best embedding to be decided by E groups.
    cmds.append(cmd_join([args.python, "scripts/train_bbox_mlp.py",
                          "--embedding_dir", args.embedding_all_dir,
                          "--arch", "plain", "--loss", "smoothl1",
                          "--run_name", "A1_plain_smoothl1"]))
    cmds.append(cmd_join([args.python, "scripts/train_bbox_mlp.py",
                          "--embedding_dir", args.embedding_all_dir,
                          "--arch", "residual", "--loss", "smoothl1",
                          "--run_name", "A2_residual_smoothl1"]))

    # Loss comparison (L1/L2/L3), run on chosen embedding+arch manually after E/A conclusions.
    for loss in ("smoothl1", "iou", "ciou"):
        cmds.append(cmd_join([args.python, "scripts/train_bbox_mlp.py",
                              "--embedding_dir", args.embedding_all_dir,
                              "--arch", "residual", "--loss", loss,
                              "--run_name", f"L_{loss}"]))

    print("Planned commands:")
    for c in cmds:
        print(c)

    if args.execute:
        for c in cmds:
            subprocess.run(c, shell=True, check=True)


if __name__ == "__main__":
    main()
