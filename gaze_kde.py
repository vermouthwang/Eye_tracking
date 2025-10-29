#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with x_norm,y_norm")
    ap.add_argument("--out", default=None, help="Optional output image path")
    ap.add_argument("--bw", type=float, default=0.5, help="Bandwidth adjust")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    x, y = df["x_norm"], df["y_norm"]

    fig = plt.figure(figsize=(16,9), facecolor="Black")
    ax = fig.add_subplot(111,facecolor="Black")
    ax.set_aspect("auto")
    sns.kdeplot(x=x, y=y, fill=True, cmap="viridis", bw_adjust=args.bw, thresh=0.05)
    
    ax.set_xlim(0, 1)
    
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.margins(0)
    ax.set_position([0,0,1,1])

    ax.invert_yaxis()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight",pad_inches=0,transparent=False)
        print(f"[OK] Saved to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

# # #!/usr/bin/env python3
# import argparse
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="CSV with x_norm,y_norm")
#     ap.add_argument("--out", default=None, help="Optional output image path")
#     ap.add_argument("--bw", type=float, default=0.5, help="Bandwidth adjust")
#     args = ap.parse_args()

#     df = pd.read_csv(args.csv)
#     x, y = df["x_norm"], df["y_norm"]

#     plt.figure(figsize=(16,9))
#     sns.kdeplot(x=x, y=y, fill=True, cmap="viridis", bw_adjust=args.bw, thresh=0.05)
    
#     plt.xlim(0,1)
#     # ax.set_y(1,0)
    
#     # ax.set_axis_off()
#     # for spine in ax.spines.values():
#     #     spine.set_visible(False)
#     # plt.margins(0)
#     # ax.set_position([0,0,1,1])
#     plt.title("Gaze Heatmap (KDE)")
#     plt.xlabel("x_norm")
#     plt.ylabel("y_norm")
#     plt.gca().invert_yaxis()

#     if args.out:
#         plt.savefig(args.out, dpi=150, bbox_inches="tight",pad_inches=0,transparent=True)
#         print(f"[OK] Saved to {args.out}")
#     else:
#         plt.show()

# if __name__ == "__main__":
#     main()