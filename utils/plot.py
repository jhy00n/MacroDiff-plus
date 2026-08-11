import matplotlib.pyplot as plt
import imageio
import io


def plot_design(pos, size, max_size, save_dir=None, design=None, time=None):
    pos_cpu = pos.cpu().numpy()
    size_cpu = size.cpu().numpy()
    max_size_cpu = max_size.cpu().numpy()

    fig, ax = plt.subplots()

    for i in range(pos.shape[0]):
        x = pos_cpu[i, 0].item()
        y = pos_cpu[i, 1].item()
        width = size_cpu[i, 0].item()
        height = size_cpu[i, 1].item()
        rect = plt.Rectangle((x, y), width, height, facecolor='blue', edgecolor='none', alpha=0.5)
        ax.add_patch(rect)

    ax.set_xlim(0, max_size_cpu[0])
    ax.set_ylim(0, max_size_cpu[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f'{design} - Step {time}')

    if save_dir is not None:
        filename = f'{save_dir}/{design}.png' if time is None else f'{save_dir}/{design}_{time}.png'
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_gif_design(pos_list, size, save_dir=None, design=None):
    size_cpu = size.cpu()

    frames = []
    for t, pos in enumerate(pos_list):
        fig, ax = plt.subplots()
        pos_cpu = pos.cpu()

        for i in range(pos_cpu.shape[0]):
            x, y = pos_cpu[i]
            w, h = size_cpu[i]
            rect = plt.Rectangle((x, y), w, h, facecolor='blue', edgecolor='none', alpha=0.5)
            ax.add_patch(rect)

        border = plt.Rectangle((-1, -1), 2, 2,
                               fill=False,
                               edgecolor='black',
                               linewidth=1)
        ax.add_patch(border)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal', adjustable='box')

        #ticks = [-2, -1, 0, 1, 2]
        ax.set_xticks([])
        ax.set_yticks([])

        #plt.title(f"Step {t}")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    imageio.mimsave(f'{save_dir}/{design}.gif', frames, duration=50)
    print(f"Saved GIF to {save_dir}/{design}.gif")