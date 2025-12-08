import os
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists. If not, create it.
    """
    os.makedirs(path, exist_ok=True)


def save_figure(fig, filename: str, folder: str = "visuals") -> None:
    """
    Save a matplotlib figure to the specified folder.
    """
    ensure_dir(folder)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def print_separator(title: str = "", char: str = "-") -> None:
    """
    Print a visual separator with an optional title, for cleaner console logs.
    """
    line = char * 60
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(f"\n{line}")
