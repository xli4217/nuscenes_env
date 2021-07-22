import code
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch


class COLORS:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    END = "\033[0m"


PRIMITIVE_TYPES = (int, float, bool, str, type)


def pretty_print(contents: dict):
    """ Prints a nice summary of the top-level contents in a checkpoint dictionary. """
    col_size = max(len(str(k)) for k in contents)
    for k, v in sorted(contents.items()):
        key_length = len(str(k))
        line = " " * (col_size - key_length)
        line += f"{k}: {COLORS.BLUE}{type(v).__name__}{COLORS.END}"
        if isinstance(v, PRIMITIVE_TYPES):
            line += f" = "
            line += f"{COLORS.CYAN}{repr(v)}{COLORS.END}"
        elif isinstance(v, Sequence):
            line += ", "
            line += f"{COLORS.CYAN}len={len(v)}{COLORS.END}"
        elif isinstance(v, torch.Tensor):
            if v.ndimension() in (0, 1) and v.numel() == 1:
                line += f" = "
                line += f"{COLORS.CYAN}{v.item()}{COLORS.END}"
            else:
                line += ", "
                line += f"{COLORS.CYAN}shape={list(v.shape)}{COLORS.END}"
                line += ", "
                line += f"{COLORS.CYAN}dtype={v.dtype}{COLORS.END}"
        print(line)


def get_attribute(obj: object, name: str) -> object:
    if isinstance(obj, Mapping):
        return obj[name]
    if isinstance(obj, Namespace):
        return obj.name
    return getattr(object, name)
    

def peek(filepath=None, args: Namespace=None, interactive=False):
    if args is not None:
        file = Path(args.file).absolute()
    else:
        file = filepath
    ckpt = torch.load(file, map_location=torch.device("cpu"))
    selection = dict()

    if args is not None:
        attribute_names = args.attributes or list(ckpt.keys())
    else:
        attribute_names = list(ckpt.keys())
    for name in attribute_names:
        parts = name.split("/")
        current = ckpt
        for part in parts:
            current = get_attribute(current, part)
        selection.update({name: current})
    pretty_print(selection)

    if interactive:
        import ipdb; ipdb.set_trace()
    #if args.interactive:
    # if interactive or args.interactive:
    #     code.interact(
    #         banner="Entering interactive shell. You can access the checkpoint contents through the local variable 'checkpoint'.",
    #         local={"checkpoint": ckpt, "torch": torch},
    #     )
        

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "file",
        type=str,
        help="The checkpoint file to inspect. Must be a pickle binary saved with 'torch.save'.",
    )
    parser.add_argument(
        "attributes",
        nargs="*",
        help="Name of one or several attributes to query. To access an attribute within a nested structure, use '/' as separator.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Drops into interactive shell after printing the summary.",
    )
    args = parser.parse_args()
    peek(args=args)


if __name__ == "__main__":
    main()
