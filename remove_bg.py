"""Remove image backgrounds with BRIA RMBG-2.0 or RMBG-3.0.

    python remove_bg.py photo.jpg                  # RMBG-2.0 (default)
    python remove_bg.py photo.jpg --model 3.0      # RMBG-3.0
    python remove_bg.py *.jpg -o cutouts --matte   # batch, also save the mattes

Each input produces `<name>_cutout.png`: the original image with the predicted
matte as its alpha channel.

The two models are very different and have separate requirements:

* RMBG-2.0 is a BiRefNet segmentation network (0.2B parameters). It is fast and
  runs on small GPUs, Apple silicon and CPU. Install `requirements.txt`.
* RMBG-3.0 is a Fibo 2.0 fine-tune (8B parameters) run through Diffusers. It
  needs a CUDA GPU with 32 GB or more. Install `requirements-rmbg3.txt`.

Both sets of weights are gated on Hugging Face. Accept the access form on the
model page (for RMBG-3.0 also on briaai/Fibo-2.0-edit-turbo), then log in with
`hf auth login`.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

RMBG2_REPO = "briaai/RMBG-2.0"
RMBG2_SIZE = (1024, 1024)

RMBG3_REPO = "briaai/RMBG-3.0"
RMBG3_BASE = "briaai/Fibo-2.0-edit-turbo"
RMBG3_FONTS = ("Aileron-Regular.otf", "DejaVuSans.ttf")
# RMBG-3.0 was trained on this exact text and nothing else. A changed or missing
# instruction gives a coloured picture instead of a matte, so do not edit it.
RMBG3_INSTRUCTION = (
    "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the "
    "background to black. Produce soft, anti-aliased grayscale gradients at the edges of the "
    "subject to represent fine details and transparency."
)
RMBG3_STEPS = 4  # distilled to 4 steps
RMBG3_GUIDANCE = 1.0  # no classifier-free guidance


def pick_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_rmbg2(device):
    """Return a function mapping an RGB PIL image to its matte (mode "L")."""
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(RMBG2_REPO, trust_remote_code=True)
    torch.set_float32_matmul_precision("high")
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(RMBG2_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def predict(image):
        batch = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(batch)[-1].sigmoid().cpu()[0].squeeze()
        return transforms.ToPILImage()(pred).resize(image.size)

    return predict


def load_rmbg3(device, seed):
    """Return a function mapping an RGB PIL image to its matte (mode "L")."""
    import torch
    from diffusers import AutoModel, BriaFibo2Pipeline
    from huggingface_hub import hf_hub_download

    if not device.startswith("cuda"):
        print(
            f"warning: RMBG-3.0 is only measured on CUDA GPUs with 32 GB or more; "
            f"running on '{device}' may be very slow or run out of memory.",
            file=sys.stderr,
        )

    # The RMBG-3.0 repo holds only the fine-tuned transformer. The text encoder,
    # VAE and fonts come from Fibo 2.0 Edit Turbo.
    transformer = AutoModel.from_pretrained(
        RMBG3_REPO, subfolder="transformer", dtype=torch.bfloat16
    )
    pipe = BriaFibo2Pipeline.from_pretrained(
        RMBG3_BASE, transformer=transformer, dtype=torch.bfloat16
    ).to(device)
    fonts = [hf_hub_download(RMBG3_BASE, name, subfolder="fonts") for name in RMBG3_FONTS]
    prompt = json.dumps({"edit_instruction": RMBG3_INSTRUCTION})
    generator_device = device if device.startswith("cuda") else "cpu"

    def predict(image):
        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=RMBG3_STEPS,
            guidance_scale=RMBG3_GUIDANCE,
            generator=torch.Generator(generator_device).manual_seed(seed),
            fonts=fonts,
        ).images[0]
        return result.convert("L").resize(image.size, Image.LANCZOS)

    return predict


def load_image(path):
    """Open an image as RGB, upright, with any existing transparency dropped."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def apply_matte(image, matte):
    """Return `image` as RGBA with `matte` (white = keep) as the alpha channel."""
    cutout = image.copy()
    cutout.putalpha(matte)
    return cutout


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove image backgrounds with BRIA RMBG-2.0 or RMBG-3.0."
    )
    parser.add_argument("images", nargs="+", type=Path, help="input image file(s)")
    parser.add_argument(
        "--model",
        choices=["2.0", "3.0"],
        default="2.0",
        help="2.0: fast BiRefNet segmentation (default). "
        "3.0: best quality on hard images, needs a 32 GB+ CUDA GPU.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="where to write results (default: next to each input)",
    )
    parser.add_argument(
        "--matte", action="store_true", help="also save the grayscale matte as <name>_matte.png"
    )
    parser.add_argument(
        "--device", default="auto", help="cuda, mps, cpu or auto (default: auto)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed, RMBG-3.0 only (default: 0)"
    )
    args = parser.parse_args(argv)

    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        parser.error("input file(s) not found: " + ", ".join(missing))
    return args


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    predict = load_rmbg3(device, args.seed) if args.model == "3.0" else load_rmbg2(device)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        image = load_image(path)
        matte = predict(image)
        out_dir = args.output_dir or path.parent
        cutout_path = out_dir / f"{path.stem}_cutout.png"
        apply_matte(image, matte).save(cutout_path)
        print(cutout_path)
        if args.matte:
            matte_path = out_dir / f"{path.stem}_matte.png"
            matte.save(matte_path)
            print(matte_path)


if __name__ == "__main__":
    main()
