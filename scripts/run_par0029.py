#!/usr/bin/env python3
import argparse
from pathlib import Path
import itk


def read_image(path: Path, pixel_type=itk.F):
    img = itk.imread(str(path))
    return itk.cast_image_filter(img, ttype=[type(img), itk.Image[pixel_type, img.GetImageDimension()]])


def read_mask(path: Path):
    img = itk.imread(str(path))
    return itk.cast_image_filter(img, ttype=[type(img), itk.Image[itk.UC, img.GetImageDimension()]])


def main():
    parser = argparse.ArgumentParser(description="Run Par0029 step2+step3 on 3D NIfTI using itk-elastix.")
    parser.add_argument("--fixed", required=True, type=Path, help="Path to fixed 3D NIfTI (.nii or .nii.gz)")
    parser.add_argument("--moving", required=True, type=Path, help="Path to moving 3D NIfTI (.nii or .nii.gz)")
    parser.add_argument("--out", required=True, type=Path, help="Output directory (will be created)")
    parser.add_argument("--step2", type=Path, default=Path("ElastixModelZoo/models/Par0029/Par0029-step2-bspline.txt"),
                        help="Path to Par0029 step2 registration parameter file (use a registration map, not a transform-only file)")
    parser.add_argument("--step3", type=Path, default=Path("ElastixModelZoo/models/Par0029/Par0029-step3-bspline.txt"),
                        help="Path to Par0029 step3 parameter file")
    parser.add_argument("--fixed-mask", type=Path, default=None, help="Optional fixed mask (same space as fixed)")
    parser.add_argument("--moving-mask", type=Path, default=None, help="Optional moving mask (same space as moving)")
    parser.add_argument("--initial-transform", type=Path, default=None, help="Optional initial transform parameter file (.txt)")
    parser.add_argument("--log", action="store_true", help="Also write elastix logs to files in output directory")

    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    fixed = read_image(args.fixed)
    moving = read_image(args.moving)

    dim = fixed.GetImageDimension()
    if dim != 3:
        raise SystemExit(f"Expected 3D data; got dimension={dim}")

    po = itk.ParameterObject.New()
    try:
        po.AddParameterFile(str(args.step2))
        po.AddParameterFile(str(args.step3))
    except Exception as e:
        raise SystemExit(f"Failed to add parameter files: {e}")

    kwargs = {
        "parameter_object": po,
        "log_to_console": True,
        "output_directory": str(args.out),
        "log_to_file": bool(args.log),
    }

    if args.initial_transform is not None:
        kwargs["initial_transform_parameter_file_name"] = str(args.initial_transform)

    if args.fixed_mask is not None:
        kwargs["fixed_mask"] = read_mask(args.fixed_mask)
    if args.moving_mask is not None:
        kwargs["moving_mask"] = read_mask(args.moving_mask)

    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed, moving, **kwargs
    )

    out_image_path = args.out / "registered_step2_step3.nii.gz"
    itk.imwrite(result_image, str(out_image_path))

    print("Saved registered image:", out_image_path)
    print("Transform parameters (per stage) in:", args.out)
    print(" -", args.out / "TransformParameters.0.txt")
    print(" -", args.out / "TransformParameters.1.txt")


if __name__ == "__main__":
    main()
