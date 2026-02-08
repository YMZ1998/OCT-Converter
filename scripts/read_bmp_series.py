import argparse
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image


def safe_imread(path):
    """
    Safely read BMP images supporting Chinese paths.
    Returns BGR numpy array.
    
    Args:
        path (str): Path to the image file
        
    Returns:
        numpy.ndarray: BGR image array
    """
    img = Image.open(path)
    img = img.convert("RGB")  # Ensure 3 channels
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def sort_bmp_files(files):
    """
    Sort BMP files according to F1, F2, ... RF logic.
    
    Args:
        files (list): List of BMP filenames
        
    Returns:
        list: Sorted list of filenames
    """

    def sort_key(name):
        if "-RF" in name:
            return (float('inf'), 0)  # Use infinity instead of hardcoded large number
        match = re.search(r"-F(\d+)", name)
        if match:
            return (int(match.group(1)), 0)
        # Files without F# designation come before RF but after numbered F files
        return (float('inf') - 1, 0)

    return sorted(files, key=sort_key)


def bmp_sequence_to_avi_tiff(
    input_dir,
    output_avi,
    output_tiff,
    fps=2,
    show=True
):
    """
    Convert a sequence of BMP images to AVI video and TIFF image sequence.
    
    Args:
        input_dir (str): Directory containing BMP files
        output_avi (str): Output AVI file path
        output_tiff (str): Output TIFF file path
        fps (int): Frames per second for AVI output
        show (bool): Whether to display the images during processing
        
    Raises:
        RuntimeError: If no BMP files are found in the directory
    """
    # Find BMP files
    bmp_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".bmp")]
    if not bmp_files:
        raise RuntimeError("No BMP files found in directory")

    bmp_files = sort_bmp_files(bmp_files)

    print("BMP sequence order:")
    for f in bmp_files:
        print(" ", f)

    frames_tiff = []

    # Read first frame to get dimensions
    first_img_path = os.path.join(input_dir, bmp_files[0])
    try:
        first_img = safe_imread(first_img_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read first image: {first_img_path}, Error: {e}")

    print(first_img.shape)
    h, w, c = first_img.shape

    # Create AVI writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_avi, fourcc, fps, (w, h))

    if show:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.show()

    for i, fname in enumerate(bmp_files):
        path = os.path.join(input_dir, fname)
        try:
            img_bgr = safe_imread(path)
        except Exception as e:
            print(f"Warning: Skipping unreadable file: {fname}, Error: {e}")
            continue

        if img_bgr is None:
            print(f"Warning: Skipping unreadable file: {fname}")
            continue

        # Convert for TIFF (uses RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames_tiff.append(img_rgb)

        # Write to AVI
        writer.write(img_bgr)

        # Display if requested
        if show:
            ax.clear()
            ax.imshow(img_rgb)
            ax.set_title(f"{i + 1}/{len(bmp_files)} : {fname}")
            ax.axis("off")
            plt.pause(0.001)  # Small pause to update display

    if show:
        plt.ioff()  # Turn off interactive mode
        plt.close(fig)

    writer.release()

    # Write multi-page TIFF
    print(f"Writing TIFF: {output_tiff}")
    tifffile.imwrite(output_tiff, frames_tiff, photometric="rgb")

    print(f"Conversion completed successfully")
    print(f"  AVI : {output_avi}")
    print(f"  TIFF: {output_tiff}")


def test(default_input_dir):
    parent_dir = os.path.dirname(default_input_dir)
    dirname = os.path.basename(parent_dir)
    default_output_dir = "E:\Data\OCT\Result"
    default_output_avi = os.path.join(default_output_dir, f"{dirname}.avi")
    default_output_tiff = os.path.join(default_output_dir, f"{dirname}.tiff")

    parser = argparse.ArgumentParser(description="Convert BMP image sequence to AVI and TIFF")
    parser.add_argument("--input_dir", type=str, default=default_input_dir,
                        help=f"Input directory containing BMP files (default: {default_input_dir})")
    parser.add_argument("--output_avi", type=str, default=default_output_avi,
                        help=f"Output AVI file path (default: {default_output_avi})")
    parser.add_argument("--output_tiff", type=str, default=default_output_tiff,
                        help=f"Output TIFF file path (default: {default_output_tiff})")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for AVI output (default: 2)")
    parser.add_argument("--show", action="store_true", help="Display images during processing")
    parser.set_defaults(show=True)

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_avi), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_tiff), exist_ok=True)

    bmp_sequence_to_avi_tiff(
        input_dir=args.input_dir,
        output_avi=args.output_avi,
        output_tiff=args.output_tiff,
        fps=args.fps,
        show=args.show
    )


if __name__ == "__main__":
    # default_input_dir = r"E:\Data\OCT\CFP图像\KH902-R10-Certification-49-LLF-FP-OD"
    default_input_dir = r"E:\Data\OCT\CFP图像\KH902-R10-Certification-49-LLF-FP-OS"
    test(default_input_dir)
