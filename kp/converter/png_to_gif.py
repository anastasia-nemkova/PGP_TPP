import imageio.v2 as imageio
import os
import sys
import glob
import argparse

def create_gif_from_png(png_dir, output_gif, fps=30, scale=1.0, start=0, end=None):
    """
    Создает GIF из последовательности PNG файлов
    """
    png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    
    if not png_files:
        print(f"No PNG files found in {png_dir}")
        return False
    if end is None:
        end = len(png_files) - 1
    
    filtered_files = []
    for i, file in enumerate(png_files):
        if start <= i <= end:
            filtered_files.append(file)
    
    if not filtered_files:
        print(f"No frames in range {start}-{end}")
        return False
    
    print(f"Creating GIF from {len(filtered_files)} frames...")
    print(f"FPS: {fps}, Scale: {scale}")

    frames = []
    for i, png_file in enumerate(filtered_files):
        img = imageio.imread(png_file)
 
        if scale != 1.0:
            from skimage.transform import resize
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = resize(img, (new_h, new_w), preserve_range=True).astype('uint8')
        
        frames.append(img)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(filtered_files)} frames")

    imageio.mimsave(output_gif, frames, fps=fps, loop=0)
    print(f"\nGIF saved: {output_gif}")
    print(f"Size: {len(frames)} frames, Duration: {len(frames)/fps:.1f} seconds")
    
    return True

def create_gif_from_data_direct(input_pattern, output_gif, fps=30, scale=0.5):
    """
    Прямая конвертация .data -> GIF без промежуточных PNG
    """
    from data_to_png import read_data_file
    
    frames = []
    frame_num = 0
    
    print("Creating GIF directly from .data files...")
    
    while True:
        data_file = input_pattern % frame_num
        if not os.path.exists(data_file):
            if frame_num > 10:
                break
            frame_num += 1
            continue
        img, width, height = read_data_file(data_file)
        if img is not None:
            if scale != 1.0:
                from skimage.transform import resize
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = resize(img, (new_h, new_w), preserve_range=True).astype('uint8')
            
            frames.append(img)
            print(f"  Added frame {frame_num}")
        
        frame_num += 1
    
    if not frames:
        print("No frames found!")
        return False

    imageio.mimsave(output_gif, frames, fps=fps, loop=0)
    print(f"\nGIF saved: {output_gif} ({len(frames)} frames)")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create GIF from PNG frames')
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory with PNG files')
    parser.add_argument('--output', '-o', default='animation.gif',
                       help='Output GIF file (default: animation.gif)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor (default: 1.0)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start frame (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='End frame (default: all)')
    parser.add_argument('--direct', action='store_true',
                       help='Direct conversion from .data files (uses pattern)')
    
    args = parser.parse_args()
    
    if args.direct:
        create_gif_from_data_direct(args.input, args.output, args.fps, args.scale)
    else:
        create_gif_from_png(args.input, args.output, args.fps, args.scale, args.start, args.end)

if __name__ == "__main__":
    main()