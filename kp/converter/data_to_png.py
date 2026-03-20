import numpy as np
import imageio.v2 as imageio
import sys
import os
import glob

def read_data_file(filename):
    """
    Чтение .data файла в формате ЛР2:
    width (int), height (int), данные RGB (float)
    """
    try:
        with open(filename, 'rb') as f:
            width_bytes = f.read(4)
            height_bytes = f.read(4)
            
            if len(width_bytes) < 4 or len(height_bytes) < 4:
                print(f"Error: File too small {filename}")
                return None
            
            width = int.from_bytes(width_bytes, byteorder='little', signed=True)
            height = int.from_bytes(height_bytes, byteorder='little', signed=True)
            
            data = np.fromfile(f, dtype=np.float32)
            
            expected_size = width * height * 3
            if len(data) != expected_size:
                print(f"Warning: Expected {expected_size} floats, got {len(data)} in {filename}")
                if len(data) < expected_size:
                    data = np.concatenate([data, np.zeros(expected_size - len(data))])
                else:
                    data = data[:expected_size]

            img = data.reshape((height, width, 3))
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            return img, width, height
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, 0, 0

def convert_single_file(input_file, output_file):
    """Конвертирует один .data файл в PNG"""
    img, width, height = read_data_file(input_file)
    if img is not None:
        imageio.imwrite(output_file, img)
        print(f"Converted: {input_file} -> {output_file} ({width}x{height})")
        return True
    return False

def convert_sequence(input_pattern, output_dir, start=0, end=None):
    """
    Конвертирует последовательность .data файлов в PNG
    input_pattern: шаблон с %d, например 'output/img_%d.data'
    output_dir: папка для сохранения PNG
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frame_num = start
    converted_count = 0
    
    while True:
        if end is not None and frame_num > end:
            break
        data_file = input_pattern % frame_num
        
        if not os.path.exists(data_file):
            frame_num += 1
            if frame_num > start + 10:
                break
            continue

        png_file = os.path.join(output_dir, f"frame_{frame_num:04d}.png")

        if convert_single_file(data_file, png_file):
            converted_count += 1
        
        frame_num += 1
    
    print(f"\nConverted {converted_count} frames to {output_dir}/")
    return converted_count

def convert_all_in_directory(data_dir, output_dir, pattern="img_*.data"):
    """Конвертирует все .data файлы в директории"""
    os.makedirs(output_dir, exist_ok=True)

    data_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    
    if not data_files:
        print(f"No .data files found in {data_dir} with pattern {pattern}")
        return 0
    
    converted_count = 0
    
    for data_file in data_files:
        base_name = os.path.basename(data_file)
        frame_num = ''.join(filter(str.isdigit, base_name))
        if not frame_num:
            frame_num = converted_count

        png_file = os.path.join(output_dir, f"frame_{int(frame_num):04d}.png")
        
        if convert_single_file(data_file, png_file):
            converted_count += 1
    
    print(f"\nConverted {converted_count} files to {output_dir}/")
    return converted_count

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert .data files to PNG images')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input pattern (e.g., "output/img_%%d.data") or directory')
    parser.add_argument('--output', '-o', default='frames',
                       help='Output directory for PNG files (default: frames)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start frame number (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='End frame number (default: auto-detect)')
    parser.add_argument('--pattern', default='img_*.data',
                       help='File pattern for directory mode (default: img_*.data)')
    parser.add_argument('--dir-mode', action='store_true',
                       help='Treat input as directory instead of pattern')
    
    args = parser.parse_args()
    
    if args.dir_mode:
        convert_all_in_directory(args.input, args.output, args.pattern)
    else:
        convert_sequence(args.input, args.output, args.start, args.end)

if __name__ == "__main__":
    main()