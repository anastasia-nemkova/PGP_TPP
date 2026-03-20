echo "=== Ray Tracing Converter ==="
echo ""

echo "1. Installing dependencies..."
python3 converter/install_deps.py

mkdir -p frames animations

echo ""
echo "2. Converting .data files to PNG..."
python3 converter/data_to_png.py \
  --input "output/img_%d.data" \
  --output frames \
  --start 0

echo ""
echo "3. Creating GIF animation..."
python3 converter/png_to_gif.py \
  --input frames \
  --output animations/output1.gif \
  --fps 30 \
  --scale 0.5

echo ""
echo "=== Conversion complete! ==="
echo "PNG frames: frames/"
echo "Animations: animations/"