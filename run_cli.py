from PIL import Image
import sys
from inference import Captioner

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_cli.py <path_to_image>")
        return
    path = sys.argv[1]
    img = Image.open(path)
    cap = Captioner().caption(img)
    print(cap)

if __name__ == "__main__":
    main()
