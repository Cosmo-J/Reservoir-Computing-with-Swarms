from PIL import Image, ImageSequence

def compress_gif(input_path, output_path, optimize=True, quality=80, resize_factor=1.0):
    with Image.open(input_path) as img:
        frames = []
        for frame in ImageSequence.Iterator(img):
            if resize_factor != 1.0:
                w, h = frame.size
                frame = frame.resize((int(w * resize_factor), int(h * resize_factor)), Image.LANCZOS)
            frames.append(frame.convert("P", palette=Image.ADAPTIVE))

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=optimize,
            quality=quality,
            loop=img.info.get("loop", 0),
            duration=img.info.get("duration", 100),
            disposal=2
        )

# Example usage:
# compress_gif("input.gif", "compressed.gif", optimize=True, quality=70_

compress_gif("gifs/death_spiral.gif","gifs/death_spiral_COMP.gif",True,quality=10,resize_factor=0.25)