from rembg import remove
from rembg import bg
from PIL import Image

input_path = 'group.jpg'
output_path = 'bg_removed.png'

input = Image.open(input_path)
output = remove(input,bgcolor=[255,120,31,120])
output.save(output_path)

#output = bg.apply_background_color(input,[255,255,255,128])
output.save(output_path)