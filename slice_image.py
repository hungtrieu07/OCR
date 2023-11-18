from PIL import Image

# Load the image
image = Image.open("output/piece1.jpg")

# Get the dimensions of the original image
width, height = image.size

# Calculate the dimensions for the two portrait pieces
piece1 = image.crop((0, 0, width // 2 - 10, height))
piece2 = image.crop((width // 2-20, 0, width, height))

# Save the two pieces
piece1.save("output/piece1_1.jpg")
piece2.save("output/piece1_2.jpg")