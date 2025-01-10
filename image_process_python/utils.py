import matplotlib.pyplot as plt


# Define a function using matplotlib to display the images
def show_image(image, title="Image", cmap_type="gray"):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_image_contour(image, contours):
    plt.figure()
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    plt.imshow(image, interpolation="nearest", cmap="gray_r")
    plt.title("Contours")
    plt.axis("off")
    plt.show()


def show_image_with_corners(image, coords, title="Corners detected"):
    plt.figure(figsize=(12, 10))
    plt.imshow(image, interpolation="nearest", cmap="gray")
    plt.title(title)
    plt.plot(coords[:, 1], coords[:, 0], "+r", markersize=35)
    plt.axis("off")
    plt.show()
    plt.close()


def show_detected_face(result, detected, title="Face image"):
    plt.figure()
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap("gray")
    plt.title(title)
    plt.axis("off")

    for patch in detected:

        img_desc.add_patch(
            patches.Rectangle(
                (patch["c"], patch["r"]),
                patch["width"],
                patch["height"],
                fill=False,
                color="r",
                linewidth=2,
            )
        )
    plt.show()
    crop_face(result, detected)


def getFaceRectangle(d, group_image):
    """Extracts the face from the image using the coordinates of the detected image"""
    # X and Y starting points of the face rectangle
    x, y = d["r"], d["c"]

    # The width and height of the face rectangle
    width, height = d["r"] + d["width"], d["c"] + d["height"]

    # Extract the detected face
    face = group_image[x:width, y:height]
    return face


def mergeBlurryFace(d, original, gaussian_image):
    # X and Y starting points of the face rectangle
    x, y = d["r"], d["c"]
    # The width and height of the face rectangle
    width, height = d["r"] + d["width"], d["c"] + d["height"]

    original[x:width, y:height] = gaussian_image
    return original
