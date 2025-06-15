import matplotlib

matplotlib.use("agg")

import base64
import io

from matplotlib.figure import Figure
from plotly import graph_objs as go  # type: ignore


def get_base64_image(fig: Figure) -> tuple[Figure, str]:
    # Create a buffer to store the plot image
    buffer = io.BytesIO()

    fig.savefig(  # type: ignore
        buffer, format="png", bbox_inches="tight", dpi=100
    )
    buffer.seek(0)

    # Encode the image as base64
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    return fig, img_str


def get_base64_image_from_plotly(fig: go.Figure) -> str:
    buffer = io.BytesIO()

    fig.write_image(buffer, format="png")  # type: ignore
    buffer.seek(0)

    # Encode the image as base64
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    return img_str


def get_base64_gif(figs: list[Figure]) -> tuple[list[Figure], str]:
    import base64
    import io

    from PIL import Image
    from PIL.ImageFile import ImageFile

    images: list[ImageFile] = []

    for fig in figs:
        buffer = io.BytesIO()

        fig.savefig(  # type: ignore
            buffer, format="png", bbox_inches="tight", dpi=100
        )
        buffer.seek(0)

        image = Image.open(buffer)

        images.append(image)

    buffer = io.BytesIO()
    images[0].save(
        buffer,
        save_all=True,
        append_images=images[1:],
        format="GIF",
        duration=500,
        loop=0,
        transparency=0,
        disposal=2,
    )

    buffer.seek(0)

    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return figs, img_str
