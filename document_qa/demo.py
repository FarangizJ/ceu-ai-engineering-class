"""Generate two fictional product manuals, including one image-only PDF."""
from pathlib import Path
import pymupdf

MANUALS = {
    "lumina": [
        "Lumina cleansing gel\nPack size: 150 mL. Refill code: LUM-42.",
        "Lumina cleansing gel\nPackaging: amber bottle. Outer carton: kraft paper.",
    ],
    "velora": [
        "Velora hand cream\nPack size: 75 mL. Refill code: VEL-19.",
        "Velora hand cream\nPackaging: blue tube. Outer carton: white paper.",
    ],
}


def main():
    folder = Path("demo_docs")
    folder.mkdir(exist_ok=True)
    for name, texts in MANUALS.items():
        with pymupdf.open() as digital, pymupdf.open() as scanned:
            for content in texts:
                page = digital.new_page()
                page.insert_text((70, 100), content, fontsize=18)
                if name == "velora":
                    image = page.get_pixmap(dpi=200).tobytes("png")
                    image_page = scanned.new_page()
                    image_page.insert_image(image_page.rect, stream=image)
            output = scanned if name == "velora" else digital
            output.save(folder / f"{name}.pdf")


if __name__ == "__main__":
    main()
