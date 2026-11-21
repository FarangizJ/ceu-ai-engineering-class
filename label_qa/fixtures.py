"""Build two fictional product sheets, including a raster-only insert."""
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
DOCS = HERE / "samples"


def create():
    DOCS.mkdir(exist_ok=True)
    with fitz.open() as catalog:
        for name, lines in [
            ("Cloud Balm", ["Cloud Balm comes in a 30 ml aluminum tin.", "Cloud Balm has an unscented finish."]),
            ("Reed Mist", ["Reed Mist comes in a 60 ml glass bottle.", "Reed Mist has a citrus scent."]),
        ]:
            page = catalog.new_page(width=440, height=190)
            page.insert_text((22, 35), name + " | product card", fontsize=16)
            page.insert_text((22, 75), "\n".join(lines), fontsize=12)
        catalog.save(DOCS / "catalog.pdf")
    with fitz.open() as master, fitz.open() as scan:
        page = master.new_page(width=440, height=190)
        page.insert_text((22, 35), "Cloud Balm | packaging insert", fontsize=16)
        page.insert_text((22, 75), "Cloud Balm refill trays are packed in sets of four.\nReturn empty refill trays in the blue mailer.", fontsize=12)
        # A faded line simulates a poor photocopy. Keep the transcript for audit.
        page.insert_text((22, 133), "Cloud Balm batch code: LUMI-81.", fontsize=7, color=(0.97,)*3)
        raster = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25))
        scan.new_page(width=440, height=190).insert_image(page.rect, stream=raster.tobytes("png"))
        scan.save(DOCS / "insert.pdf", deflate=True)


if __name__ == "__main__":
    create()
