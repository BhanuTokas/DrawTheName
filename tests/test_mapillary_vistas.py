import numpy as np
import pytest
from PIL import Image

from drawthename.data.mapillary_vistas import MapillaryVistasDataset


def _write_split(root, split, image_id, labels_subdir):
    images_dir = root / split / "images"
    labels_dir = root / split / labels_subdir
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
        images_dir / f"{image_id}.jpg"
    )
    Image.fromarray(np.full((4, 4), 13, dtype=np.uint8)).save(
        labels_dir / f"{image_id}.png"
    )


def _write_continents(path, image_id):
    path.write_text(f"Mapillary_ID,continent\n{image_id},Europe\n")


@pytest.mark.parametrize("labels_subdir", ["labels", "v1.2/labels"])
def test_dataset_finds_labels_in_either_release_layout(tmp_path, labels_subdir):
    _write_split(tmp_path, "training", "abc", labels_subdir)
    continents = tmp_path / "continents.csv"
    _write_continents(continents, "abc")

    dataset = MapillaryVistasDataset(
        tmp_path, continent_labels_path=continents, splits=("training",)
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.continent == "Europe"
    assert (sample.ground_truth == 0).all()  # raw Vistas Road (13) -> trainId 0
