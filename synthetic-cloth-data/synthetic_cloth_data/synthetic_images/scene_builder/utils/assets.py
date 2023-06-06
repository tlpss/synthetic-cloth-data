import dataclasses
import json
from typing import List, Optional


@dataclasses.dataclass
class AssetConfig:
    # TODO: make a structured (pydantic?) scheme for the assets!

    """base class for Blender asset configs that load a set of assets from a json snapshot file."""
    asset_list: List[dict] = dataclasses.field(init=False)
    asset_json_path: str = None

    tags: Optional[List[str]] = None
    types: Optional[List[str]] = None  # TODO: validate these against the possible blender asset types
    max_amount: Optional[int] = None

    def __post_init__(self):
        self.asset_list = json.load(open(self.asset_json_path, "r"))["assets"]
        self.asset_list = self._filter_assets(self.asset_list)

    def _filter_assets(self, asset_list: List[dict]):
        """filter assets"""
        if self.tags is not None:
            asset_list = [asset for asset in asset_list if set(self.tags).issubset(set(asset["tags"]))]
        if self.types is not None:
            asset_list = [asset for asset in asset_list if asset["type"] in self.types]
        if self.max_amount is not None:
            asset_list = asset_list[: self.max_amount]
        return asset_list
