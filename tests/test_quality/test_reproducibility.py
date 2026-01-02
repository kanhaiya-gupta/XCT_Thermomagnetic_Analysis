"""
Tests for reproducibility module.
"""

import numpy as np
import pytest
from pathlib import Path
from src.quality.reproducibility import (
    AnalysisConfig,
    ProvenanceTracker,
    SeedManager,
    save_analysis_config,
    load_analysis_config,
)


class TestAnalysisConfig:
    """Test AnalysisConfig class."""

    @pytest.mark.unit
    def test_analysis_config_creation(self):
        """Test creating analysis configuration."""
        config = AnalysisConfig(
            segmentation_method="otsu",
            voxel_size=(0.1, 0.1, 0.1),
            filters={"min_volume": 100},
        )

        assert config.segmentation_method == "otsu"
        assert config.voxel_size == (0.1, 0.1, 0.1)
        assert config.filters["min_volume"] == 100

    @pytest.mark.unit
    def test_analysis_config_to_dict(self):
        """Test converting config to dictionary."""
        config = AnalysisConfig(segmentation_method="otsu", voxel_size=(0.1, 0.1, 0.1))

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["segmentation_method"] == "otsu"


class TestProvenanceTracker:
    """Test ProvenanceTracker class."""

    @pytest.mark.unit
    def test_provenance_tracker_basic(self):
        """Test basic provenance tracking."""
        tracker = ProvenanceTracker()

        tracker.add_step("segmentation", {"method": "otsu"})
        tracker.add_step("filtering", {"min_volume": 100})

        history = tracker.get_history()

        assert len(history) == 2
        assert history[0]["step"] == "segmentation"
        assert history[1]["step"] == "filtering"

    @pytest.mark.unit
    def test_provenance_tracker_export(self):
        """Test exporting provenance."""
        tracker = ProvenanceTracker()
        tracker.add_step("segmentation", {"method": "otsu"})

        provenance_dict = tracker.export()

        assert "history" in provenance_dict
        assert "timestamp" in provenance_dict


class TestSeedManager:
    """Test SeedManager class."""

    @pytest.mark.unit
    def test_seed_manager_basic(self):
        """Test basic seed management."""
        manager = SeedManager(seed=42)

        # Set seed
        manager.set_seed()

        # Generate random numbers
        r1 = np.random.rand()

        # Reset seed
        manager.set_seed()
        r2 = np.random.rand()

        # Should be the same with same seed
        assert r1 == r2

    @pytest.mark.unit
    def test_seed_manager_get_seed(self):
        """Test getting current seed."""
        manager = SeedManager(seed=42)
        seed = manager.get_seed()

        assert seed == 42


class TestConfigIO:
    """Test configuration save/load."""

    @pytest.mark.unit
    def test_save_load_config(self, tmp_path):
        """Test saving and loading configuration."""
        config = AnalysisConfig(segmentation_method="otsu", voxel_size=(0.1, 0.1, 0.1))

        filepath = tmp_path / "config.json"
        save_analysis_config(config, str(filepath))

        loaded_config = load_analysis_config(str(filepath))

        assert loaded_config.segmentation_method == config.segmentation_method
        assert loaded_config.voxel_size == config.voxel_size
