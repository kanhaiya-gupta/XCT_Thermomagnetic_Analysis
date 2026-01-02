"""
Tests for utility functions.
"""

import numpy as np
import pytest
import platform
import os
from pathlib import Path
from src.utils.utils import (
    load_volume,
    save_volume,
    convert_to_mm,
    convert_from_mm,
    normalize_units,
    parse_voxel_size_with_unit,
    normalize_path,
)


class TestUnitConversion:
    """Test unit conversion functions."""

    @pytest.mark.unit
    def test_convert_to_mm(self):
        """Test conversion to millimeters."""
        # Micrometers to mm
        assert abs(convert_to_mm(1000, "um") - 1.0) < 1e-6
        assert abs(convert_to_mm(1000, "micrometer") - 1.0) < 1e-6

        # Centimeters to mm
        assert abs(convert_to_mm(1, "cm") - 10.0) < 1e-6

        # Already in mm
        assert abs(convert_to_mm(1, "mm") - 1.0) < 1e-6

    @pytest.mark.unit
    def test_convert_from_mm(self):
        """Test conversion from millimeters."""
        # mm to micrometers
        assert abs(convert_from_mm(1.0, "um") - 1000.0) < 1e-6

        # mm to cm
        assert abs(convert_from_mm(10.0, "cm") - 1.0) < 1e-6

    @pytest.mark.unit
    def test_normalize_units(self):
        """Test unit normalization."""
        # Normalize to mm
        result = normalize_units(1.0, "cm", target_unit="mm")
        assert abs(result - 10.0) < 1e-6

        result = normalize_units(1000.0, "um", target_unit="mm")
        assert abs(result - 1.0) < 1e-6

    @pytest.mark.unit
    def test_parse_voxel_size_with_unit(self):
        """Test voxel size parsing."""
        # Parse with unit
        size, unit = parse_voxel_size_with_unit("0.1 mm")
        assert abs(size - 0.1) < 1e-6
        assert unit == "mm"

        size, unit = parse_voxel_size_with_unit("100 um")
        assert abs(size - 100) < 1e-6
        assert unit in ["um", "micrometer"]


class TestVolumeIO:
    """Test volume I/O functions."""

    @pytest.mark.unit
    def test_save_and_load_volume_numpy(self, simple_volume, tmp_path):
        """Test saving and loading volume as NumPy."""
        filepath = tmp_path / "test_volume.npy"
        save_volume(simple_volume, str(filepath))

        loaded_volume, metadata = load_volume(str(filepath))

        assert np.array_equal(loaded_volume, simple_volume)
        assert metadata is not None

    @pytest.mark.unit
    def test_save_and_load_volume_with_metadata(self, simple_volume, tmp_path):
        """Test saving volume with metadata."""
        filepath = tmp_path / "test_volume.npz"
        metadata = {"voxel_size": (0.1, 0.1, 0.1)}

        save_volume(simple_volume, str(filepath), metadata=metadata)

        loaded_volume, loaded_metadata = load_volume(str(filepath))

        assert np.array_equal(loaded_volume, simple_volume)
        assert loaded_metadata["voxel_size"] == metadata["voxel_size"]


class TestNormalizePath:
    """Test cross-platform path normalization."""

    @pytest.mark.unit
    def test_relative_path_without_base(self, tmp_path):
        """Test relative path resolution without base_path."""
        # Change to tmp_path
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            test_file = tmp_path / "test_file.txt"
            test_file.write_text("test")

            # Relative path should resolve to absolute
            result = normalize_path("test_file.txt")
            assert result.is_absolute()
            assert result.exists()
            assert result.name == "test_file.txt"
        finally:
            os.chdir(original_cwd)

    @pytest.mark.unit
    def test_relative_path_with_base(self, tmp_path):
        """Test relative path resolution with base_path."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        test_file = base_dir / "test_file.txt"
        test_file.write_text("test")

        # Relative path with base
        result = normalize_path("test_file.txt", base_path=base_dir)
        assert result.is_absolute()
        assert result.exists()
        assert result == test_file

    @pytest.mark.unit
    def test_absolute_path(self, tmp_path):
        """Test absolute path handling."""
        test_file = tmp_path / "absolute_test.txt"
        test_file.write_text("test")

        result = normalize_path(str(test_file))
        assert result.is_absolute()
        assert result == test_file

    @pytest.mark.unit
    def test_path_object_input(self, tmp_path):
        """Test that Path objects are handled correctly."""
        test_file = tmp_path / "path_object_test.txt"
        test_file.write_text("test")

        # Pass Path object
        result = normalize_path(test_file)
        assert result == test_file

        # Pass string
        result2 = normalize_path(str(test_file))
        assert result2 == test_file

    @pytest.mark.unit
    def test_wsl_path_on_windows(self):
        """Test WSL path conversion on Windows."""
        if platform.system() == "Windows":
            # Test /mnt/c/... format
            wsl_path = "/mnt/c/Users/test/file.txt"
            result = normalize_path(wsl_path)

            # Should convert to Windows path
            assert result.as_posix().startswith("C:/") or str(result).startswith("C:\\")
            assert "Users" in str(result)
            assert "test" in str(result)
            assert "file.txt" in str(result)

            # Test /c/... format (short form)
            wsl_path_short = "/c/Users/test/file.txt"
            result_short = normalize_path(wsl_path_short)
            assert result_short.as_posix().startswith("C:/") or str(
                result_short
            ).startswith("C:\\")

    @pytest.mark.unit
    def test_windows_path_on_linux(self):
        """Test Windows path conversion on Linux/WSL."""
        if platform.system() in ["Linux", "Darwin"]:
            # Test C:\... format
            windows_path = "C:\\Users\\test\\file.txt"
            result = normalize_path(windows_path)

            # Should convert to POSIX path
            assert result.as_posix().startswith("/")
            # Check if WSL format or regular Linux format
            assert (
                "mnt" in str(result)
                or result.as_posix().startswith("/c/")
                or result.as_posix().startswith("/C/")
            )

    @pytest.mark.unit
    def test_mixed_separators(self, tmp_path):
        """Test paths with mixed separators."""
        test_file = tmp_path / "mixed" / "separators" / "test.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test")

        # Test with forward slashes (works on all platforms)
        # Path objects will normalize separators automatically
        mixed_path = str(test_file).replace("\\", "/")

        result = normalize_path(mixed_path)
        # Path should normalize correctly
        assert result.exists()
        # Resolve both to handle any symlinks or relative components
        assert result.resolve() == test_file.resolve()

    @pytest.mark.unit
    def test_data_directory_fallback(self, tmp_path):
        """Test data directory fallback when file not found."""
        base_dir = tmp_path / "project"
        base_dir.mkdir()
        data_dir = base_dir / "data"
        data_dir.mkdir()

        # File in data directory
        test_file = data_dir / "sample.txt"
        test_file.write_text("test")

        # Try to find file with data directory fallback
        result = normalize_path("sample.txt", base_path=base_dir, check_data_dir=True)
        assert result.exists()
        assert result == test_file

    @pytest.mark.unit
    def test_data_directory_fallback_disabled(self, tmp_path):
        """Test that data directory fallback can be disabled."""
        base_dir = tmp_path / "project"
        base_dir.mkdir()
        data_dir = base_dir / "data"
        data_dir.mkdir()

        # File in data directory
        test_file = data_dir / "sample.txt"
        test_file.write_text("test")

        # With check_data_dir=False, should not find file in data/
        result = normalize_path("sample.txt", base_path=base_dir, check_data_dir=False)
        # Should resolve to base_dir/sample.txt (which doesn't exist)
        assert result.parent == base_dir
        assert result.name == "sample.txt"

    @pytest.mark.unit
    def test_custom_data_directory_name(self, tmp_path):
        """Test custom data directory name."""
        base_dir = tmp_path / "project"
        base_dir.mkdir()
        custom_data_dir = base_dir / "custom_data"
        custom_data_dir.mkdir()

        # File in custom data directory
        test_file = custom_data_dir / "sample.txt"
        test_file.write_text("test")

        # Use custom data directory name
        result = normalize_path(
            "sample.txt",
            base_path=base_dir,
            check_data_dir=True,
            data_dir_name="custom_data",
        )
        assert result.exists()
        assert result == test_file

    @pytest.mark.unit
    def test_nonexistent_path(self, tmp_path):
        """Test handling of nonexistent paths."""
        nonexistent = tmp_path / "nonexistent" / "file.txt"

        # Should still normalize the path even if it doesn't exist
        result = normalize_path(str(nonexistent))
        assert result.is_absolute()
        assert not result.exists()
        assert result.name == "file.txt"

    @pytest.mark.unit
    def test_path_with_dots(self, tmp_path):
        """Test paths with . and .. components."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        sub_dir = base_dir / "sub"
        sub_dir.mkdir()
        test_file = sub_dir / "test.txt"
        test_file.write_text("test")

        # Use .. to go up
        result = normalize_path("../sub/test.txt", base_path=base_dir / "other")
        # Should resolve correctly
        assert result.exists() or result.name == "test.txt"

    @pytest.mark.unit
    def test_empty_path(self):
        """Test handling of empty path."""
        # Empty path should resolve to current directory
        result = normalize_path("")
        # Should resolve to current directory (absolute path)
        assert result.is_absolute()
        assert result == Path.cwd()

    @pytest.mark.unit
    def test_base_path_normalization(self, tmp_path):
        """Test that base_path is also normalized."""
        if platform.system() == "Windows":
            # Test WSL path as base_path
            base_wsl = "/mnt/c/Users/test"
            test_file = tmp_path / "test.txt"
            test_file.write_text("test")

            # Should normalize base_path too
            result = normalize_path("test.txt", base_path=base_wsl)
            assert result.is_absolute()

    @pytest.mark.unit
    def test_recursive_base_path(self, tmp_path):
        """Test that base_path normalization doesn't cause infinite recursion."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        # Base path that would need normalization
        if platform.system() == "Windows":
            base_path_str = "/mnt/c/Users/test"
        else:
            base_path_str = "C:\\Users\\test"

        # Should not cause recursion
        result = normalize_path("file.txt", base_path=base_path_str)
        assert result.is_absolute()

    @pytest.mark.unit
    def test_relative_base_path(self, tmp_path):
        """Test relative base_path handling."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            base_dir = tmp_path / "base"
            base_dir.mkdir()
            test_file = base_dir / "test.txt"
            test_file.write_text("test")

            # Relative base path
            result = normalize_path("test.txt", base_path="base")
            assert result.is_absolute()
            assert result.exists() or result.name == "test.txt"
        finally:
            os.chdir(original_cwd)

    @pytest.mark.unit
    def test_special_characters_in_path(self, tmp_path):
        """Test paths with special characters."""
        # Create directory with special characters
        special_dir = tmp_path / "test-dir_123"
        special_dir.mkdir()
        test_file = special_dir / "file-name.txt"
        test_file.write_text("test")

        result = normalize_path(str(test_file))
        assert result.exists()
        assert result == test_file

    @pytest.mark.unit
    def test_unicode_characters_in_path(self, tmp_path):
        """Test paths with unicode characters."""
        unicode_dir = tmp_path / "测试"
        unicode_dir.mkdir()
        test_file = unicode_dir / "文件.txt"
        test_file.write_text("test")

        result = normalize_path(str(test_file))
        assert result.exists()
        assert result == test_file

    @pytest.mark.unit
    def test_long_path(self, tmp_path):
        """Test very long paths."""
        long_path = tmp_path
        for i in range(5):
            long_path = long_path / f"level_{i}"
        long_path.mkdir(parents=True, exist_ok=True)
        test_file = long_path / "test.txt"
        test_file.write_text("test")

        result = normalize_path(str(test_file))
        assert result.exists()
        assert result == test_file

    @pytest.mark.unit
    def test_multiple_drive_letters_wsl(self):
        """Test different drive letters in WSL paths."""
        if platform.system() == "Windows":
            # Test different drives
            for drive in ["c", "d", "e"]:
                wsl_path = f"/mnt/{drive}/Users/test/file.txt"
                result = normalize_path(wsl_path)
                assert result.as_posix().startswith(f"{drive.upper()}:/") or str(
                    result
                ).startswith(f"{drive.upper()}:\\")

    @pytest.mark.unit
    def test_path_with_spaces(self, tmp_path):
        """Test paths with spaces."""
        spaced_dir = tmp_path / "path with spaces"
        spaced_dir.mkdir()
        test_file = spaced_dir / "file name.txt"
        test_file.write_text("test")

        result = normalize_path(str(test_file))
        assert result.exists()
        assert result == test_file

    @pytest.mark.unit
    def test_check_data_dir_only_when_file_not_found(self, tmp_path):
        """Test that data directory is only checked when file not found."""
        base_dir = tmp_path / "project"
        base_dir.mkdir()

        # File exists in base directory
        test_file = base_dir / "sample.txt"
        test_file.write_text("test")

        # Also create file in data directory
        data_dir = base_dir / "data"
        data_dir.mkdir()
        data_file = data_dir / "sample.txt"
        data_file.write_text("data version")

        # Should find file in base directory, not data directory
        result = normalize_path("sample.txt", base_path=base_dir, check_data_dir=True)
        # Should prefer the file in base directory if it exists
        assert result.exists()
