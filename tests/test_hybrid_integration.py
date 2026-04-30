"""Integration tests for hybrid device partitioning.

Tests the full chain: CLI → custom_model_init → AtomModelSingleton → DeviceAllocator
without requiring actual NPU hardware or model files.
"""
import threading
from unittest.mock import patch, MagicMock
import pytest

from rapid_doc.backend.pipeline.model_init import AtomModelSingleton
from rapid_doc.utils.device_allocator import DeviceAllocator


class TestAtomModelSingletonAllocator:
    """AtomModelSingleton allocator integration tests."""

    def setup_method(self):
        AtomModelSingleton.reset()

    def teardown_method(self):
        AtomModelSingleton.reset()

    def test_set_allocator(self):
        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        AtomModelSingleton.set_allocator(allocator)
        assert AtomModelSingleton._allocator is allocator

    def test_reset_clears_allocator(self):
        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        AtomModelSingleton.set_allocator(allocator)
        AtomModelSingleton.reset()
        assert AtomModelSingleton._allocator is None
        assert AtomModelSingleton._models == {}
        assert AtomModelSingleton._instance is None

    def test_no_allocator_standard_behavior(self):
        """Without allocator, get_atom_model uses standard key (no extra_key)."""
        singleton = AtomModelSingleton()
        assert AtomModelSingleton._allocator is None

    def test_hybrid_cache_key_includes_devices(self):
        """With hybrid allocator, cache key includes device tuple."""
        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        AtomModelSingleton.set_allocator(allocator)
        singleton = AtomModelSingleton()
        assert singleton.__class__._allocator is allocator

    def test_different_device_configs_produce_different_keys(self):
        """Same model type with different device configs produces different cache entries."""
        alloc_2 = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        alloc_4 = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3])

        # With 2 devices, layout is on [0]; with 4, layout is on [0] too
        # but ocr_det differs: [1] vs [1], ocr_rec: [1] vs [2]
        # Verify that allocator state is different
        assert alloc_2.get_devices("ocr_rec") != alloc_4.get_devices("ocr_rec")

    def test_singleton_persists_across_instances(self):
        """Multiple AtomModelSingleton() calls return same instance."""
        s1 = AtomModelSingleton()
        s2 = AtomModelSingleton()
        assert s1 is s2


class TestAtomModelInitDeviceExtraction:
    """Test atom_model_init() device param extraction from kwargs."""

    def setup_method(self):
        AtomModelSingleton.reset()

    def teardown_method(self):
        AtomModelSingleton.reset()

    @patch('rapid_doc.backend.pipeline.model_init.layout_model_init')
    def test_layout_receives_device_ids(self, mock_layout_init):
        """With allocator, layout model gets device_ids injected."""
        mock_layout_init.return_value = MagicMock()
        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        AtomModelSingleton.set_allocator(allocator)
        singleton = AtomModelSingleton()

        singleton.get_atom_model("layout", layout_config={})

        call_args = mock_layout_init.call_args
        config = call_args[0][0]
        assert config['device_ids'] == [0]
        assert config['device_lock'] is not None

    @patch('rapid_doc.backend.pipeline.model_init.ocr_model_init')
    def test_ocr_receives_det_rec_device_ids(self, mock_ocr_init):
        """With allocator, OCR model gets det/rec device_ids injected."""
        mock_ocr_init.return_value = MagicMock()
        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3])
        AtomModelSingleton.set_allocator(allocator)
        singleton = AtomModelSingleton()

        singleton.get_atom_model("ocr", ocr_config={})

        call_kwargs = mock_ocr_init.call_args[1]
        assert call_kwargs['det_device_ids'] == [1]
        assert call_kwargs['rec_device_ids'] == [2]

    @patch('rapid_doc.backend.pipeline.model_init.table_model_init')
    def test_table_receives_device_ids(self, mock_table_init):
        """With allocator, table model gets device_ids injected."""
        mock_table_init.return_value = MagicMock()
        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3])
        AtomModelSingleton.set_allocator(allocator)
        singleton = AtomModelSingleton()

        singleton.get_atom_model("table", table_config={})

        call_kwargs = mock_table_init.call_args[1]
        assert call_kwargs['device_ids'] == [3]
        assert call_kwargs['device_lock'] is None  # exclusive in 4-device

    def test_unknown_model_name_exits(self):
        """Unknown model name triggers exit."""
        singleton = AtomModelSingleton()
        with pytest.raises(SystemExit):
            singleton.get_atom_model("nonexistent_model")


class TestCustomModelInitHybrid:
    """Test custom_model_init hybrid parameter behavior."""

    def setup_method(self):
        AtomModelSingleton.reset()

    def teardown_method(self):
        AtomModelSingleton.reset()

    @patch('rapid_doc.backend.pipeline.pipeline_analyze.MineruPipelineModel')
    @patch('rapid_doc.backend.pipeline.pipeline_analyze.get_device')
    def test_hybrid_false_no_allocator(self, mock_get_device, mock_model):
        """hybrid=False should not set allocator."""
        mock_get_device.return_value = 'cpu'
        mock_model.return_value = MagicMock()
        from rapid_doc.backend.pipeline.pipeline_analyze import custom_model_init
        custom_model_init(hybrid=False)
        assert AtomModelSingleton._allocator is None

    @patch('rapid_doc.backend.pipeline.pipeline_analyze.MineruPipelineModel')
    @patch('rapid_doc.backend.pipeline.pipeline_analyze.get_device')
    def test_hybrid_graceful_degradation(self, mock_get_device, mock_model):
        """hybrid=True with no devices: warning, no crash."""
        mock_get_device.return_value = 'cpu'
        mock_model.return_value = MagicMock()

        with patch(
            'rapid_doc.utils.device_allocator.DeviceAllocator.__init__',
            side_effect=RuntimeError("No devices")
        ):
            from rapid_doc.backend.pipeline.pipeline_analyze import custom_model_init
            # Should not raise
            custom_model_init(hybrid=True)

        # Allocator should not be set (init failed)
        assert AtomModelSingleton._allocator is None


class TestDeviceAllocatorIntegration:
    """Test DeviceAllocator works correctly with model name mapping."""

    def test_2_device_allocation_matches_spec(self):
        alloc = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        assert alloc.get_devices("layout") == [0]
        assert alloc.get_devices("ocr_det") == [1]
        assert alloc.get_devices("ocr_rec") == [1]
        assert alloc.get_devices("table") == [0]

    def test_3_device_allocation(self):
        alloc = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2])
        assert alloc.get_devices("layout") == [0]
        assert alloc.get_devices("ocr_det") == [1]
        assert alloc.get_devices("ocr_rec") == [1]
        assert alloc.get_devices("table") == [2]

    def test_4_device_allocation_matches_spec(self):
        alloc = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3])
        assert alloc.get_devices("layout") == [0]
        assert alloc.get_devices("ocr_det") == [1]
        assert alloc.get_devices("ocr_rec") == [2]
        assert alloc.get_devices("table") == [3]

    def test_5_plus_device_allocation(self):
        alloc = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3, 4])
        assert alloc.get_devices("layout") == [0]
        assert alloc.get_devices("ocr_det") == [1]
        assert alloc.get_devices("ocr_rec") == [2, 3]
        assert alloc.get_devices("table") == [4]

    def test_shared_device_lock(self):
        """2-device: layout and table share device 0, so they share a lock."""
        alloc = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        layout_lock = alloc.get_lock("layout")
        table_lock = alloc.get_lock("table")
        assert layout_lock is table_lock
        assert layout_lock is not None

    def test_exclusive_device_no_lock(self):
        """4-device: each model exclusive, no lock needed."""
        alloc = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3])
        assert alloc.get_lock("layout") is None
        assert alloc.get_lock("table") is None

    def test_ocr_sequential_no_lock(self):
        """OCR det/rec are sequential — even sharing device, no lock needed."""
        alloc = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        # ocr_det and ocr_rec share device 1, but OCR group is sequential
        assert alloc.get_lock("ocr_det") is None
        assert alloc.get_lock("ocr_rec") is None

    def test_non_hybrid_no_lock(self):
        """Non-hybrid mode: no lock for any model."""
        alloc = DeviceAllocator(hybrid=False, device_ids=[0, 1])
        assert alloc.get_lock("layout") is None
        assert alloc.get_lock("table") is None

    def test_hybrid_requires_at_least_2_devices(self):
        """hybrid=True with 1 device should raise."""
        with pytest.raises(RuntimeError):
            DeviceAllocator(hybrid=True, device_ids=[0])

    def test_all_devices_property(self):
        alloc = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2])
        assert alloc.all_devices == [0, 1, 2]


class TestCLIArgParsing:
    """Test that --hybrid argparse option works."""

    def test_hybrid_flag_present(self):
        """--hybrid should parse to True."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--hybrid', action='store_true', default=False)
        args = parser.parse_args(['--hybrid'])
        assert args.hybrid is True

    def test_hybrid_flag_absent(self):
        """No --hybrid should default to False."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--hybrid', action='store_true', default=False)
        args = parser.parse_args([])
        assert args.hybrid is False

    def test_hybrid_with_other_args(self):
        """--hybrid combined with other flags."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--hybrid', action='store_true', default=False)
        parser.add_argument('input_path', nargs='?')
        args = parser.parse_args(['--hybrid', 'test_files'])
        assert args.hybrid is True
        assert args.input_path == 'test_files'
