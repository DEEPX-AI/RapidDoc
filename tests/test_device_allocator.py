# tests/test_device_allocator.py
"""DeviceAllocator 할당 로직 단위 테스트."""
import threading
import pytest


class TestDeviceAllocator:
    """DeviceAllocator 기본 할당 로직."""

    def test_non_hybrid_mode_all_devices_shared(self):
        """hybrid=False일 때 모든 모델이 전체 디바이스를 공유한다."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=False, device_ids=[0, 1, 2, 3])
        assert allocator.get_devices("layout") == [0, 1, 2, 3]
        assert allocator.get_devices("ocr_det") == [0, 1, 2, 3]
        assert allocator.get_devices("ocr_rec") == [0, 1, 2, 3]
        assert allocator.get_devices("table") == [0, 1, 2, 3]

    def test_hybrid_2_devices(self):
        """2 디바이스: Layout+Table → Dev0, OCR → Dev1."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        assert allocator.get_devices("layout") == [0]
        assert allocator.get_devices("ocr_det") == [1]
        assert allocator.get_devices("ocr_rec") == [1]
        assert allocator.get_devices("table") == [0]

    def test_hybrid_3_devices(self):
        """3 디바이스: Layout → Dev0, OCR → Dev1, Table → Dev2."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2])
        assert allocator.get_devices("layout") == [0]
        assert allocator.get_devices("ocr_det") == [1]
        assert allocator.get_devices("ocr_rec") == [1]
        assert allocator.get_devices("table") == [2]

    def test_hybrid_4_devices(self):
        """4 디바이스: 완전 분리."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3])
        assert allocator.get_devices("layout") == [0]
        assert allocator.get_devices("ocr_det") == [1]
        assert allocator.get_devices("ocr_rec") == [2]
        assert allocator.get_devices("table") == [3]

    def test_hybrid_5_devices(self):
        """5+ 디바이스: OCR-rec에 다중 디바이스."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3, 4])
        assert allocator.get_devices("layout") == [0]
        assert allocator.get_devices("ocr_det") == [1]
        assert allocator.get_devices("ocr_rec") == [2, 3]
        assert allocator.get_devices("table") == [4]

    def test_hybrid_less_than_2_devices_raises(self):
        """1 디바이스에서 hybrid 모드는 RuntimeError."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        with pytest.raises(RuntimeError, match="2대 이상"):
            DeviceAllocator(hybrid=True, device_ids=[0])

    def test_hybrid_zero_devices_raises(self):
        """0 디바이스에서도 RuntimeError."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        with pytest.raises(RuntimeError, match="2대 이상"):
            DeviceAllocator(hybrid=True, device_ids=[])


class TestDeviceAllocatorLocking:
    """Per-device lock 로직 테스트."""

    def test_exclusive_device_no_lock(self):
        """전용 디바이스인 모델은 lock=None."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        # 4-device: layout=0, ocr_det=1, ocr_rec=2, table=3 → 모두 exclusive
        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3])
        assert allocator.get_lock("layout") is None
        assert allocator.get_lock("ocr_det") is None
        assert allocator.get_lock("ocr_rec") is None
        assert allocator.get_lock("table") is None

    def test_shared_device_returns_lock(self):
        """공유 디바이스인 모델은 동일한 Lock 객체를 반환한다."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        # 2-device: layout=0, table=0 → 공유
        layout_lock = allocator.get_lock("layout")
        table_lock = allocator.get_lock("table")
        assert layout_lock is not None
        assert table_lock is not None
        assert layout_lock is table_lock  # 같은 Lock 객체

    def test_ocr_det_rec_shared_device_same_lock(self):
        """OCR-det과 OCR-rec이 같은 디바이스면 같은 lock."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2])
        # 3-device: ocr_det=1, ocr_rec=1 → 공유
        det_lock = allocator.get_lock("ocr_det")
        rec_lock = allocator.get_lock("ocr_rec")
        assert det_lock is rec_lock

    def test_non_hybrid_no_locks(self):
        """non-hybrid 모드에서는 모든 lock이 None (기존 동작 유지)."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=False, device_ids=[0, 1, 2, 3])
        assert allocator.get_lock("layout") is None
        assert allocator.get_lock("ocr_det") is None
        assert allocator.get_lock("table") is None

    def test_is_exclusive(self):
        """is_exclusive 판별 정확성."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        # 2-device: layout(0) + table(0) 공유
        alloc2 = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        assert alloc2.is_exclusive("layout") is False
        assert alloc2.is_exclusive("table") is False
        assert alloc2.is_exclusive("ocr_det") is True
        assert alloc2.is_exclusive("ocr_rec") is True

        # 4-device: 모두 exclusive
        alloc4 = DeviceAllocator(hybrid=True, device_ids=[0, 1, 2, 3])
        assert alloc4.is_exclusive("layout") is True
        assert alloc4.is_exclusive("ocr_det") is True
        assert alloc4.is_exclusive("ocr_rec") is True
        assert alloc4.is_exclusive("table") is True


class TestDeviceAllocatorDetection:
    """디바이스 감지 로직 테스트."""

    def test_explicit_device_ids_override_detection(self):
        """device_ids를 명시적으로 전달하면 감지 건너뜀."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        allocator = DeviceAllocator(hybrid=True, device_ids=[0, 1])
        assert allocator.all_devices == [0, 1]

    def test_env_var_detection(self, monkeypatch):
        """DXNN_DEVICES 환경변수로 디바이스 감지."""
        from rapid_doc.utils.device_allocator import DeviceAllocator

        monkeypatch.setenv("DXNN_DEVICES", "0,1,2")
        allocator = DeviceAllocator(hybrid=True, device_ids=None)
        assert allocator.all_devices == [0, 1, 2]
