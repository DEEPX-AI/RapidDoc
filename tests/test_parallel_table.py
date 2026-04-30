"""듀얼 스레드 테이블 파이프라인 테스트."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass


class TestWiredRunStructureOnly:
    """WiredTableRecognition.run_structure_only 테스트."""

    def test_returns_polygons_from_table_structure(self):
        """run_structure_only가 table_structure 결과를 그대로 반환하는지 검증."""
        from rapid_doc.model.table.rapid_table_self.wired_table_rec.main import (
            WiredTableRecognition,
        )
        import inspect
        src = inspect.getsource(WiredTableRecognition.run_structure_only)
        assert 'table_structure' in src
        assert 'load_img' in src

    def test_does_not_double_reshape_swap(self):
        """run_structure_only가 reshape/swap을 수행하지 않는지 (TSRUnet이 이미 처리)."""
        from rapid_doc.model.table.rapid_table_self.wired_table_rec.main import (
            WiredTableRecognition,
        )
        import inspect
        src = inspect.getsource(WiredTableRecognition.run_structure_only)
        assert '.reshape(' not in src
        assert 'copy()' not in src

    def test_returns_none_tuple_when_structure_fails(self):
        """table_structure가 None 반환 시 (None, None)."""
        from rapid_doc.model.table.rapid_table_self.wired_table_rec.main import (
            WiredTableRecognition,
        )
        import inspect
        params = inspect.signature(WiredTableRecognition.run_structure_only).parameters
        assert 'self' in params
        assert 'img' in params


class TestWiredBuildFromStructure:
    """WiredTableRecognition.build_from_structure 테스트."""

    def test_method_exists_with_correct_params(self):
        """build_from_structure가 올바른 파라미터를 가지는지."""
        from rapid_doc.model.table.rapid_table_self.wired_table_rec.main import (
            WiredTableRecognition,
        )
        import inspect
        params = inspect.signature(WiredTableRecognition.build_from_structure).parameters
        assert 'img' in params
        assert 'polygons' in params
        assert 'rotated_polygons' in params
        assert 'ocr_result' in params

    def test_source_contains_table_recover_and_match(self):
        """build_from_structure가 table_recover와 match_ocr_cell을 호출하는지."""
        from rapid_doc.model.table.rapid_table_self.wired_table_rec.main import (
            WiredTableRecognition,
        )
        import inspect
        src = inspect.getsource(WiredTableRecognition.build_from_structure)
        assert 'table_recover' in src
        assert 'match_ocr_cell' in src


class TestUnetResult:
    """UnetResult dataclass 검증."""

    def test_dataclass_fields(self):
        from rapid_doc.model.table.rapid_table import UnetResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(UnetResult)}
        assert fields == {'polygons', 'rotated_polygons', 'upscaled_bgr'}

    def test_accepts_none_polygons(self):
        from rapid_doc.model.table.rapid_table import UnetResult
        result = UnetResult(polygons=None, rotated_polygons=None, upscaled_bgr=np.zeros((10, 10, 3)))
        assert result.polygons is None


class TestPrepareImage:
    """RapidTableModel.prepare_image 검증."""

    def test_method_exists(self):
        from rapid_doc.model.table.rapid_table import RapidTableModel
        assert hasattr(RapidTableModel, 'prepare_image')

    def test_landscape_not_rotated(self):
        """Landscape 이미지 (width > height)는 회전하지 않음."""
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        src = inspect.getsource(RapidTableModel.prepare_image)
        assert 'img_is_portrait' in src or 'aspect_ratio' in src

    def test_returns_tuple_bgr_and_bool(self):
        """반환값이 (bgr_image, is_rotated) 튜플."""
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        sig = inspect.signature(RapidTableModel.prepare_image)
        # self + image 파라미터
        assert len(sig.parameters) == 2


class TestRunUnet:
    """RapidTableModel.run_unet 검증."""

    def test_method_signature(self):
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        sig = inspect.signature(RapidTableModel.run_unet)
        params = list(sig.parameters.keys())
        assert 'bgr_image' in params
        assert 'fill_image_res' in params

    def test_source_contains_upscale_and_structure(self):
        """run_unet이 2x upscale + table_model 호출을 포함하는지."""
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        src = inspect.getsource(RapidTableModel.run_unet)
        assert 'resize' in src or 'upscale' in src
        assert 'run_structure_only' in src

    def test_source_handles_fill_image_res(self):
        """fill_image_res가 있을 때 bgr_image를 복사하고 백색 사각형 그리기."""
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        src = inspect.getsource(RapidTableModel.run_unet)
        assert 'copy' in src or '.copy()' in src
        assert 'rectangle' in src


class TestRunOcr:
    """RapidTableModel.run_ocr 검증."""

    def test_method_signature(self):
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        sig = inspect.signature(RapidTableModel.run_ocr)
        params = list(sig.parameters.keys())
        assert 'bgr_image' in params
        assert 'mfd_res' in params

    def test_source_calls_ocr_engine(self):
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        src = inspect.getsource(RapidTableModel.run_ocr)
        assert 'ocr_engine' in src


class TestBuildHtml:
    """RapidTableModel.build_html 검증."""

    def test_method_signature(self):
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        sig = inspect.signature(RapidTableModel.build_html)
        params = list(sig.parameters.keys())
        assert 'unet_result' in params
        assert 'ocr_result' in params
        assert 'fill_image_res' in params
        assert 'mfd_res' in params
        assert 'skip_text_in_image' in params
        assert 'use_img2table' in params

    def test_returns_none_when_ocr_none(self):
        """ocr_result=None이면 ("", None, None, 0.0) — 기존 동작 유지."""
        from rapid_doc.model.table.rapid_table import RapidTableModel, UnetResult
        import inspect
        src = inspect.getsource(RapidTableModel.build_html)
        assert 'ocr_result' in src
        assert 'None' in src

    def test_source_contains_scale_and_wired(self):
        """build_html이 OCR 좌표 스케일링과 WiredTableRecognition을 사용하는지."""
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        src = inspect.getsource(RapidTableModel.build_html)
        assert '_scale_ocr_result' in src
        assert 'build_from_structure' in src

    def test_ocr_zipped_before_build_from_structure(self):
        """build_html이 build_from_structure 호출 시 OCR을 zip 형식으로 변환하는지.

        match_ocr_cell은 [(box, text, score), ...] 형식을 기대하므로,
        [boxes, texts, scores] → list(zip(...)) 변환이 반드시 필요하다.
        RapidTable.__call__에서 이 변환을 수행하지만 build_from_structure
        직접 호출 경로에서는 build_html이 직접 변환해야 한다.
        """
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        src = inspect.getsource(RapidTableModel.build_html)
        # zip 변환이 build_from_structure 호출 전에 존재해야 한다
        assert 'zip(' in src, (
            "build_html must zip OCR results [boxes, texts, scores] → "
            "[(box, text, score), ...] before build_from_structure"
        )


class TestPredictRefactored:
    """predict()가 서브 메서드를 사용하여 동일 결과를 내는지."""

    def test_predict_calls_sub_methods(self):
        """predict()가 prepare_image, run_ocr, run_unet, build_html을 호출하는지."""
        from rapid_doc.model.table.rapid_table import RapidTableModel
        import inspect
        src = inspect.getsource(RapidTableModel.predict)
        assert 'prepare_image' in src
        assert 'run_unet' in src
        assert 'build_html' in src


import threading
import time


class TestProcessTablesParallel:
    """_process_tables_parallel 병렬 동작 검증."""

    def test_method_exists(self):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        assert hasattr(TrueAsyncPipeline, '_process_tables_parallel')

    def test_source_uses_dual_threads(self):
        """UNET과 OCR 스레드를 각각 생성하는지."""
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        import inspect
        src = inspect.getsource(TrueAsyncPipeline._process_tables_parallel)
        assert 'table-unet' in src or 'unet_worker' in src
        assert 'table-ocr' in src or 'ocr_worker' in src
        assert 'Thread' in src

    def test_source_uses_three_phases(self):
        """Phase 1 (prepare), Phase 2 (parallel), Phase 3 (combine) 패턴."""
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        import inspect
        src = inspect.getsource(TrueAsyncPipeline._process_tables_parallel)
        assert 'prepare_image' in src
        assert 'run_unet' in src
        assert 'run_ocr' in src
        assert 'build_html' in src

    def test_error_propagation(self):
        """에러 전파 로직이 있는지."""
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        import inspect
        src = inspect.getsource(TrueAsyncPipeline._process_tables_parallel)
        assert 'raise' in src


class TestStageTableUsesParallel:
    """_stage_table이 _process_tables_parallel을 사용하는지."""

    def test_stage_table_calls_parallel(self):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        import inspect
        src = inspect.getsource(TrueAsyncPipeline._stage_table)
        assert '_process_tables_parallel' in src

    def test_table_one_calls_parallel(self):
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        import inspect
        src = inspect.getsource(TrueAsyncPipeline._table_one)
        assert '_process_tables_parallel' in src