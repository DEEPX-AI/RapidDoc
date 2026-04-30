"""Stage 병렬화 테스트: Formula(CPU) ∥ OCR-det(NPU)."""
import threading
import time
import pytest
from unittest.mock import MagicMock, patch


class TestRunParallelFormulaOcrDet:
    """_run_parallel_formula_ocr_det 메서드 테스트."""

    def _make_pipeline(self, formula_enable=True, formula_rec_enable=True):
        """테스트용 파이프라인 인스턴스를 mock으로 생성."""
        from rapid_doc.backend.pipeline.async_pipeline import TrueAsyncPipeline
        with patch.object(TrueAsyncPipeline, '__init__', lambda self: None):
            pipe = TrueAsyncPipeline.__new__(TrueAsyncPipeline)
        pipe.formula_enable = formula_enable
        pipe.formula_rec_enable = formula_rec_enable
        pipe._stage_formula = MagicMock()
        pipe._stage_ocr_det = MagicMock()
        return pipe

    def test_both_stages_called(self):
        """Formula와 OCR-det 모두 호출된다."""
        pipe = self._make_pipeline()
        contexts = [MagicMock()]
        pipe._run_parallel_formula_ocr_det(contexts)
        pipe._stage_formula.assert_called_once_with(contexts)
        pipe._stage_ocr_det.assert_called_once_with(contexts)

    def test_formula_disabled_skips_formula(self):
        """formula_enable=False이면 _stage_formula를 호출하지 않는다."""
        pipe = self._make_pipeline(formula_enable=False)
        contexts = [MagicMock()]
        pipe._run_parallel_formula_ocr_det(contexts)
        pipe._stage_formula.assert_not_called()
        pipe._stage_ocr_det.assert_called_once_with(contexts)

    def test_formula_rec_disabled_skips_formula(self):
        """formula_rec_enable=False이면 _stage_formula를 호출하지 않는다."""
        pipe = self._make_pipeline(formula_rec_enable=False)
        contexts = [MagicMock()]
        pipe._run_parallel_formula_ocr_det(contexts)
        pipe._stage_formula.assert_not_called()
        pipe._stage_ocr_det.assert_called_once_with(contexts)

    def test_runs_in_parallel(self):
        """두 stage가 동시에 실행된다 (wall time < 합계)."""
        pipe = self._make_pipeline()

        def slow_formula(contexts):
            time.sleep(0.1)
        def slow_ocr_det(contexts):
            time.sleep(0.1)

        pipe._stage_formula = MagicMock(side_effect=slow_formula)
        pipe._stage_ocr_det = MagicMock(side_effect=slow_ocr_det)

        contexts = [MagicMock()]
        t0 = time.perf_counter()
        pipe._run_parallel_formula_ocr_det(contexts)
        elapsed = time.perf_counter() - t0

        # 병렬이면 ~0.1s, 순차면 ~0.2s
        assert elapsed < 0.15, f"Expected parallel execution but took {elapsed:.3f}s"

    def test_cpu_error_propagated(self):
        """Formula 스레드 예외가 메인 스레드로 전파된다."""
        pipe = self._make_pipeline()
        pipe._stage_formula = MagicMock(side_effect=RuntimeError("formula crash"))

        contexts = [MagicMock()]
        with pytest.raises(RuntimeError, match="formula crash"):
            pipe._run_parallel_formula_ocr_det(contexts)

    def test_npu_error_propagated(self):
        """OCR-det 스레드 예외가 메인 스레드로 전파된다."""
        pipe = self._make_pipeline()
        pipe._stage_ocr_det = MagicMock(side_effect=RuntimeError("npu crash"))

        contexts = [MagicMock()]
        with pytest.raises(RuntimeError, match="npu crash"):
            pipe._run_parallel_formula_ocr_det(contexts)

    def test_both_errors_first_cpu_raised(self):
        """양쪽 모두 실패 시 CPU(formula) 에러가 먼저 raise된다."""
        pipe = self._make_pipeline()
        pipe._stage_formula = MagicMock(side_effect=RuntimeError("cpu fail"))
        pipe._stage_ocr_det = MagicMock(side_effect=RuntimeError("npu fail"))

        contexts = [MagicMock()]
        with pytest.raises(RuntimeError, match="cpu fail"):
            pipe._run_parallel_formula_ocr_det(contexts)

    def test_thread_names(self):
        """스레드 이름이 올바르게 설정된다 (디버깅용)."""
        pipe = self._make_pipeline()
        thread_names = []

        def capture_formula(contexts):
            thread_names.append(threading.current_thread().name)
        def capture_ocr(contexts):
            thread_names.append(threading.current_thread().name)

        pipe._stage_formula = MagicMock(side_effect=capture_formula)
        pipe._stage_ocr_det = MagicMock(side_effect=capture_ocr)

        contexts = [MagicMock()]
        pipe._run_parallel_formula_ocr_det(contexts)

        assert "stage-formula-cpu" in thread_names
        assert "stage-ocr-det-npu" in thread_names
