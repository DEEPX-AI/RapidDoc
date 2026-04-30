#!/usr/bin/env bash
# =============================================================================
# run_tests.sh — TrueAsyncPipeline 전체 테스트 실행 스크립트
#
# 사용법:
#   bash tests/run_tests.sh [옵션]
#
# 옵션:
#   --unit-only          L1+L2 단위/Mock 테스트만 실행 (모델 불필요, 기본 동작)
#   --integration        L3 통합 테스트까지 포함 (모델 + PDF + DX 환경 필요)
#   --pdf <PATH>         통합 테스트에 사용할 PDF 경로 (기본: test_files 첫 번째 파일)
#   --venv <PATH>        사용할 virtualenv 경로 (기본: ./venv)
#   --loop <N>           N회 반복 실행 (aging 테스트, 기본: 1)
#   --no-color           컬러 출력 비활성화
#   --verbose            pytest -v 출력
#   -h, --help           도움말 출력
#
# 예시:
#   # L1+L2 빠른 검증 (CI 용도)
#   bash tests/run_tests.sh
#
#   # 전체 통합 테스트
#   bash tests/run_tests.sh --integration
#
#   # PDF 지정 + 풀 통합
#   bash tests/run_tests.sh --integration --pdf test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf
#
#   # aging 테스트 — 통합 포함 10회 반복
#   bash tests/run_tests.sh --integration --loop 10
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 색상 정의
# ─────────────────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_RESET="\033[0m"
    C_RED="\033[0;31m"
    C_GREEN="\033[0;32m"
    C_YELLOW="\033[1;33m"
    C_BLUE="\033[0;34m"
    C_CYAN="\033[0;36m"
    C_BOLD="\033[1m"
else
    C_RESET="" C_RED="" C_GREEN="" C_YELLOW="" C_BLUE="" C_CYAN="" C_BOLD=""
fi

# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
info()    { echo -e "${C_BLUE}[INFO]${C_RESET}  $*"; }
ok()      { echo -e "${C_GREEN}[ OK ]${C_RESET}  $*"; }
warn()    { echo -e "${C_YELLOW}[WARN]${C_RESET}  $*"; }
error()   { echo -e "${C_RED}[ERR ]${C_RESET}  $*" >&2; }
header()  { echo -e "\n${C_BOLD}${C_CYAN}$*${C_RESET}"; }
divider() { echo -e "${C_CYAN}$(printf '─%.0s' {1..70})${C_RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
# 인수 파싱
# ─────────────────────────────────────────────────────────────────────────────
RUN_INTEGRATION=0
PDF_PATH=""
VENV_PATH="venv"
VERBOSE=""
LOOP_COUNT=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --integration)   RUN_INTEGRATION=1; shift ;;
        --unit-only)     RUN_INTEGRATION=0; shift ;;
        --pdf)           PDF_PATH="$2"; shift 2 ;;
        --venv)          VENV_PATH="$2"; shift 2 ;;
        --loop)          LOOP_COUNT="$2"; shift 2 ;;
        --no-color)      C_RESET="" C_RED="" C_GREEN="" C_YELLOW=""
                         C_BLUE="" C_CYAN="" C_BOLD=""; shift ;;
        --verbose|-v)    VERBOSE="-v"; shift ;;
        -h|--help)
            sed -n '/^# =\{10\}/,/^# =\{10\}/p' "$0" | sed 's/^# \{0,1\}//' | grep -v '^=\{10\}'
            exit 0 ;;
        *) error "알 수 없는 옵션: $1"; exit 1 ;;
    esac
done

# loop 횟수 유효성 검사
if ! [[ "${LOOP_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    error "--loop 값은 1 이상의 정수여야 합니다: '${LOOP_COUNT}'"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 프로젝트 루트로 이동
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# 타임스탬프 / 로그 파일
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/tests/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/test_run_${TIMESTAMP}.log"

header "TrueAsyncPipeline 테스트 스위트"
divider
info "프로젝트 루트 : ${PROJECT_ROOT}"
info "로그 파일    : ${LOG_FILE}"
info "실행 시각    : $(date '+%Y-%m-%d %H:%M:%S')"
[[ "${RUN_INTEGRATION}" -eq 1 ]] && info "모드          : L1 + L2 + L3 (통합)" \
                                 || info "모드          : L1 + L2 (단위/Mock, 모델 불필요)"
[[ "${LOOP_COUNT}" -gt 1 ]] && info "반복 횟수     : ${LOOP_COUNT}회 (aging 모드)"
divider

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 가상환경 활성화
# ─────────────────────────────────────────────────────────────────────────────
header "[1/5] 가상환경 확인"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
    ok "가상환경 활성화: ${VENV_PATH}"
else
    warn "가상환경 '${VENV_PATH}/bin/activate' 없음 — 시스템 Python 사용"
fi

PYTHON="${PYTHON:-python}"
PIP="${PIP:-pip}"
PYTHON_VERSION="$("${PYTHON}" --version 2>&1)"
ok "Python: ${PYTHON_VERSION}  ($(which "${PYTHON}"))"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 환경 변수 설정 (통합 테스트 시에만 필수)
# ─────────────────────────────────────────────────────────────────────────────
header "[2/5] 환경 변수 설정"

REQUIRED_ENV_VARS=(
    "CUSTOM_INTER_OP_THREADS_COUNT:1"
    "CUSTOM_INTRA_OP_THREADS_COUNT:2"
    "DXRT_DYNAMIC_CPU_THREAD:1"
    "DXRT_TASK_MAX_LOAD:3"
    "NFH_INPUT_WORKER_THREADS:2"
    "NFH_OUTPUT_WORKER_THREADS:4"
)

ENV_OK=1
for entry in "${REQUIRED_ENV_VARS[@]}"; do
    var="${entry%%:*}"
    expected="${entry##*:}"
    actual="${!var:-}"
    if [[ "${actual}" == "${expected}" ]]; then
        ok "${var}=${actual}"
    elif [[ "${RUN_INTEGRATION}" -eq 1 ]]; then
        error "${var}가 설정되지 않았거나 값이 다릅니다. (현재='${actual}', 기대='${expected}')"
        ENV_OK=0
    else
        warn "${var} 미설정 (단위 테스트에서는 무시)"
    fi
done

if [[ "${RUN_INTEGRATION}" -eq 1 && "${ENV_OK}" -eq 0 ]]; then
    error "환경 변수가 올바르지 않습니다."
    error "다음 명령을 먼저 실행하세요:"
    error "  source ./deepx_scripts/set_env.sh 1 2 1 3 2 4"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: PDF 파일 결정 (통합 테스트 전용)
# ─────────────────────────────────────────────────────────────────────────────
header "[3/5] PDF 파일 확인"

if [[ "${RUN_INTEGRATION}" -eq 1 ]]; then
    if [[ -n "${PDF_PATH}" ]]; then
        if [[ ! -f "${PDF_PATH}" ]]; then
            error "지정한 PDF 파일을 찾을 수 없습니다: ${PDF_PATH}"
            exit 1
        fi
        ok "PDF (지정): ${PDF_PATH}"
    else
        # test_files/ 에서 첫 번째 파일 자동 선택
        PDF_PATH="$(ls "${PROJECT_ROOT}/test_files/"*.pdf 2>/dev/null | sort | head -1 || true)"
        if [[ -z "${PDF_PATH}" ]]; then
            error "test_files/ 에 PDF 파일이 없습니다."
            error "  --pdf <PATH> 옵션으로 직접 지정하거나"
            error "  test_files/ 에 PDF 파일을 추가하세요."
            exit 1
        fi
        ok "PDF (자동): ${PDF_PATH}"
    fi
    export RD_TEST_PDF="${PDF_PATH}"
    info "RD_TEST_PDF=${RD_TEST_PDF}"
else
    export RD_TEST_SKIP_INTEGRATION=1
    info "통합 테스트 스킵 (RD_TEST_SKIP_INTEGRATION=1)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: pytest 의존성 확인
# ─────────────────────────────────────────────────────────────────────────────
header "[4/5] pytest 의존성 확인"

if ! "${PYTHON}" -m pytest --version &>/dev/null; then
    warn "pytest 없음 — 설치 시도 중..."
    "${PIP}" install pytest pytest-mock --quiet
fi
PYTEST_VERSION="$("${PYTHON}" -m pytest --version 2>&1 | head -1)"
ok "${PYTEST_VERSION}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 테스트 실행
# ─────────────────────────────────────────────────────────────────────────────
header "[5/5] 테스트 실행"
divider

# pytest 마커 설정
if [[ "${RUN_INTEGRATION}" -eq 1 ]]; then
    PYTEST_MARKER=""        # 마커 필터 없음 → 전체
    SUITE_LABEL="전체 (L1+L2+L3)"
else
    PYTEST_MARKER="-m 'not integration'"
    SUITE_LABEL="단위+Mock (L1+L2)"
fi

PYTEST_CMD=(
    "${PYTHON}" -m pytest
    "tests/test_async_pipeline.py"
    ${VERBOSE:+-v}
    --tb=short
    --no-header
    -p no:cacheprovider
)
if [[ "${RUN_INTEGRATION}" -ne 1 ]]; then
    PYTEST_CMD+=(-m "not integration")
fi

info "실행 명령: ${PYTEST_CMD[*]}"
info "테스트 범위: ${SUITE_LABEL}"
[[ "${LOOP_COUNT}" -gt 1 ]] && info "aging 반복: ${LOOP_COUNT}회"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Loop (aging) 실행
# ─────────────────────────────────────────────────────────────────────────────
AGING_TOTAL_PASSED=0
AGING_TOTAL_FAILED=0
AGING_TOTAL_SKIPPED=0
AGING_FAIL_ITERS=""
AGING_ANY_FAILED=0

AGING_START="$(date +%s%3N)"

for (( ITER=1; ITER<=LOOP_COUNT; ITER++ )); do

    if [[ "${LOOP_COUNT}" -gt 1 ]]; then
        echo -e "${C_BOLD}${C_CYAN}━━━ Iteration ${ITER} / ${LOOP_COUNT} ━━━${C_RESET}"
    fi

    # 이터레이션별 로그 파일
    if [[ "${LOOP_COUNT}" -gt 1 ]]; then
        ITER_LOG="${LOG_DIR}/test_run_${TIMESTAMP}_iter$(printf '%04d' ${ITER}).log"
    else
        ITER_LOG="${LOG_FILE}"
    fi

    T_ITER_START="$(date +%s%3N)"

    set +e
    "${PYTEST_CMD[@]}" 2>&1 | tee "${ITER_LOG}"
    ITER_EXIT="${PIPESTATUS[0]}"
    set -e

    T_ITER_END="$(date +%s%3N)"
    ITER_SEC="$(echo "scale=2; $(( T_ITER_END - T_ITER_START ))/1000" | bc)"

    # 이터레이션 통계 파싱
    ITER_PASSED="$(grep -oP '\d+ passed' "${ITER_LOG}" | tail -1 | grep -oP '\d+' || echo 0)"
    ITER_FAILED="$(grep -oP '\d+ failed' "${ITER_LOG}" | tail -1 | grep -oP '\d+' || echo 0)"
    ITER_SKIPPED="$(grep -oP '\d+ skipped' "${ITER_LOG}" | tail -1 | grep -oP '\d+' || echo 0)"

    AGING_TOTAL_PASSED=$(( AGING_TOTAL_PASSED + ITER_PASSED ))
    AGING_TOTAL_FAILED=$(( AGING_TOTAL_FAILED + ITER_FAILED ))
    AGING_TOTAL_SKIPPED=$(( AGING_TOTAL_SKIPPED + ITER_SKIPPED ))

    if [[ "${ITER_EXIT}" -ne 0 ]]; then
        AGING_ANY_FAILED=1
        AGING_FAIL_ITERS="${AGING_FAIL_ITERS} ${ITER}"
        echo -e "  ${C_RED}✗ iter ${ITER}: ${ITER_FAILED} failed (${ITER_SEC}s)${C_RESET}"
    else
        if [[ "${LOOP_COUNT}" -gt 1 ]]; then
            echo -e "  ${C_GREEN}✓ iter ${ITER}: ${ITER_PASSED} passed, ${ITER_SKIPPED} skipped (${ITER_SEC}s)${C_RESET}"
        fi
    fi

done

AGING_END="$(date +%s%3N)"
ELAPSED_MS="$(( AGING_END - AGING_START ))"
ELAPSED_SEC="$(echo "scale=2; ${ELAPSED_MS}/1000" | bc)"

# 전체 결과 log 심볼릭 링크 (loop>1 시 마지막 iter 기록)
[[ "${LOOP_COUNT}" -gt 1 ]] && cp "${LOG_DIR}/test_run_${TIMESTAMP}_iter$(printf '%04d' ${LOOP_COUNT}).log" \
    "${LOG_FILE}" 2>/dev/null || true

EXIT_CODE="${AGING_ANY_FAILED}"

# ─────────────────────────────────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────────────────────────────────
divider
header "테스트 결과 요약"
divider
info "소요 시간  : ${ELAPSED_SEC}s"
info "로그 파일  : ${LOG_FILE}"

# 결과 통계 (단일 실행은 로그 파싱, loop는 누산 값 사용)
if [[ "${LOOP_COUNT}" -gt 1 ]]; then
    PASSED="${AGING_TOTAL_PASSED}"
    FAILED="${AGING_TOTAL_FAILED}"
    SKIPPED="${AGING_TOTAL_SKIPPED}"
else
    PASSED="$(grep -oP '\d+ passed' "${LOG_FILE}" | tail -1 | grep -oP '\d+' || echo 0)"
    FAILED="$(grep -oP '\d+ failed' "${LOG_FILE}" | tail -1 | grep -oP '\d+' || echo 0)"
    SKIPPED="$(grep -oP '\d+ skipped' "${LOG_FILE}" | tail -1 | grep -oP '\d+' || echo 0)"
fi
ERRORS="$(grep -oP '\d+ error' "${LOG_FILE}" | tail -1 | grep -oP '\d+' || echo 0)"

echo ""
echo -e "  ${C_GREEN}통과${C_RESET}: ${PASSED}"
echo -e "  ${C_RED}실패${C_RESET}: ${FAILED}"
echo -e "  ${C_YELLOW}건너뜀${C_RESET}: ${SKIPPED}"
[[ "${ERRORS}" -gt 0 ]] && echo -e "  ${C_RED}에러${C_RESET}: ${ERRORS}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Aging 요약 (loop > 1 인 경우)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${LOOP_COUNT}" -gt 1 ]]; then
    AGING_TOTAL_RUNS=$(( LOOP_COUNT ))
    AGING_PASS_RUNS=$(( AGING_TOTAL_RUNS - ${#AGING_FAIL_ITERS} ))
    # 실패 이터레이션 수 = 공백으로 구분된 단어 수
    AGING_FAIL_COUNT=$(echo "${AGING_FAIL_ITERS}" | wc -w)
    AGING_PASS_COUNT=$(( AGING_TOTAL_RUNS - AGING_FAIL_COUNT ))
    SUCCESS_RATE="$(echo "scale=1; ${AGING_PASS_COUNT}*100/${AGING_TOTAL_RUNS}" | bc)"

    echo -e "${C_BOLD}${C_CYAN}━━━ Aging 결과 요약 (${LOOP_COUNT}회 반복) ━━━${C_RESET}"
    echo -e "  총 실행  : ${AGING_TOTAL_RUNS}회"
    echo -e "  성공     : ${C_GREEN}${AGING_PASS_COUNT}회${C_RESET}"
    echo -e "  실패     : ${C_RED}${AGING_FAIL_COUNT}회${C_RESET}"
    echo -e "  성공률   : ${SUCCESS_RATE}%"
    echo -e "  누산 통과: ${AGING_TOTAL_PASSED}"
    echo -e "  누산 실패: ${AGING_TOTAL_FAILED}"
    echo -e "  누산 스킵: ${AGING_TOTAL_SKIPPED}"
    if [[ -n "${AGING_FAIL_ITERS// /}" ]]; then
        echo -e "  실패 iter:${C_RED}${AGING_FAIL_ITERS}${C_RESET}"
    fi
    echo -e "  이터레이션 로그: ${LOG_DIR}/test_run_${TIMESTAMP}_iter*.log"
    echo ""
fi

if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo -e "${C_GREEN}${C_BOLD}✅  모든 테스트 통과 (exit 0)${C_RESET}"
else
    echo -e "${C_RED}${C_BOLD}❌  테스트 실패 또는 에러 (exit ${EXIT_CODE})${C_RESET}"
    echo ""
    echo -e "${C_YELLOW}실패 상세 확인:${C_RESET}"
    echo -e "  grep -A 20 'FAILED\\|ERROR' ${LOG_FILE}"
fi

divider

# ─────────────────────────────────────────────────────────────────────────────
# 통합 테스트 미실행 시 안내
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_INTEGRATION}" -eq 0 && "${SKIPPED}" -gt 0 ]]; then
    echo ""
    info "L3 통합 테스트 ${SKIPPED}개가 스킵되었습니다."
    info "전체 테스트를 실행하려면:"
    echo ""
    echo -e "  ${C_CYAN}source ./deepx_scripts/set_env.sh 1 2 1 3 2 4${C_RESET}"
    echo -e "  ${C_CYAN}bash tests/run_tests.sh --integration [--pdf <PDF경로>]${C_RESET}"
    echo ""
fi

exit "${EXIT_CODE}"
