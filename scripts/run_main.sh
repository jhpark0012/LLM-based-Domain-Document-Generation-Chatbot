#!/bin/bash

# 프로젝트 루트 경로
BASE_DIR="/workspace/experiment/Clarifying-Ambiguity-Project"

# 실험별 로그 디렉토리
LOG_DIR="$BASE_DIR/logs/"
mkdir -p "$LOG_DIR"

case_name_list=(
  '건물명도및손해배상'
  '건물인도청구_국가원고'
  '건물철거'
  '공사대금청구의소'
  '공유물분할'
  '대여금_법인'
  '부당등기말소'
  '사해행위취소'
  '손해배상_미성년자원고'
  '손해배상_불법행위'
  '용역대금'
  '임대차보증금반환_선정당사자'
  '임대차보증금청구'
  '저당권말소'
  '채권양도금'
  '채권양도금2'
)

############################################ Arguments 설정
LOAD_FILE="Ambig_Ans"
HF_PATH="$BASE_DIR/data/$LOAD_FILE/"
HF_ACTION="pull"

SAVE_LOG_NUM=$(date +%Y%m%d_%H%M%S) # 딱 실행했을때의 시간
LOG_FILE="$LOG_DIR/${SAVE_LOG_NUM}.log" # 로그 파일 경로

############################################ 함수
FUNC_NAMES=(
  "generate_AA"
  "generate_CQ"
  "hf_dataset_io"
)

# ===================================================================================================================
CYAN="\033[1;36m"; YELLOW="\033[1;33m"; GREEN="\033[0;32m"; RED="\033[0;31m"; NC="\033[0m"
hr(){ echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
step(){ echo -e "🔹 ${CYAN}$1${NC}"; }
ok(){ echo -e "${GREEN}✅ $1${NC}"; }
err(){ echo -e "${RED}❌ $1${NC}"; }

show_menu () {
  hr
  echo -e "Enter steps in the ${YELLOW}desired execution order${NC}"
  echo -e "Example: 2 5 1"
  echo -e "Enter 0 to exit\n"
  for i in "${!FUNC_NAMES[@]}"; do
    idx=$((i+1))
    echo "${idx}) ${FUNC_NAMES[$i]}"
  done
  hr
}

# ===== Input =====
show_menu
read -rp "Your choice : " -a STEPS

# Exit
if [[ ${#STEPS[@]} -eq 1 && "${STEPS[0]}" == "0" ]]; then
  ok "Exit."
  exit 0
fi

UNC_LIST=""
MAX_IDX=${#FUNC_NAMES[@]}

for s in "${STEPS[@]}"; do
    if ! [[ "$s" =~ ^[0-9]+$ ]]; then
        echo "Invalid input: '$s' (numbers only)"
        exit 1
    fi

    # 범위 체크 (1..MAX_IDX)
    if (( s < 1 || s > MAX_IDX )); then
        echo "Invalid step: $s (allowed: 1~$MAX_IDX)"
        exit 1
    fi

    fname="${FUNC_NAMES[$((s-1))]}"
    FUNC_LIST="${FUNC_LIST:+$FUNC_LIST/}$fname" 

done

{
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 python -u "$BASE_DIR/src/main.py" \
  --case_name_list "${case_name_list[*]}" \
  --hf_path "$HF_PATH" \
  --hf_action "$HF_ACTION" \
  --func_list "${FUNC_LIST[@]}" \

} 2>&1 | tee "$LOG_FILE"
