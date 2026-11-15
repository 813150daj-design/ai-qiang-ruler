#!/bin/bash
# GPT writes the training texts: each text a fresh `codex exec` in its own empty folder.
# s1 gpt-6.1-sol, s2 gpt-6-chat-thinking, s3 gpt-6.1-terra, all at the owner's codex defaults (effort xhigh,
# priority tier). Existing files are skipped, so it can resume. Before each start it waits while 3 or more
# `codex exec` lanes run on this machine (owner's limit, counting all sessions).
#   bash gen_gpt.sh                                   prompts.json, samples s1-s3 for every prompt
#   bash gen_gpt.sh prompts_v2.json rotate 0 2        one text per prompt, sample (i mod 3)+1; lane 0 of 2
# -> ../data/ai/gpt/<pid>_s<k>.txt and meta.jsonl; logs and temp folders go to the work folder
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../data/ai/gpt"
WORK="$HERE/../../AI腔尺子_工作/gen_logs"
TMP="$WORK/gpt_tmp"
LOGS="$WORK/gpt_logs"
mkdir -p "$OUT" "$TMP" "$LOGS"
PF="${1:-prompts.json}"; ROT="${2:-}"; LN="${3:-0}"; LC="${4:-1}"
lanes() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='codex.exe'\" | Where-Object { \$_.CommandLine -match ' exec ' }).Count" 2>/dev/null | tr -d '\r '
}
wait_lane() {
  while true; do
    n=$(lanes); n=${n:-0}
    if [ "$n" -lt 3 ]; then return; fi
    echo "$(date +%T) $n codex lanes running, waiting"
    sleep 60
  done
}
model_for() { case $1 in 1) echo gpt-6.1-sol;; 2) echo gpt-6-chat-thinking;; 3) echo gpt-6.1-terra;; esac; }
PJ="$(cygpath -w "$HERE/$PF")"
PYTHONIOENCODING=utf-8 python -c "
import io, json, sys
ps = json.load(io.open(sys.argv[1], encoding='utf-8'))['prompts']
rot = sys.argv[2] == 'rotate'
ln, lc = int(sys.argv[3]), int(sys.argv[4])
for k in (1, 2, 3):
    for i, p in enumerate(ps):
        if i % lc != ln:
            continue
        if rot and i % 3 + 1 != k:
            continue
        print(str(k) + chr(9) + p['id'] + chr(9) + p['text'])
" "$PJ" "$ROT" "$LN" "$LC" > "$TMP/jobs_$LN.tsv"
while IFS=$'\t' read -r k pid text; do
  f="$OUT/${pid}_s${k}.txt"
  if [ -s "$f" ]; then continue; fi
  m=$(model_for "$k")
  for effort in default medium; do
    wait_lane
    d="$TMP/${pid}_s${k}"; rm -rf "$d"; mkdir -p "$d"
    extra=(); if [ $effort != default ]; then extra=(-c model_reasoning_effort="$effort"); fi
    t0=$(date +%s)
    timeout 900 codex exec --skip-git-repo-check -m "$m" "${extra[@]}" -C "$d" -s read-only -c approval_policy="never" \
      -o "$f" "$text" > "$LOGS/${pid}_s${k}_${effort}.log" 2>&1 < /dev/null
    code=$?
    if [ -s "$f" ]; then
      echo "{\"model\": \"GPT\", \"prompt\": \"$pid\", \"sample\": $k, \"served_model\": \"$m\", \"effort\": \"$effort\", \"seconds\": $(( $(date +%s) - t0 )), \"time\": \"$(date '+%F %T')\"}" >> "$OUT/meta.jsonl"
      echo "$(date +%T) $pid s$k $m ($effort) ok $(wc -c < "$f") bytes"
      break
    fi
    echo "$(date +%T) $pid s$k $m ($effort) exit $code, no text"
  done
done < "$TMP/jobs_$LN.tsv"
echo "lane done"
