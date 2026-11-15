#!/bin/bash
# How the human texts were downloaded (11-14 night). They go to a folder OUTSIDE the repo and are never committed;
# only statistics derived from them (the model weights and the evaluation numbers) are part of the repo.
#   bash fetch_human.sh <work folder>      (default ../../AI腔尺子_工作)
# Then: python build_human.py --work <work folder>
set -u
W="${1:-$(dirname "$0")/../../AI腔尺子_工作}"
H="$W/human_raw"
mkdir -p "$H/hc3" "$H/clue" "$H/cmrc" "$H/cedcc" "$H/weibo"
get() { for t in 1 2 3 4 5; do curl -sS -m 300 -L "$@" && return 0; sleep 4; done; return 1; }

# HC3-Chinese (Hello-SimpleAI): human answers written before ChatGPT; split licences are listed in the README.
for s in open_qa finance psychology baike nlpcc_dbqa medicine law; do
  get -o "$H/hc3/$s.jsonl" "https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/resolve/main/$s.jsonl"
done

# CLUECorpusSmall (MIT; from brightmart/nlp_chinese_corpus, texts up to 2019): 60 slices of 300 KB spread over the
# 13.7 GB file (HTTP range requests), not the whole file.
U="https://huggingface.co/datasets/austenjs/ClueCorpusSmallDataset/resolve/main/CLUECorpusSmall.txt"
N=13740512692
for i in $(seq 0 59); do
  off=$(( N / 60 * i + 1000 )); end=$(( off + 300000 ))
  get -r "$off-$end" -o "$H/clue/off_$(printf %02d "$i").txt" "$U"
done

# CMRC 2018 (CC BY-SA 4.0, Chinese Wikipedia paragraphs) through the Hugging Face datasets-server rows API
for off in 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
           2200 2400 2600 2800 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000; do
  get -o "$H/cmrc/rows_$off.json" \
    "https://datasets-server.huggingface.co/rows?dataset=hfl/cmrc2018&config=default&split=train&offset=$off&length=100"
done

# CEDCC (EMNLP 2023, middle-school essays written in 2021; no licence stated -> evaluation only)
get -o "$H/cedcc/CEDCC.json" "https://raw.githubusercontent.com/cubenlp/CEDCC_corpus/main/data/CEDCC.json"

# weibo_senti_100k (Sina Weibo posts, around 2012; no licence stated -> evaluation only)
get -o "$H/weibo/weibo_senti_100k.csv" \
  "https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k/resolve/main/weibo_senti_100k.csv"
echo done
