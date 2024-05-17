#!/bin/bash
# Adapted from https://github.com/facebookresearch/fairseq/blob/main/examples/translation/prepare-wmt14en2de.sh

if [ ! -d mosesdecoder ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v9.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v9.de-en"
)

src=en
tgt=de
prep=dataset/wmt14en_de
orig=orig

mkdir -p $orig $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        tar zxvf $file
    fi
done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    if [ -f $prep/train.$l ]; then
        echo "$prep/train.$l was already processed, skipping."
    else
        for f in "${CORPORA[@]}"; do
            cat $orig/$f.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 24 -a -l $l >> $prep/train.$l
        done
    fi
done

echo "pre-processing test data..."
for l in $src $tgt; do
    rm $prep/test.$l
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 24 -a -l $l > $prep/test.$l
    echo ""
done
