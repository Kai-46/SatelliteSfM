#!/usr/bin/env bash
cur_dir=$(pwd)

work_dir=$(dirname "$0")
cd $work_dir

if [[ ! -f "/usr/bin/gcc-7"  ]]; then
    echo "error: /usr/bin/gcc-7 does not exist."
    exit 1
fi

if [[ ! -f "/usr/bin/g++-7"  ]]; then
    echo "error: /usr/bin/g++-7 does not exist."
    exit 1
fi

if [[ ! -d "ColmapForVisSat" ]]; then
    git clone https://github.com/Kai-46/ColmapForVisSat.git
fi

CC=/usr/bin/gcc-7 CXX=/usr/bin/g++-7 \
            python3 ColmapForVisSat/scripts/python/build.py \
                            --build_path ColmapForVisSat/build \
                            --colmap_path ColmapForVisSat 2>&1 | tee ColmapForVisSat/build_log.txt
cd $cur_dir
