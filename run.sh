#!/usr/bin/env bash

PYTHON=python

default_pool_size=10
default_weeks=30000000
default_seed=2

set -e


function build {
    g++ -O3 --std=c++11 shipit.cpp -o shipit
}


function run_workers {
    cmd="$1"
    pool_size=${2:-$default_pool_size}
    echo -e "All commands:\n$cmds"
    echo "Workers pool size = $pool_size"
    echo -e "$cmds" | xargs -t -P "$pool_size" -L 1 time > /dev/null
}


function workers {
    pool_size=${1:-$default_pool_size}
    xargs -t -P "$pool_size" -L 1 time > /dev/null
}


function shipit_continue_points {
    filename=${1:-"shipit_results.txt"}
    weeks=${2:-"$default_weeks"}
    seed=${3:-"$default_seed"}
    pool_size=${4:-$default_pool_size}
    echo "Continue: $filename, weeks=$weeks, pool_size=$pool_size"
    $PYTHON ./optimize.py --command generate_cmd --src "$filename"  --weeks "$weeks" --seed "$seed" |
        grep -v SKIP |
        xargs -t -P "$pool_size" -L 1 time > /dev/null
}


function shipit {
    cmds=""
    weeks=${1:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    pool_size=${3:-"$default_pool_size"}
    for e in 2 4 6 8 10 16 22 32; do
        for m in 14 20 30 32; do
            for mean in -0.2 -0.5 -1.0 -1.5; do
                cmd="$PYTHON ./optimize.py --seed "$seed" --method $m --error $e --mean $mean --sigma 1 --weeks $weeks"
                cmds="$cmd\n$cmds"
            done;
        done
    done
    run_workers "$cmds" "$pool_size"
}


function shipit_sigma {
    cmds=""
    s=${1:-"1.0"}
    weeks=${2:-"$default_weeks"}
    seed=${2:-"$default_seed"}

    for e in 4 10 16 22 32 35; do
        for m in 14 20 30 32; do
            for mean in -1.0; do
                cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma $s --weeks $weeks"
                cmds="$cmd\n$cmds"
            done;
        done
    done
    run_workers "$cmds"
}


function shipit_mean {
    cmds=""
    mean=${1:-"-1.0"}
    weeks=${2:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    for e in 2 4 8 16 32; do
        for m in 14 20 30 32; do
            cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma 1 --weeks $weeks"
            cmds="$cmd\n$cmds"
        done
    done
    run_workers "$cmds"
}


function shipit_method {
    cmds=""
    m=${1:-"32"}
    weeks=${2:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    for e in 2 4 6 8 10 16 32 35; do
        for mean in -0.1 -0.2 -0.5 -0.8 -1.0 -1.2 -1.5 -1.8 -2.0; do
            cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma 1 --weeks $weeks"
            cmds="$cmd\n$cmds"
        done
    done
    run_workers "$cmds"
}

function shipit_main {
    cmds=""
    m=${1:-"21"}
    weeks=${2:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    for e in 2 3 4 6 8 9 10 16 22 32 35; do
    # for e in 3 12 25 30 35; do
        for mean in -1 -0.5 -1.5; do
            cmd="$PYTHON ./optimize.py --prefix shipit_points/pt2_ --seed $seed --method $m --error $e --mean $mean --sigma 1 --weeks $weeks"
            cmds="$cmd\n$cmds"
        done
    done
    run_workers "$cmds"
}


function shipit_selected {
    cmds=""
    weeks=${1:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    for m in 14 20 30 32; do

        sigma=0.5;
        mean="-0.2"
        e=35;

        cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        cmds="$cmd\n$cmds"

        sigma=1;
        mean="-1.5"

        cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        cmds="$cmd\n$cmds"

    done

    run_workers "$cmds"
}


function shipit_e35 {
    cmds=""
    weeks=${1:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    for e in 35; do
        for m in 14 20 30 32; do
            for mean in -0.2 -0.5 -0.7 -1.0 -1.5 -2.0; do
                for sigma in 1; do
                    cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
                    cmds="$cmd\n$cmds"
                done
            done
        done
    done
    run_workers "$cmds"
}


function shipit_best {
    cmds=""
    weeks=${1:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    # for e in 4 6 8 10 16 22 32 35; do
    for e in 32 35; do
        for m in 14 20 30 32; do
            for mean in -0.2 -0.5 -0.7 -1.0 -1.5; do
                for sigma in 1; do
                    cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
                    cmds="$cmd\n$cmds"
                done
            done
        done
    done
    run_workers "$cmds"
}


function shipit_ls_grid {
    cmds=""
    for e in 2 4 6 8 16 32; do
        for m in 14 20 30 32; do
            for mean in -0.2 -0.5 -1.0 -1.5; do
                for sigma in 0.7 1.5 1 2 4 8 10; do
                    mean_p=$(echo $mean | perl -pe 's/\-//')
                    suffix="m${m}_a${mean_p}s1e${e}"
                    # echo
                    -f "shipit_points/pt_${suffix}.txt" || echo "NO shipit_points/pt_${suffix}.txt"
                    ls "YES shipit_points/*${suffix}*.txt"
                done
            done
        done
    done
}


function calc_results {
    weeks=${1:-"20000000"}
    seed=${2:-"4"}
    pool_size=${3:-"$default_pool_size"}
    echo "calculating result"
    echo "  weeks = $weeks, seed = $seed"
    echo "  workers = $pool_size"
    # use weeks=10000000 and special seed = 4 used only to calculate results
    echo "pt=\$1; cat \$pt | perl -pe 's/^(\d+) (\d+) (\d+) /\$1 $seed $weeks /g' | tee \$pt.in | ./shipit > \$pt.tmp; mv \$pt.tmp \$pt.result" > calc_result.sh
    chmod +x calc_result.sh
    cmds=""
    for pt in `find shipit_points/ -name '*.txt'`; do
        if [ ! -f "$pt.result" ] || [ "$pt" -nt "$pt.result" ] ; then
            cmd="./calc_result.sh $pt"
            cmds="$cmd\n$cmds"
        fi
    done
    run_workers "$cmds" "$pool_size"
}


# cats tab-separated points data with result and result_sigma two first columns
function print_results {
    (
        echo result,result_sigma,m,seed,weeks,trials,mean,sigma,error,p1,p2,p3,p4,p5 |
            perl -pe 's/,/\t/g'
        find shipit_points/ -name '*.txt' |
            sort |
            for pt in `cat`; do
                if [ -f "$pt.result" ]; then
                    cat "$pt.result" | perl -pe 's/\n/ /g' | perl  -pe 's/ +/ /g'
                    cat "$pt.in" | cut -d ' ' -f 1-12
                    # cat "$pt" | cut -d ' ' -f 1-12
                fi
            done |
            perl -pe 's/ +/\t/g'

    ) | sort -t$'\t' -gk3,3 -gk6,9
}


"$@"
