#!/usr/bin/env bash

PYTHON=python

default_pool_size=5
default_weeks=30000000

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

function continue_points {
    filename=${1:-"shipit_results.txt"}
    weeks=${2:-"$default_weeks"}
    pool_size=${3:-$default_pool_size}
    echo "Continue: $filename, weeks=$weeks, pool_size=$pool_size"
    $PYTHON ./optimize.py --command generate_cmd --src "$filename"  --weeks 10000000 |
        grep -v SKIP |
        xargs -t -P "$pool_size" -L 1 time > /dev/null
}

function shipit {
    cmds=""
    for e in 2 4 6 8 16 22 32; do
        for m in 1 2 3 4 5 6 7; do
            for mean in -0.2 -0.5 -1.0 -1.5; do
                cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma 1 --weeks $default_weeks"
                cmds="$cmd\n$cmds"
            done;
        done
    done
    run_workers "$cmds" 10
}

function shipit_sigmas {
    cmds=""
    e=${1:-"16"}
    for s in 0.7 1.5 1 2 4 8 10; do
        for m in 2 7; do
            for mean in -1.0; do
                cmd="$PYTHON ./optimize.py --prefix shipit_points/pt --fn shipit --method $m --error $e --mean $mean --sigma $s --weeks $default_weeks"
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
    for e in 2 4 8 16 32; do
        for m in 1 2 3 4 5 6; do
            cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma 1 --weeks $weeks"
            cmds="$cmd\n$cmds"
        done
    done
    run_workers "$cmds"
}

function shipit_method {
    cmds=""
    m=${1:-"71"}
    weeks=${2:-"$default_weeks"}
    for e in 2 4 6 8 16 32; do
        for mean in -0.1 -0.2 -0.5 -0.8 -1.0 -1.2 -1.5 -1.8 -2.0; do
            cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma 1 --weeks $weeks"
            cmds="$cmd\n$cmds"
        done
    done
    run_workers "$cmds"
}

function shipit_selected {
    cmds=""
    weeks=${1:-"$default_weeks"}
    for e in 3; do
        m=7
        mean="-1"; sigma=1;
        cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        cmds="$cmd\n$cmds"
        sigma=0.5;
        cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        cmds="$cmd\n$cmds"
        sigma=0.64;
        cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        cmds="$cmd\n$cmds"

        # m=51
        # mean="-0.5"; sigma=0.64
        # cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        # cmds="$cmd\n$cmds"
        # mean="-0.2"; sigma=0.5
        # cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        # cmds="$cmd\n$cmds"
        # mean="-0.5"; sigma=0.82
        # cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        # cmds="$cmd\n$cmds"
        # mean="-1.2"; sigma=1.0
        # cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
        # cmds="$cmd\n$cmds"
    done

    run_workers "$cmds"
}


function shipit_best {
    cmds=""
    for e in 4 6 8 16 20 22 32 35; do
        # for m in 2 5 7; do
        for m in 51; do
            # for mean in -0.06 -0.1 -0.2 -0.7 -1.0 -1.5 -2.0; do
            for mean in -1.0; do
                for sigma in 1; do
                    cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks 15000000"
                    cmds="$cmd\n$cmds"
                done
            done
        done
    done
    run_workers "$cmds"
}

function shipit_52 {
    cmds=""
    for e in 4 6 8 16 20 35 22 32; do
        # for m in 2 5 7; do
        for m in 52; do
            # for mean in -0.06 -0.1 -0.2 -0.7 -1.0 -1.5 -2.0; do
            for mean in -1.0; do
                for sigma in 1; do
                    cmd="$PYTHON ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma $sigma --weeks 1000000"
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
        for m in 1 2 4 5 6 7; do
            for mean in -0.2 -0.5 -1.0 -1.5; do
                for sigma in 0.7 1.5 1 2 4 8 10; do
                    mean_p=$(echo $mean | perl -pe 's/\-//')
                    suffix="m${m}_a${mean_p}s1e${e}"
                    # echo
                    -f "shipit_points/pt_${suffix}.txt" || echo "NO $suffix"
                    ls "YES shipit_points/*${suffix}*.txt"
                done
            done
        done
    done
}

function calc_results {
    # use weeks=10000000 and special seed = 4 used only to calculate results
    echo "pt=\$1; cat \$pt | perl -pe 's/^(\d+) (\d+) (\d+) /\$1 4 20000000 /g' | ./shipit > \$pt.tmp; mv \$pt.tmp \$pt.result" > calc_result.sh
    chmod +x calc_result.sh
    cmds=""
    for pt in `find shipit_points/ -name '*.txt'`; do
        if [ ! -f "$pt.result" ] || [ "$pt" -nt "$pt.result" ] ; then
            cmd="./calc_result.sh $pt"
            cmds="$cmd\n$cmds"
        fi
    done
    run_workers "$cmds"
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
                    # cat "$pt" | perl -pe 's/^(\d+) (\d+) (\d+) /$1 4 10000000 /g'
                    cat "$pt" | cut -d ' ' -f 1-12
                fi
            done |
            perl -pe 's/ +/\t/g'

    )
}

"$@"
