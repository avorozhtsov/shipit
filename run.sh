#!/usr/bin/env bash

PYTHON=python

default_pool_size=5
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
    for e in 2 4 6 8 10 16 22 32; do
        for m in 1 2 3 4 5 51 52 6 7 71; do
            for mean in -0.2 -0.5 -1.0 -1.5; do
                cmd="$PYTHON ./optimize.py --seed "$seed" --method $m --error $e --mean $mean --sigma 1 --weeks $weeks"
                cmds="$cmd\n$cmds"
            done;
        done
    done
    run_workers "$cmds" 10
}


function shipit_sigma {
    cmds=""
    s=${1:-"1.0"}
    weeks=${2:-"$default_weeks"}
    seed=${2:-"$default_seed"}

    for e in 4 10 16 22 32 35; do
        for m in 1 5 51 52 7 71 ; do
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
        for m in 1 2 3 4 5 6; do
            cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma 1 --weeks $weeks"
            cmds="$cmd\n$cmds"
        done
    done
    run_workers "$cmds"
}


function shipit_method {
    cmds=""
    m=${1:-"71"}
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


function shipit_selected {
    cmds=""
    weeks=${1:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    for m in 7 71 51 52; do

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
        # for m in 1 7 52 71; do
        for m in 5 51 7 71 52; do
            # for mean in -0.06 -0.1 -0.2 -0.7 -1.0 -1.5 -2.0; do
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
        # for m in 1 7 52 71; do
        for m in 5 52 7 71; do
            # for mean in -0.06 -0.1 -0.2 -0.7 -1.0 -1.5 -2.0; do
            for mean in -0.5 -1.0 -1.5; do
                for sigma in 1; do
                    cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma $sigma --weeks $weeks"
                    cmds="$cmd\n$cmds"
                done
            done
        done
    done
    run_workers "$cmds"
}


function shipit_51e35 {
    cmds=""
    weeks=${1:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    for e in 35; do
        for m in 51; do
            for mean in -0.06 -0.1 -0.2 -0.5 -0.7 -1.0 -1.5 -2.0; do
            # for mean in -1.0; do
                for sigma in 1; do
                    cmd="$PYTHON ./optimize.py --seed $seed --method $m --error $e --mean $mean --sigma $sigma --weeks 1000000"
                    cmds="$cmd\n$cmds"
                done
            done
        done
    done
    run_workers "$cmds"
}


function shipit_51 {
    cmds=""
    weeks=${1:-"$default_weeks"}
    seed=${2:-"$default_seed"}
    for e in 2 4 6 8 10 16 20 22 32 35; do
        for m in 51; do
            for mean in -1.0; do
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
        for m in 1 2 4 5 6 7; do
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
