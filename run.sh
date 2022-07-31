 

function build { 
    g++ --std=c++11 shipit.cpp -o shipit
}

function shipit {
    mean="$1"
    for e in 2 4 8 16; do
        for m in 1 2 3 4 5 6; do
            ./optimize.py --fn shipit --method $m --error $e --mean $mean --sigma 1 -weeks 200000
        done
    done  
}

# build()

shipit -1
