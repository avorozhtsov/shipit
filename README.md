# Shipit

Imagine you have infinite pool of ideas to test.
You need to AB-test them have limited number if users and events they are generating.
You can consiquently test idead one after another using different criteria for stopping experiments and diciding "ship it" or "discard".
What is the best criteria if you want to optimize profit per week?


## Required

- python -m pip install scipy scikit-optimize diskcache

## Work

- To get initial set of points by youself:
```
./run.sh shipit_best 600000
```

- To generate precalculated points set:
```
python ./optimize.py --command results2points --src shipit_results.txt --prefix shipit_points/pt_
```
Use `--drurun` option to see output without creating any files.

- To generate results for all new points:
```
./run.sh calc_results
```

- To generate shipit_results.txt from all points (after calc_results):
```
./run.sh print_results > shipit_results.txt
```

- To continue searching for better parameters:
```
./run.sh shipit_continue_points shipit_results.txt
```

- To run some special tasks for searching better parameters:
```
./run.sh shipit_mean -0.5 30000000
./run.sh shipit_sigma 0.7 30000000
./run.sh shipit_method 51 30000000
```