# Shipit

Imagine you have infinite pool of ideas to test.
You need to AB-test them have limited number if users and events they are generating.
You can consiquently test idead one after another using different criteria for stopping experiments and diciding "ship it" or "discard".
What is the best criteria if yoy want to optimize profit per week?


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