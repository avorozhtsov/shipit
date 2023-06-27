# Shipit

## The Problem

Imagine you have an infinite pool of versions (product features, ideas) to test. You need to A/B-test them, but you have limited number of users and, therefore, limited number of events they are generating. You can consequently test versions one after another using a criteria for stopping and choosing between two verdicts: "launch" or "discard". What is the best criteria if you want to optimise profit per time?

## Idea

A/B testing is about testing versions of product and getting statistically significant answer to the question is the version better than control or not.

But experiments require resources: time and objects/users that you allocate to each experiment. So what is the best strategy for stopping experiment, launching and discarding versions when you have shortage of resources, and unlimited queue of versions to test? In fact, this could be viewed as one more statement of exploration vs exploitation problem. The well-known state-of-art approach to A/B testing with fixed sample size is to estimate the probability P[gain > 0], and launch if P[gain > 0] > P_threshold.  Fixed sample size test is not optimal. The intuition behind this is simple: some of the really bad and really good ideas can be discarded and launched earlier, before required sample size is achieved.

The naive approach of continuously monitoring the value of P[gain > 0] does not work, and provides unreliable results. But there are approaches for sequential A/B-testing that provide valid predefined false rates of "launched" and "discarded" versions. This is an interesting mathematical result and a tangible step towards practioners' demands.

But!!!

But there is second important thing from the real world: sometimes there is no need to bother about false rates. In fact, one's goal may be to maximise profit from launched versions over a long period of time. And it is quite reasonable goal.

## Results
Here we are experimenting with sequential A/B testing approach aiming to maximise profit per time. And we do not bother about false rates of launched and rejected versions. We consider simple model where versions profit gains are i.i.d gaussian random variables $\mathcal{N}(a_0, \sigma_0^2)$. Values $a_0$ and $\sigma_0$ are the only prior knowledge about a new version.

The main result is that having a posteriori values of $a$ and $\sigma$ of a version

- the classic "launch" and "dicard" criteria  $a > z_{launch}\cdot \sigma$ and  $-a > z_{discard}\cdot \sigma$ are not effective for any constants $z_{launch}$ and $z_{discard}$;
- the shifted linear criteria $a > z \cdot (\sigma - \sigma_{shift})$ and $-a > z \cdot (\sigma - \sigma_{shift})$ are very good but still not optimal, and constants $z$ and $\sigma_{shift}$ depends on $(a_0, \sigma_0^2)$ and the point $(a_0, \sigma_0)$ sits on the "discard" border;
- the optimal criteria are $\sigma^2 \cdot \mathrm{PDF}(a, \sigma^2) + a \cdot \mathrm{CDF}(a, \sigma^2) -\xi /\sigma^2 < g_0$ and symmetrical condition with $-a$ instead of $a$; $g_0$ is the value of the right part for $(a, \sigma) = (a_0, \sigma_0)$, i.e. again, the point $(a_0, \sigma_0)$ sits on the "discard" border.

## Required

- python -m pip install scipy scikit-optimize diskcache

## Usage

- To build binary shipit
```
./run.sh build
```


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
