# Shipit

## The Problem

Imagine you have infinite pool of ideas (product features) to test. You need to A/B-test them, but you have limited number of users and, therefore, limited number of events they are generating. You can consequently test ideas one after another using different criteria for stopping experiments choosing between two verdicts "ship it" or "discard". What is the best criteria if you want to optimise profit per week?

## Idea

Companies typically optimise their product offerings using randomised controlled trials (RCTs) in industry parlance this is known as A/B testing.
The rapid rise of A/B testing has led to the emergence of a number of widely used platforms that handle the implementation of these experiments.

A/B testing is about testing each idea of improving product and getting statistically significant answers about the hypothesis if an idea is useful or not.

But experiments require resources: time and objects/users that you allocate to each experiment experiment. So what is the best strategy for stopping experiment, shipping and discarding ideas when you have shortage of resources, and unlimited queue of ideas? In fact, this could be viewed as one more formulation of exploration vs exploitation problem. The well-known state-of-art approach for A/B testing with fixed size sample size is based on two conditions,  called significance and power conditions. To collect fixed size data usually you need fixed  period of time.  Fixed data size test is not optimal in the sense of minimising number of tested ideas in fied long period of time. The intuition for this is simple: some of really bad and really good ideas can be rejected and launched faster, before required (in average) data size is reached.

Naive approach of continuously monitoring p-values does not work, and provide unreliable results. But there are approach for sequential analysis that provides always valid estimations of p-values of both hypotheses: "test is better than control" and "test is worse than control". It's an interesting mathematical result and tangible step toward practionieers' demand.

But!!!

But there is the second important thing from real world: sometimes there is no need  to threshold significance and power of a test. In fact, one's goal may be just to maximise profit of launched ideas for some long period of time. And it is quite reasonable goal.

## Results
Here we experimenting with sequential A/B testing approach aiming to maximise profit where we do not bother about p-values of shipped ideas. We consider simple model where an ideas' relative profits are i.i.d gaussian random variables $\mathcal{N}(a_0, \sigma_0^2)$. Values $a_0$ and $\sigma_0$ are the only prior knowledge about any new idea.

The main result is that having continuously monitoring a posteriori values $a$ and $\sigma$ of an idea

- the condition $a > C_1\cdot \sigma + C_2$ is the best criterion for shipping, for some constants $C_1$ and $C_2$ depending on $(a_0, \sigma_0^2)$
- the condition $\sigma^2 \cdot \mathrm{PDF}(a,\; \sigma^2) + a \cdot \mathrm{CDF}(a, \sigma^2) < g_0$, where $g_0 =\sigma_0^2 \cdot \mathrm{PDF}(a,\; \sigma_0^2) + a \cdot \mathrm{CDF}(a_0, \sigma_0^2) < g_0$ is the best criterion for discarding.

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
