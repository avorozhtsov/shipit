#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<random>
#include<map>
#include<vector>
#include <algorithm>

std::random_device dev;
std::mt19937 rng{dev()};
std::normal_distribution<> norm_g(0.0, 1.0);

class TRng {
public:
    TRng(int seed_val) {
        rng.seed(seed_val);
    };

    double nrand() {
        return norm_g(rng);
    };
};


double Sqr(double& val) {
    return val * val;
}


void Help() {
    printf(
        "Usage: echo method seed steps base_mean base_sigma measurement_sigma param1 param2 ... | ./shipit\n"
        "  method: 0 = moss, 5 = p-value, 6 = g-value\n"
        "  seed = seed  for random generator\n"
        "  steps = number of steps; typical value is 1000000\n"
        "  base_mean = mean profit of a random idea; typical value is -1\n"
        "  base_mean = sigma = sqrt(D) of profit for ideas in the pool; typical value is 1\n"
        "  measurement_sigma = sigma of one measurement (one step); typical value is 8\n"
        "Example:\n"
        " echo \"6 123 100000 -1 1 8\" | ./shipit\n"
    );
}


class TIdea {
public:
    TIdea() {

    };

    TIdea(double Mean, double Sigma, double Weight): Mean(Mean), Sigma(Sigma), Weight(Weight) {


    };
    
    /* One week experiment updates prior distribution */
    void ExploreForAWeek(TRng &rng, double measurementSigma, double measurementWeight) {
        double thisWeekProfit = TrueProfit + measurementSigma * rng.nrand();
        Mean = (Mean * Weight + thisWeekProfit * measurementWeight) / (Weight + measurementWeight);
        Weight += measurementWeight;
        Sigma = 1.0 / sqrt(Weight);
        Weeks++;
    }

    void ReplaceWithNew(TRng &rng, const TIdea& pool) {
        Mean = pool.Mean;
        Sigma = pool.Sigma;
        Weight = pool.Weight;
        TrueProfit = Mean + Sigma * rng.nrand();
        Weeks = 0;
    }

    double Mean = 0;
    double Sigma = 1;
    double Weight = 1;
    double TrueProfit = 0;
    int Weeks = 0;
};


struct TParams {
    double ShipSigmas = 0.0;
    double StopSigmas = 0.0;
    double ShipMul = 0.0;
    double StopMul = 0.0;
    double Epsilon = 0.0;
    double Ksi = 0.0;
    double A = 0.0;
    double B = 0.0;
    double S = 0.0;
    double L = 0.0;
    double MaxTestWeeks = 0.0;
};

struct TResult {
    double Profit = 0.0;
    double Sigma = 0.0;
};

/* a-la Gittins Index for Gaussian Processes */
static inline double
GValueIndex(TIdea& idea, TParams& params) {
    double sigma = params.S * idea.Sigma;
    double sigma2 = Sqr(sigma);
    double mean = idea.Mean;
    return (
        sigma * exp(-0.5 * Sqr(mean) / sigma2) / sqrt(2.0 * M_PI) +
        mean * (0.5 + 0.5 * erf(mean / sigma / M_SQRT2)) +
        params.Ksi / sigma2
    );
}

/*  a-la UCB1 function*/
static inline double
MossIndex(TIdea& idea, TParams& params) {
    double sigma = params.S * idea.Sigma;
    double mean = idea.Mean;
    return (mean + sigma * sqrt(fmax(params.Ksi, params.L + log(sigma))));
}

/* Qn - fixed quantile */
static inline double
QnIndex(TIdea& idea, TParams& params) {
    return idea.Mean + params.S * idea.Sigma;
}

static inline double
ComplexIndex(TIdea& idea, TParams& params) {
    double sigma = params.S * idea.Sigma;
    double sigma2 = Sqr(sigma);
    double mean = idea.Mean;
    return (
        params.A * mean +
        params.Epsilon * (
            params.A * exp(M_SQRT2  * mean / params.Epsilon + sigma2 / Sqr(params.Epsilon)) +
            params.Ksi * params.Epsilon / sigma2
        )
    );
}

/* Ship experiments by pValue (-mean / sigma < StopSigmas to stop, and mean / sigma > ShipSigmas to ship) */
static TResult
EvaluateBanditsPValue(TRng &rng, int weeks, int trials, double baseMean, double baseSigma, double measurementSigma, TParams& params) {
    double totalProfit = 0;
    double totalProfitSqr = 0;
    TResult result;
    double baseWeight = 1.0 / Sqr(baseSigma);
    double measurementWeight = 1.0 / Sqr(measurementSigma);
    TIdea baseIdea{baseMean, baseSigma, baseWeight};

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.ReplaceWithNew(rng, baseIdea);
        double shippedProfit = 0;
        for (int week = 0; week < weeks; ++week) {
            idea.ExploreForAWeek(rng, measurementSigma, measurementWeight);
            if (-(idea.Mean - params.S * baseIdea.Mean) >  (idea.Sigma - params.S * baseIdea.Sigma) * params.StopSigmas || (idea.Weeks > params.MaxTestWeeks && idea.Mean < 0)) {
                idea.ReplaceWithNew(rng, baseIdea);
            } else if (idea.Mean > idea.Sigma * params.ShipSigmas || (idea.Weeks > params.MaxTestWeeks && idea.Mean > 0)) {
                shippedProfit += idea.TrueProfit;
                idea.ReplaceWithNew(rng, baseIdea);
            } // else continue exploring the idea
        }
        double avgProfit = shippedProfit / weeks;
        totalProfit += avgProfit;
        totalProfitSqr += Sqr(avgProfit);
    }
    result.Profit = totalProfit / trials;
    result.Sigma = sqrt((totalProfitSqr / trials -  Sqr(result.Profit)) / (trials - 1));
    return result;
}

/* Ship experiment by Index function. Current winner with indexFn == GValueIndex */
template <typename TIndexFn>
static TResult
EvaluateBanditsGValue(TRng &rng, TIndexFn indexFn, int weeks, int trials, double baseMean, double baseSigma, double measurementSigma, TParams& params) {
    double totalProfit = 0;
    double totalProfitSqr = 0;
    TResult result;
    double baseWeight = 1.0 / Sqr(baseSigma);
    double measurementWeight = 1.0 / Sqr(measurementSigma);
    TIdea baseIdea{baseMean, baseSigma, baseWeight};
    double baseIndex = indexFn(baseIdea, params);

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.ReplaceWithNew(rng, baseIdea);
        double shippedProfit = 0;
        for (int week = 0; week < weeks; ++week) {
            idea.ExploreForAWeek(rng, measurementSigma, measurementWeight);
            double ideaIndex = indexFn(idea, params);
            if (ideaIndex <= baseIndex) {
                idea.ReplaceWithNew(rng, baseIdea);
            } else if (idea.Mean >= ideaIndex * params.ShipMul) {
                shippedProfit += idea.TrueProfit;
                idea.ReplaceWithNew(rng, baseIdea);
            } // else continue exploring the idea
        }
        double avgProfit = shippedProfit / weeks;
        totalProfit += avgProfit;
        totalProfitSqr += Sqr(avgProfit);
    }
    result.Profit = totalProfit / trials;
    result.Sigma = sqrt((totalProfitSqr / trials -  Sqr(result.Profit)) / (trials - 1));
    return result;
}


/* same as previous with debug info */
template <typename TIndexFn>
static TResult
EvaluateBanditsGValueDebug(TRng &rng, TIndexFn indexFn, int weeks, int trials, double baseMean, double baseSigma, double measurementSigma, TParams& params) {
    double totalProfit = 0;
    double totalProfitSqr = 0;
    TResult result;
    double baseWeight = 1.0 / Sqr(baseSigma);
    double measurementWeight = 1.0 / Sqr(measurementSigma);
    TIdea baseIdea{baseMean, baseSigma, baseWeight};
    double baseIndex = indexFn(baseIdea, params);
    /*
    double unshippedPositives = 0;
    double unshippedPositivesPrior = 0;
    double shippedNegatives = 0;
    int unshippedPositivesCount= 0;
    int shippedNegativesCount = 0;
    */

    //TMap<int, int> ship;
    //TMap<int, int> stop;
    //int step;
    int trace_count = 0;
    unsigned long max_trace_size = 0;
    std::vector<std::pair<double, double>> trace;

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.ReplaceWithNew(rng, baseIdea);
        double shippedProfit = 0;
        //step = 1;
        for (int week = 0; week < weeks; ++week) {
            idea.ExploreForAWeek(rng, measurementSigma, measurementWeight);
            trace.push_back(std::make_pair(idea.Mean, idea.Sigma));
            double ideaIndex = indexFn(idea, params);
            if (ideaIndex <= baseIndex) {
                /* if (idea.Mean > 0) {
                    unshippedPositives += idea.TrueProfit;
                    unshippedPositivesPrior += idea.Mean;
                    unshippedPositivesCount++;
                } */
                if (max_trace_size + 10 < trace.size() || (trace.size() > 25 && trace_count < 5) || (trace.size() > 65 && trace_count < 10) || (trace.size() > 120 && trace_count < 15)) {
                    printf("{\"stop\", {\n {%lf, %lf}", baseIdea.Mean, baseIdea.Sigma);
                    for(auto p: trace) {
                        printf(", {%lf, %lf}", p.first, p.second);
                    }
                    printf("}},\n");
                    trace_count++;
                    max_trace_size = std::max(max_trace_size, trace.size());
                }
                trace.clear();
                idea.ReplaceWithNew(rng, baseIdea);
                //stop[step]++; step = 0;
            } else if (idea.Mean >= ideaIndex * params.ShipMul) {
                /*
                if (idea.Mean < 0) {
                     shippedNegatives += idea.TrueProfit; shippedNegativesCount++;
                }*/
                shippedProfit += idea.TrueProfit;
                idea.ReplaceWithNew(rng, baseIdea);
                if (max_trace_size + 10 < trace.size() || (trace.size() > 25 && trace_count < 6) || (trace.size() > 65 && trace_count < 11) || (trace.size() > 120 && trace_count < 16)) {
                    printf("{\"ship\", {\n{%lf, %lf}", baseIdea.Mean, baseIdea.Sigma);
                    for(auto p: trace) {
                        printf(", {%lf, %lf}", p.first, p.second);
                    }
                    printf("}},\n");
                    trace_count++;
                    max_trace_size = std::max(max_trace_size, trace.size());
                }
                trace.clear();
                //ship[step]++; step = 0;
            } // else continue exploring the idea
            //++step;
        }
        double avgProfit = shippedProfit / weeks;
        totalProfit += avgProfit;
        totalProfitSqr += Sqr(avgProfit);
    }
    result.Profit = totalProfit / trials;
    result.Sigma = sqrt((totalProfitSqr / trials -  Sqr(result.Profit)) / (trials - 1));
    /* printf(
        "unshipped: #=%d, p=%lf (%lf), avg=%lf\n",
        unshippedPositivesCount,
        unshippedPositives, unshippedPositivesPrior,
        unshippedPositives / (unshippedPositivesCount + 0.1)
    );
    printf("shipped: #=%d, p=%lf, avg=%lf\n", shippedNegativesCount, shippedNegatives, shippedNegatives / (shippedNegativesCount + 0.1));
    */

    /*
    printf("# type\tstep\tfreq\n");
    for (auto &pair: ship) {
        printf("1\t%d\t%d\n", pair.first, pair.second);
    }
    for (auto &pair: stop) {
        printf("0\t%d\t%d\n", pair.first, pair.second);
    }
    */
    return result;
}

template <typename TIndexFn>
static TResult
EvaluateBanditsCustom(TRng &rng, TIndexFn indexFn, int weeks, int trials, double baseMean, double baseSigma, double measurementSigma, TParams& params) {
    double totalProfit = 0;
    double totalProfitSqr = 0;
    TResult result;
    double baseWeight = 1.0 / Sqr(baseSigma);
    double measurementWeight = 1.0 / Sqr(measurementSigma);
    TIdea baseIdea{baseMean, baseSigma, baseWeight};
    double baseIndex = indexFn(baseIdea, params);

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.ReplaceWithNew(rng, baseIdea);
        double shippedProfit = 0;
        for (int week = 0; week < weeks; ++week) {
            idea.ExploreForAWeek(rng, measurementSigma, measurementWeight);
            double ideaIndex = indexFn(idea, params);
            if (ideaIndex <= params.StopMul * baseIndex || idea.Mean >= params.ShipMul * idea.Sigma || idea.Weeks > params.MaxTestWeeks) {
                if (idea.Mean >= 0) {
                    shippedProfit += idea.TrueProfit;
                }
                idea.ReplaceWithNew(rng, baseIdea);
            }
        }
        double avgProfit = shippedProfit / weeks;
        totalProfit += avgProfit;
        totalProfitSqr += Sqr(avgProfit);
    }
    result.Profit = totalProfit / trials;
    result.Sigma = sqrt((totalProfitSqr / trials -  Sqr(result.Profit)) / (trials - 1));
    return result;
}

template <typename TIndexFn>
static TResult
EvaluateBanditsCustom2(TRng &rng, TIndexFn indexFn, int weeks, int trials, double baseMean, double baseSigma, double measurementSigma, TParams& params) {
    double totalProfit = 0;
    double totalProfitSqr = 0;
    TResult result;
    double baseWeight = 1.0 / Sqr(baseSigma);
    double measurementWeight = 1.0 / Sqr(measurementSigma);
    TIdea baseIdea{baseMean, baseSigma, baseWeight};
    double baseIndex = indexFn(baseIdea, params);

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.ReplaceWithNew(rng, baseIdea);
        double shippedProfit = 0;
        for (int week = 0; week < weeks; ++week) {
            idea.ExploreForAWeek(rng, measurementSigma, measurementWeight);
            double ideaIndex = indexFn(idea, params);
            if (ideaIndex <= params.StopMul * baseIndex || idea.Mean >= params.ShipMul * ideaIndex || idea.Weeks > params.MaxTestWeeks) {
                if (idea.Mean >= 0) {
                    shippedProfit += idea.TrueProfit;
                }
                idea.ReplaceWithNew(rng, baseIdea);
            }
        }
        double avgProfit = shippedProfit / weeks;
        totalProfit += avgProfit;
        totalProfitSqr += Sqr(avgProfit);
    }
    result.Profit = totalProfit / trials;
    result.Sigma = sqrt((totalProfitSqr / trials -  Sqr(result.Profit)) / (trials - 1));
    return result;
}


int
main() {
    int method, seed, weeks, trials;
    double baseMean, baseSigma, measurementSigma;
    TResult result;
    TParams params;
    int inputs_count = 0;

    inputs_count += scanf("%d%d%d%d", &method, &seed, &weeks, &trials);
    inputs_count += scanf("%lf%lf%lf", &baseMean, &baseSigma, &measurementSigma);
    if (inputs_count < 7) {
        Help();
        return 1;
    }
    auto rng = TRng(seed);

    if (trials <= 1) {
        fprintf(stderr, "WARN: trials should be >= 2");
        trials = 2;
    }
    // params.MaxTestWeeks =  Max(2, (int)(0.5 + 1.673 * Sqr(0.72 + measurementSigma / baseSigma)));
    params.MaxTestWeeks = 100000; // fixed

    if (seed == 0) {
        rng = TRng(rand());
    }

    if (method == 1) { // Moss Index with shipCnd = (mean > c * sigma)
        params.S = 1.0;
        params.L = 2.0;
        params.ShipMul = 0.75;
        params.StopMul = 1.0;
        scanf("%lf%lf%lf%lf%lf", &params.S, &params.L, &params.ShipMul, &params.StopMul, &params.Ksi);
        result = EvaluateBanditsCustom(rng, MossIndex, weeks, trials, baseMean, baseSigma, measurementSigma, params);
    } else if (method == 2) { // MossIndex with shipCnd = (mean > c * Index)
        params.S = 1.0;
        params.L = 2.0;
        params.ShipMul = 0.75;
        params.StopMul = 1.0;
        scanf("%lf%lf%lf%lf%lf", &params.S, &params.L, &params.ShipMul, &params.StopMul, &params.Ksi);
        result = EvaluateBanditsCustom2(rng, MossIndex, weeks, trials, baseMean, baseSigma, measurementSigma, params);
    } else if (method == 3) { // pValue: stopCnd = ( mean < - StopSigmas * sigma), shipCnd = (mean > ShipSigmas * sigma)
        params.ShipSigmas = 0.3;
        params.StopSigmas = 1.0;
        scanf("%lf%lf", &params.ShipSigmas, &params.StopSigmas);
        result = EvaluateBanditsPValue(rng, weeks, trials, baseMean, baseSigma, measurementSigma, params);
    } else if (method == 4) { // same as prev, with stopping when weeks > MaxTestWeeks
        params.ShipSigmas = 0.3;
        params.StopSigmas = 1.0;
        scanf("%lf%lf%lf", &params.ShipSigmas, &params.StopSigmas, &params.MaxTestWeeks);
        result = EvaluateBanditsPValue(rng, weeks, trials, baseMean, baseSigma, measurementSigma, params);
    } else if (method == 5) { // same as prev, with base at point (S * base.Mean, S * base.Sigma)
        params.ShipSigmas = 0.5;
        params.StopSigmas = 1.9;
        params.S = 1.0;
        scanf("%lf%lf%lf", &params.ShipSigmas, &params.StopSigmas, &params.S);
        result = EvaluateBanditsPValue(rng, weeks, trials, baseMean, baseSigma, measurementSigma, params);
    } else if (method == 6) { // GValueIndex for stop and ship
        params.S = 1.0;
        params.ShipMul = 0.75;
        params.Ksi = 0.0; // fixed;
        scanf("%lf%lf", &params.ShipMul, &params.S);
        result = EvaluateBanditsGValue(rng, GValueIndex, weeks, trials, baseMean, baseSigma, measurementSigma, params);
    } else if (method == 7) { // GValueIndex for stop and ship and debug info
        params.S = 1.0;
        params.ShipMul = 0.75;
        params.Ksi = 0; // fixed;
        scanf("%lf%lf%lf", &params.ShipMul, &params.S, &params.Ksi);
        result = EvaluateBanditsGValueDebug(rng, GValueIndex, weeks, trials, baseMean, baseSigma, measurementSigma, params);
     } else {
        fprintf(stderr, "unknown method %d\n", method);
        Help();
        return 1;
    }
    double normCoeff = Sqr(measurementSigma);
    printf("%.8lg\n%.8g\n", normCoeff * result.Profit, normCoeff * result.Sigma);
    return 0;
}

