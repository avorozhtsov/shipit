#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<random>
#include<map>
#include<vector>
#include <algorithm>

std::random_device dev;

class TRng {
public:

    std::mt19937 rng;
    std::normal_distribution<> norm_g;

    TRng(int seed_val): rng(dev()), norm_g(0.0, 1.0) {
        rng.seed(seed_val);
    }

    double nrand() {
        return norm_g(rng);
    }
};

inline
double sqr(double val) {
    return val * val;
}


void help() {
    printf(
        "Usage: echo method seed weeks trials base_mean base_sigma week_sigma param1 param2 ... | ./shipit\n"
        "  method: 10, 11, ..., 15 = p-value, 20, 21 = moss, 30, 31 = g-value\n"
        "  seed = seed for random generator\n"
        "  weeks = number of steps; typical value is 10000000\n"
        "  trials = number of cycles, each weeks steps; used for calculating profit std; use 25\n"
        "  base_mean = mean profit of a random idea; typical value is -1\n"
        "  base_sigma = sigma = sqrt(D) of profit for ideas in the pool; typical value is 1\n"
        "  week_sigma = sigma of one measurement (aka error); for inst. 8\n\n"
        "Examples:\n"
        "  echo \"14 123 1000000 25 -1 1 16 1.4 0\" | ./shipit\n"
        "  echo \"30 123 1000000 25 -1 1 16 0.746 1.0 -0.02\" | ./shipit\n\n"
        "Output:\n"
        "  Two numbers: avg profit and std of profit over trials.\n\n"
        "Parameters:\n"
        "  P-Value family - lines on the plane (mean, sigma)\n"
        "    10: (ship_sigmas, stop_sigmas, sigma_mul, sigma_mul, max_test_weeks)\n"
        "    11: (ship_sigmas, stop_sigmas); sigma_mul = 0, stop_mul = 0\n"
        "    12: (ship_sigmas, stop_sigmas, sigma_mul); stop_mul = 1\n"
        "    13: (ship_sigmas, stop_sigmas); stop_mul = 1, sigma_mul = auto\n"
        "    14: (stop_sigmas, a); ship_sigmas = stop_sigmas + a; stop_mul=1, sigma_mul = auto\n"
        "    15: (stop_sigmas); ship_sigmas = stop_sigmas; stop_mul = 1, sigma_mul = auto\n\n"
        "  Moss-index\n"
        "    20: (S, L, ship_sigmas, ksi)\n"
        "    21: (S, L, ksi); ship_sigmas = auto\n\n"
        "  g-index\n"
        "    30: (S, ship_mul, ksi)\n"
        "    31: (S, ship_mul); ksi = 0\n"
        "    32: (S, ship_mul); ksi = auto\n"
    );
}


class TIdea {
public:
    TIdea() {

    };

    TIdea(double mean, double sigma, double weight): mean(mean), sigma(sigma), weight(weight) {

    };

    TIdea(double mean, double sigma): mean(mean), sigma(sigma) {
        weight = 1 / sqr(sigma);
    };

    /* One week experiment updates _prior distribution */
    void explore_week(TRng &rng, double week_sigma, double week_weight) {
        double week_profit = true_profit + week_sigma * rng.nrand();
        mean = (mean * weight + week_profit * week_weight) / (weight + week_weight);
        weight += week_weight;
        sigma = 1.0 / sqrt(weight);
        Weeks++;
    }

    void replace_with(TRng &rng, const TIdea& pool) {
        mean = pool.mean;
        sigma = pool.sigma;
        weight = pool.weight;
        true_profit = mean + sigma * rng.nrand();
        Weeks = 0;
    }

    double mean = 0;
    double sigma = 1;
    double weight = 1;
    double true_profit = 0;
    int Weeks = 0;
};

// parameters for all methods & indexes
struct TParams {
    double s = 0.0;
    double ship_sigmas = 0.0;
    double stop_sigmas = 0.0;
    double sigma_mul = 0.0;
    double ship_mul = 0.0;
    double stop_mul = 0.0;
    double Epsilon = 0.0;
    double ksi = 0.0;
    double l = 0.0;
    double max_test_weeks = 0.0;
};

struct TResult {
    double profit = 0.0;
    double sigma = 0.0;
};

/* a-la Gittins Index for Gaussian Processes */
static inline double
g_index(TIdea& idea, TParams& params) {
    double sigma = params.s * idea.sigma;
    double sigma2 = sqr(sigma);
    double mean = idea.mean;
    return (
        sigma * exp(-0.5 * sqr(mean) / sigma2) / sqrt(2.0 * M_PI) +
        mean * (0.5 + 0.5 * erf(mean / sigma / M_SQRT2)) +
        params.ksi / sigma2
    );
}

/*  a-la UCB1 function*/
static inline double
moss_index(TIdea& idea, TParams& params) {
    return idea.mean + params.s * idea.sigma * sqrt(fmax(params.ksi, params.l + log(idea.sigma)));
}

/* Qn - fixed quantile */
static inline double
qn_index(TIdea& idea, TParams& params) {
    return idea.mean + params.s * idea.sigma;
}


/*
Ship experiments by pValue
For p.stop_mul = 0, and p.sigma_mul = 0:
    ship:  mean / sigma > ship_sigmas
    stop: -mean / sigma < stop_sigmas
General case:
    ship:  mean - sigma * ship_sigmas > sigma0 * ship_sigmas
    stop:  mean + sigma * stop_sigmas < p.stop_mul * (base_mean + base_sigma * stop_sigmas)
    where sigma0 = p.sigma_mul * base_sigma
*/
static TResult
eval_pvalue(
    TRng &rng, int weeks, int trials,
    double base_mean, double base_sigma, double week_sigma,
    TParams& params
) {
    double total_profit = 0;
    double total_profit_sqr = 0;
    TResult result;
    double base_weight = 1.0 / sqr(base_sigma);
    double week_weight = 1.0 / sqr(week_sigma);
    TIdea base_idea{base_mean, base_sigma, base_weight};
    double sigma0 = params.sigma_mul * base_idea.sigma;

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.replace_with(rng, base_idea);
        double shipped_profit = 0;
        for (int week = 0; week < weeks; ++week) {
            idea.explore_week(rng, week_sigma, week_weight);
            double qn_base_index = base_idea.mean + base_idea.sigma * params.stop_sigmas;
            double qn_index = idea.mean + idea.sigma * params.stop_sigmas;
            if (
                qn_index < params.stop_mul * qn_base_index
                || (idea.Weeks > params.max_test_weeks && idea.mean < 0)
            ) {
                // stop & discard
                idea.replace_with(rng, base_idea);
            } else if (
                idea.mean > (idea.sigma - sigma0) * params.ship_sigmas
                || (idea.Weeks > params.max_test_weeks)
            ) {
                // shipit!
                shipped_profit += idea.true_profit;
                idea.replace_with(rng, base_idea);
            } // else continue exploration
        }
        double avg_profit = shipped_profit / weeks;
        total_profit += avg_profit;
        total_profit_sqr += sqr(avg_profit);
    }
    result.profit = total_profit / trials;
    result.sigma = sqrt((total_profit_sqr / trials -  sqr(result.profit)) / (trials - 1));
    return result;
}

/*
Ship experiment by Index function. The leader is index_fn == g_index
    ship: mean  > p.ship_mul * index
    stop: index < p.stop_mul * base_index
Hint: p.stop_mul == 1 is the best
*/
template <typename Tindex_fn>
static TResult
eval_index(
    TRng &rng, Tindex_fn index_fn, int weeks, int trials,
    double base_mean, double base_sigma, double week_sigma,
    TParams& params
) {
    double total_profit = 0;
    double total_profit_sqr = 0;
    TResult result;
    double base_weight = 1.0 / sqr(base_sigma);
    double week_weight = 1.0 / sqr(week_sigma);
    TIdea base_idea{base_mean, base_sigma};
    double base_index = index_fn(base_idea, params);

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.replace_with(rng, base_idea);
        double shipped_profit = 0;
        for (int week = 0; week < weeks; ++week) {
            idea.explore_week(rng, week_sigma, week_weight);
            double idea_index = index_fn(idea, params);
            if (idea_index <= base_index) {
                // stop & discard
                idea.replace_with(rng, base_idea);
            } else if (idea.mean >= idea_index * params.ship_mul) {
                // shipit!
                shipped_profit += idea.true_profit;
                idea.replace_with(rng, base_idea);
            } // else continue exploration
        }
        double avg_profit = shipped_profit / weeks;
        total_profit += avg_profit;
        total_profit_sqr += sqr(avg_profit);
    }
    result.profit = total_profit / trials;
    result.sigma = sqrt((total_profit_sqr / trials -  sqr(result.profit)) / (trials - 1));
    return result;
}


/* same as previous with debug info */
template <typename Tindex_fn>
static TResult
eval_index_debug(
    TRng &rng, Tindex_fn index_fn, int weeks, int trials,
    double base_mean, double base_sigma, double week_sigma,
    TParams& params
) {
    double total_profit = 0;
    double total_profit_sqr = 0;
    TResult result;
    double base_weight = 1.0 / sqr(base_sigma);
    double week_weight = 1.0 / sqr(week_sigma);
    TIdea base_idea{base_mean, base_sigma};
    double base_index = index_fn(base_idea, params);
    /*
    double unshipped_positives = 0;
    double unshipped_positives_prior = 0;
    double shipped_negatives = 0;
    int unshipped_positives_count= 0;
    int shipped_negatives_count = 0;
    */

    //TMap<int, int> ship;
    //TMap<int, int> stop;
    //int step;
    int trace_count = 0;
    unsigned long max_trace_size = 0;
    std::vector<std::pair<double, double>> trace;

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.replace_with(rng, base_idea);
        double shipped_profit = 0;
        //step = 1;
        for (int week = 0; week < weeks; ++week) {
            idea.explore_week(rng, week_sigma, week_weight);
            trace.push_back(std::make_pair(idea.mean, idea.sigma));
            double idea_index = index_fn(idea, params);
            if (idea_index <= 0.2 * base_index) {
                /* if (idea.true_profit > 0) {
                    unshipped_positives += idea.true_profit;
                    unshipped_positives_prior += idea.mean;
                    unshipped_positives_count++;
                } */
                if (
                    max_trace_size + 10 < trace.size()
                    || (trace.size() > 25 && trace_count < 5)
                    || (trace.size() > 65 && trace_count < 10)
                    || (trace.size() > 120 && trace_count < 16)
                ) {
                    printf("{\"stop\", {\n {%lf, %lf}", base_idea.mean, base_idea.sigma);
                    for(auto p: trace) {
                        printf(", {%lf, %lf}", p.first, p.second);
                    }
                    printf("}},\n");
                    trace_count++;
                    max_trace_size = std::max(max_trace_size, trace.size());
                }
                trace.clear();
                idea.replace_with(rng, base_idea);
                //stop[step]++; step = 0;
            } else if (idea.mean >= idea_index * params.ship_mul) {
                /*
                if (idea.true_profit < 0) {
                     shipped_negatives += idea.true_profit; shipped_negatives_count++;
                }*/
                shipped_profit += idea.true_profit;
                idea.replace_with(rng, base_idea);
                if (
                    max_trace_size + 10 < trace.size()
                    || (trace.size() > 25 && trace_count < 6)
                    || (trace.size() > 65 && trace_count < 11)
                    || (trace.size() > 120 && trace_count < 16)
                ) {
                    printf("{\"ship\", {\n{%lf, %lf}", base_idea.mean, base_idea.sigma);
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
        double avg_profit = shipped_profit / weeks;
        total_profit += avg_profit;
        total_profit_sqr += sqr(avg_profit);
    }
    result.profit = total_profit / trials;
    result.sigma = sqrt((total_profit_sqr / trials -  sqr(result.profit)) / (trials - 1));
    /* printf(
        "unshipped: #=%d, p=%lf (%lf), avg=%lf\n",
        unshipped_positives_count,
        unshipped_positives, unshipped_positives_prior,
        unshipped_positives / (unshipped_positives_count + 0.1)
    );
    printf("shipped: #=%d, p=%lf, avg=%lf\n", shipped_negatives_count, shipped_negatives, shipped_negatives / (shipped_negatives_count + 0.1));
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

template <typename Tindex_fn>
static TResult
eval_index_pvalue(
    TRng &rng, Tindex_fn index_fn, int weeks, int trials,
    double base_mean, double base_sigma, double week_sigma,
    TParams& params
) {
    double total_profit = 0;
    double total_profit_sqr = 0;
    TResult result;
    double base_weight = 1.0 / sqr(base_sigma);
    double week_weight = 1.0 / sqr(week_sigma);
    TIdea base_idea{base_mean, base_sigma};
    double base_index = index_fn(base_idea, params);
    double sigma0 = params.sigma_mul * base_idea.sigma;

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.replace_with(rng, base_idea);
        double shipped_profit = 0;
        for (int week = 0; week < weeks; ++week) {
            idea.explore_week(rng, week_sigma, week_weight);
            double idea_index = index_fn(idea, params);
            if (idea_index <= base_index) {
                // stop & discard
                idea.replace_with(rng, base_idea);
            } else {
                // The only block that differs from eval_index
                if (idea.mean > (idea.sigma - sigma0) * params.ship_sigmas) {
                    // shipit!
                    shipped_profit += idea.true_profit;
                    idea.replace_with(rng, base_idea);
                 } // else continue exploration
            }
        }
        double avg_profit = shipped_profit / weeks;
        total_profit += avg_profit;
        total_profit_sqr += sqr(avg_profit);
    }
    result.profit = total_profit / trials;
    result.sigma = sqrt((total_profit_sqr / trials -  sqr(result.profit)) / (trials - 1));
    return result;
}

template <typename Tindex_fn>
static TResult
eval_index_sym(
    TRng &rng, Tindex_fn index_fn, int weeks, int trials,
    double base_mean, double base_sigma, double week_sigma,
    TParams& params
) {
    double total_profit = 0;
    double total_profit_sqr = 0;
    TResult result;
    double base_weight = 1.0 / sqr(base_sigma);
    double week_weight = 1.0 / sqr(week_sigma);
    TIdea base_idea{base_mean, base_sigma};
    double base_index = index_fn(base_idea, params);

    for (int trial = 0; trial < trials; ++trial) {
        TIdea idea;
        idea.replace_with(rng, base_idea);
        double shipped_profit = 0;
        for (int week = 0; week < weeks; ++week) {
            idea.explore_week(rng, week_sigma, week_weight);
            double idea_index = index_fn(idea, params);
            if (idea_index <= base_index) {
                // stop & discard
                idea.replace_with(rng, base_idea);
            } else {
                // The only block that differs from eval_index
                TIdea mirror_idea{-idea.mean, idea.sigma};
                double mirror_idea_index = index_fn(mirror_idea, params);
                if (mirror_idea_index <= base_index) {
                    // shipit!
                    shipped_profit += idea.true_profit;
                    idea.replace_with(rng, base_idea);
                 } // else continue exploration
            }
        }
        double avg_profit = shipped_profit / weeks;
        total_profit += avg_profit;
        total_profit_sqr += sqr(avg_profit);
    }
    result.profit = total_profit / trials;
    result.sigma = sqrt((total_profit_sqr / trials -  sqr(result.profit)) / (trials - 1));
    return result;
}

int
main(int argc, char* argv[]) {
    // yes, I know, it's bad
    if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'h') {
        help();
        return 0;
    }

    int method, seed, weeks, trials;
    double base_mean, base_sigma, week_sigma;
    TResult result;
    TParams params;
    int inputs_count = 0;

    inputs_count += scanf("%d%d%d%d", &method, &seed, &weeks, &trials);
    inputs_count += scanf("%lf%lf%lf", &base_mean, &base_sigma, &week_sigma);
    if (inputs_count < 7) {
        help();
        return 1;
    }

    if (seed == 0) {
        seed = time(nullptr);
    }

    TRng rng = TRng(seed);

    if (trials <= 1) {
        fprintf(stderr, "WARN: trials should be >= 2");
        trials = 2;
    }
    // params.max_test_weeks =  Max(2, (int)(0.5 + 1.673 * sqr(0.72 + week_sigma / base_sigma)));
    params.max_test_weeks = 100000.0; // fixed
    params.ksi = 0.0;
    params.sigma_mul = 0.0;
    TIdea base_idea{base_mean, base_sigma, 1 / sqr(base_sigma)};

    if (method / 10 == 1) {
        // PValue family.
        // stopCnd = (mean + stop_sigmas * sigma < base_mean + stop_sigmas * base_sigma)
        // shipCnd = (mean > (sigma - sigma_mul * base_sigma) * ship_sigmas
        params.stop_mul = 0.0;
        params.sigma_mul = 0.0;
        params.ship_sigmas = 0.6;
        params.stop_sigmas = 2.1;
        if (method == 10) {
            scanf(
                "%lf%lf%lf%lf%lf",
                &params.ship_sigmas, &params.stop_sigmas, &params.sigma_mul,
                &params.stop_mul, &params.max_test_weeks
            );
        } else if(method == 11) {
            // stopCnd = (mean + stop_sigmas * sigma < 0)
            // shipCnd = (mean - ship_sigmas * sigma > 0)
            params.stop_mul = 0.0;
            params.sigma_mul = 0.0;
            scanf("%lf%lf", &params.ship_sigmas, &params.stop_sigmas);
        } else if(method == 12) {
            // stopCnd = (mean + stop_sigmas * sigma < base_mean + stop_sigmas * base_sigma)
            // shipCnd = (mean - ship_sigmas * (sigma - sigma0) > 0)
            // where sigma0 = sigma_mul * base_sigma
            params.stop_mul = 1.0;
            scanf("%lf%lf%lf", &params.ship_sigmas, &params.stop_sigmas, &params.sigma_mul);
        } else if(method == 13) {
            // ship && stop intersect at mean = 0
            params.stop_mul = 1.0;
            scanf("%lf%lf", &params.ship_sigmas, &params.stop_sigmas);
            params.sigma_mul = (base_mean / base_sigma /params.stop_sigmas + 1.0);
        } else if(method == 14) {
            // ship && stop intersect at mean = 0
            params.stop_mul = 1.0;
            double add_s = 0.0;
            scanf("%lf%lf", &params.stop_sigmas, &add_s);
            params.ship_sigmas = params.stop_sigmas + add_s;
            params.sigma_mul = (base_mean / base_sigma / params.stop_sigmas + 1.0);
        } else if(method == 15) {
            // ship && stop intersect at mean = 0
            // and ship_sigmas == stop_sigmas
            params.stop_mul = 1.0;
            scanf("%lf", &params.ship_sigmas);
            params.stop_sigmas = params.ship_sigmas;
            params.sigma_mul = (base_mean / base_sigma + params.stop_sigmas) / params.stop_sigmas;
        }
        result = eval_pvalue(rng, weeks, trials, base_mean, base_sigma, week_sigma, params);
    } else if (method / 10 == 2) {
        // moss-index family
        params.stop_mul = 1.0; // fixed
        params.s = 1.0;
        params.l = 2.0;
        params.ship_sigmas = 0.6;
        params.ksi = 1.75;
        if (method == 25) {
            // symetrical cnds
            scanf("%lf%lf%lf", &params.s, &params.l, &params.ksi);
            result = eval_index_sym(rng, moss_index, weeks, trials, base_mean, base_sigma, week_sigma, params);
        } else {
            if (method == 20) {
                scanf("%lf%lf%lf%lf", &params.s, &params.l, &params.ship_sigmas, &params.ksi);
            } else if (method == 21) {
                scanf("%lf%lf%lf", &params.s, &params.l, &params.ksi);
                params.ship_sigmas = params.s * sqrt(fmax(params.ksi, params.l + log(base_sigma)));
            }
            double base_index_s = moss_index(base_idea, params) / params.s;
            double sigma0 = 0.5 * base_sigma;
            sigma0 = base_index_s / sqrt(fmax(params.ksi, params.l + log(sigma0)));
            sigma0 = base_index_s / sqrt(fmax(params.ksi, params.l + log(sigma0)));
            sigma0 = base_index_s / sqrt(fmax(params.ksi, params.l + log(sigma0)));
            sigma0 = base_index_s / sqrt(fmax(params.ksi, params.l + log(sigma0)));
            params.sigma_mul = sigma0 / base_sigma;
            result = eval_index_pvalue(rng, moss_index, weeks, trials, base_mean, base_sigma, week_sigma, params);
        }
    } else if (method / 10 == 3) {
        // g-index family
        params.stop_mul = 1.0;
        params.ship_mul = 0.75;
        params.ksi = 0.015;
        if (method == 35) {
            // symetrical cnds; identical to the method = 32 with ship_mul = 1.0
            scanf("%lf", &params.s);
            params.ksi = 0.0; // for g_index in the next line
            params.ksi = - sqr(base_sigma * params.s) * g_index(base_idea, params);
            result = eval_index_sym(rng, g_index, weeks, trials, base_mean, base_sigma, week_sigma, params);
        } else if (method == 36) {
            // symetrical cnds;
            params.s = 1.0;
            scanf("%lf", &params.ksi);
            result = eval_index_sym(rng, g_index, weeks, trials, base_mean, base_sigma, week_sigma, params);
        } else {
            if (method == 30) {
                scanf("%lf%lf%lf", &params.s, &params.ship_mul, &params.ksi);
            } else if (method == 31) {
                params.ksi = 0.0;
                scanf("%lf%lf", &params.s, &params.ship_mul);
            } else if (method == 32) {
                scanf("%lf%lf", &params.s, &params.ship_mul);
                params.ksi = 0.0; // for g_index in the next line
                params.ksi = - sqr(base_sigma * params.s) * g_index(base_idea, params);
            }
            result = eval_index(rng, g_index, weeks, trials, base_mean, base_sigma, week_sigma, params);
            // result = eval_index_debug(rng, g_index, weeks, trials, base_mean, base_sigma, week_sigma, params);
        }
     } else {
        fprintf(stderr, "unknown method %d\n", method);
        help();
        return 1;
    }
    double norm_coeff = sqr(week_sigma);
    printf("%.8lg\n%.8g\n", norm_coeff * result.profit, norm_coeff * result.sigma);
    return 0;
}
