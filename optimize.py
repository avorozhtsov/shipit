#!/usr/bin/env python
import os
import re
from scipy.optimize import minimize
import argparse
import math
from skopt import gp_minimize, forest_minimize, gbrt_minimize

from diskcache import FanoutCache
cache = FanoutCache("fn_cache", shards=4, size_limit=10000)


@cache.memoize(typed=True, tag='gauss_mab')
def gauss_mab(method, seed, arms, weeks, trials, mean, sigma, error, x):
    cmd = "echo %d %d %d %d %d %s %s %s %s | ./gauss_mab" % (
        method, seed, arms, weeks, trials,
        mean, sigma, error,
        " ".join(map(str, x))
    )
    res = list(map(float, os.popen(cmd).read().split("\n")[0:2]))
    return res[0], res[1]


@cache.memoize(typed=True, tag='shipit')
def shipit(method, seed, weeks, trials, mean, sigma, error, x):
    cmd = "echo %d %d %d %d %s %s %s %s | ./shipit" % (
        method, seed, weeks, trials,
        mean, sigma, error,
        " ".join(map(str, x))
    )
    print(cmd)
    res_str = os.popen(cmd).read()
    print("%s\n" % (res_str,))
    res = list(map(float, res_str.split("\n")[0:2]))
    return -res[0], res[1]

def xargs_to_dict(fn_name, xargs):
    if fn_name == 'gauss_mab':
        names = ['method', 'seed', 'arms', 'weeks', 'trials', 'mean', 'sigma', 'error', 'x']
        method, seed, arms, weeks, trials, mean, sigma, error, *x = xargs
        x = " ".join(str(v) for v in x)
        return dict(zip(names, [method, seed, arms, weeks, trials, mean, sigma, error, x]))
    elif fn_name == 'shipit':
        names = ['method', 'seed', 'weeks', 'trials', 'mean', 'sigma', 'error', 'x']
        method, seed, weeks, trials, mean, sigma, error, *x = xargs
        x = " ".join(str(v) for v in x)
        return dict(zip(names, [method, seed, weeks, trials, mean, sigma, error, x]))
    else:
        raise ValueError(f"Bad value for fn_name = {fn_name}")


def point_filename(fn_name, prefix, *xargs):
    if fn_name == "shipit":
        method, seed, weeks, trials, mean, sigma, error, *x = xargs
        return "%sm%s_a%0gs%0ge%0g.txt" % (
            prefix, method, -float(mean), float(sigma), float(error)
        )
    elif fn_name == "gauss_mab":
        method, seed, arms, weeks, trials, mean, sigma, error, *x = xargs
        return "%sm%s_p%0gw%0ga%0gs%0ge%0g.txt" % (
            prefix, method, float(arms), float(weeks), float(mean), float(sigma), float(error)
        )
    else:
        raise ValueError(f"Bad value for fn_name = {fn_name}")


def my_optimize(fn, optimize_method, n_calls, dimensions, x0, x0s=None, fn_noise=None):
    if optimize_method == "minimize":
        res = minimize(fn, x0, method='nelder-mead', tol=1e-5, options={'disp': True})
        # res = minimize(fn, x0, method='nelder-mead', options={'xtol': 1e-5, 'disp': True})
    elif optimize_method == "forest_minimize":
        res = forest_minimize(fn, dimensions, x0=x0s, n_calls=n_calls, n_jobs=3, verbose=True)
    elif optimize_method == "gp_minimize":
        res = gp_minimize(fn, dimensions, x0=x0s, n_calls=n_calls, n_jobs=1, noise=fn_noise, verbose=True)
    elif optimize_method == "gbrt_minimize":
        res = gbrt_minimize(fn, dimensions, x0=x0s, n_calls=n_calls, n_jobs=3, verbose=True)
    else:
        raise ValueError(
            "Bad optimize_method = '%s'. Should be one of "
            "[minimize, gp_minimize, forest_minimize, gbrt_minimize]" % (optimize_method,)
        )
    return res


def optimize_mab(
    dryrun,
    prefix, fn, method, seed, arms, weeks, trials,
    mean, sigma, error,
    optimize_method="gp_minimize", n_calls=45
):
    xargs = [method, seed, arms, weeks, trials, mean, sigma, error]
    xargs_len = len(xargs)
    fn_ = lambda x: fn(*xargs, x)[0]
    fn_with_sigma_ = lambda x: fn(*xargs, x)
    point_file = point_filename(fn.__name__, prefix, *xargs)
    read_point_cmd = "cat %s 2>/dev/null" % (point_file,)
    print(read_point_cmd)
    point = [float(s) for s in os.popen(read_point_cmd).read().split()]
    x0 = point[xargs_len:]
    fn_noise = 0.0045 * math.sqrt(10000.0 / float(trials))
    if x0 is None or len(x0) == 0:
        print("File does not exists or corrupted.")
        x0 = None

    if method in [4, 5, 6]:
        x0 = x0 or [1.2, 12.0, 20.0]
        dimensions = [(0.0, 15.0), (0.0, 19.0), (0.0, 28.0)]
        x0s = [
            x0, [1.2, 0.0, 0.0], [5.0, 0.0, 0.0],
            [1.2, 4.0, 12.0], [5.0, 4.0, 12.0], [8.0, 10.0, 20.0]
        ]
    elif method in [1]:
        x0 = x0 or [11.0, 0.4]
        dimensions = [(5.0, 15.0), (0.1, 0.5)]
        x0s = [x0, [10.5, 0.4], [11.0, 0.3], [10.5, 0.4], [11.0, 0.32]]
    elif method in [2]:
        x0 = x0 or [1.0, 0.0]
        dimensions = [(0.4, 1.5), (0.0, 0.2)]
        x0s = [x0, [0.9, 0.0], [0.9, 0.1], [0.8, 0.0], [0.98, 0], [0.98, 0.1]]
    elif method in [3]:
        x0 = x0 or [1.0]
        dimensions = [(0.1, 4)]
        x0s = [x0, [0.3], [0.8], [1.5], [2.8], [3.0], [3.2]]
    elif method in [7]:
        x0 = x0 or [1.135, 1.0, 0.325]
        dimensions = [(0.9, 1.5), (0.9, 1.1), (0.29, 0.67)]
        x0s = [
            x0, [1.15, 1.0, 0.55], [1.15, 1.0, 0.32], [1.13, 1.05, 0.315],
            [1.14, 1.0, 0.32], [1.13, 1.0, 0.315]
        ]
    else:
        raise ValueError(f"Bad value {method} for method ")

    res0, sigma0 = fn_with_sigma_(x0)
    print("Initial x = %s" % x0)
    print("Initial value = %6g +- %6g" % (res0, sigma0))

    res = my_optimize(
        fn=fn_,
        optimize_method=optimize_method,
        n_calls=n_calls,
        dimensions=dimensions,
        x0=x0,
        x0s=x0s,
        fn_noise=fn_noise
    )

    print("Args: %s\n Result: %s" % ((method, seed, weeks, mean, sigma, error), res))

    if not dryrun:
        write_point_cmd = "echo %d %d %d %d %d %s %s %s %s > %s" % (
            method, seed, arms, weeks, trials,
            mean, sigma, error,
            " ".join(map(str, res.x)),
            point_file,
        )
        os.system(write_point_cmd)


def optimize_shipit(
    dryrun,
    prefix, fn, method, seed, weeks, trials,
    mean, sigma, error,
    optimize_method="gp_minimize", n_calls=45
):
    xargs = [method, seed, weeks, trials, mean, sigma, error]
    xargs_len = len(xargs)
    fn_ = lambda x: fn(*xargs, x)[0]
    fn_with_sigma_ = lambda x: fn(*xargs, x)
    # for shipit problem mean < =0, so we use -mean for point_file signature:
    pt_filename = point_filename(fn.__name__, prefix, *xargs)
    read_point_cmd = "cat %s 2>/dev/null" % (pt_filename,)
    print(read_point_cmd)
    point = [float(s) for s in os.popen(read_point_cmd).read().split()]
    if len(point) > 1:
        pt_method, pt_seed, pt_weeks, pt_trials, *_ = point
        if pt_weeks * pt_trials > weeks * trials:
            print("Stored point has greater weeks * trials. Break")
            return

    x0 = point[xargs_len:]

    if x0 is None or len(x0) == 0:
        print("File does not exists or corrupted.")
        x0 = None

    max_test_weeks = max(4, int(0.5 + 1.7 * (0.73 + float(error)) * (0.73 + float(error)) - 10))

    if method == 20:
        # Moss Index
        # S, L, ship_sigmas, ksi
        x0 = x0 or [1.55, 11.5, 0.66, 1.0, 1.6]
        x0 = x0[0:5]
        dimensions = [(0.5, 1.8), (8.0, 15.0), (0.5, 3.0), (0.0, 3.2)]
        x0s = [
            x0,
            [1.5, 12.0, 2.1, 0.05], [1.45, 12.4, 1.62,  1.0],
            [1.2, 11.7, 1.9, 0.03], [0.76, 12.0, 1.8, 2.06],
            [0.8, 12.5, 1.933, 1.9], [1.3, 10.5, 1.79, 1.73],
            [0.57, 13,  1.57, 2.016],  [0.584, 11.3, 1.818, 1.931],
            [0.55, 13.01, 2.23, 2.116]
        ]
    elif method == 21:
        # Moss Index
        # [S, L, ksi]
        x0 = x0 or [1.55, 11.5, 1.6]
        x0 = x0[0:3]
        dimensions = [(0.5, 1.8), (3.0, 15.0), (1.1, 3.2)]

        def p1_fn(x):
            return (-1 + math.sqrt(1 + 4 * x * x)) / (- 2 * x)

        mean_f = float(mean)
        x0s = [
            x0,
            [p1_fn(mean_f), 12.0, 1.5],
            [p1_fn(mean_f), 11.0, 2.0],
            [p1_fn(mean_f), 13.0, 2.5],
            [1.5, 12.0, 0.05], [0.8, 12.5, 1.9],
            [1.3, 10.5, 1.73], [0.57, 13,  2.016],
            [0.584, 11.3, 1.931], [0.55, 13.01, 2.116]
        ]
    elif method == 12:
        # PValue Index
        # [ship_sigmas, stop_sigmas, sigma_mul]
        mean_r = - float(mean) / float(sigma)
        x0 = x0 or [0.53, 1.8, 0.4]
        dimensions = [(0.3, 0.8), (1.4, 3.0), (-0.3, 0.98)]
        sigma_mul_fn = lambda b_mean, b_sigma, p_stop : (
            (float(b_mean) / float(p_stop) / float(b_sigma) + 1)
        );

        x0s = [
            x0,
            [2.07, 2.07, 0.517],
            [1.5, 1.5, 0.33],
            [2.8, 2.8, 0.14],
            [2.6, 2.6, 0.42],
            [1.74, 1.74, 0.6],
            [1.42, 1.42, 0.36],
            [1.42, 1.42, 0.6],
            [0.644, 0.644, 0.6895],
            [x0[0], x0[1], sigma0_fn(mean, sigma, x0[1])],
            [0.5 * (x0[0] + x0[1]), 0.5 * (x0[0] + x0[1]), sigma0_fn(mean, sigma, x0[1])],
            [x0[0], x0[0], sigma0_fn(mean, sigma, x0[0])],
            [x0[0], x0[0], x0[2]],
            [x0[0] * 1.02, x0[1] * 1.023, x0[2]],
            [x0[0] * 1.02, x0[1] * 1.0, x0[2] * 0.97],
            [x0[0] * 0.98, x0[1] * 1.0, x0[2] * 0.97],
            [x0[0] * 1.0, x0[1] * 0.985, x0[2] * 0.97],
            [x0[0] * 0.985, x0[1] * 1.01, x0[2] * 0.99],
            [x0[0] * 1.03, x0[1] * 0.99, x0[2]],
            [x0[0] * 0.97, x0[1], x0[2] - 0.05],
            [x0[0], 0.8 * x0[1], 0.6 * x0[2]],
        ]
    elif method == 14:
        # PValue Index
        # [ship_sigmas, stop_sigmas, sigma_mul]
        x0 = x0 or [1.5, 0.0]
        dimensions = [(0.3, 2.5), (-0.1, 0.1)]
        mean_r = - float(mean) / float(sigma)
        x0s = [
            x0,
            [x0[0] * 0.7, x0[1]],
            [x0[0] * 0.8, x0[1]],
            [x0[0] * 0.98, x0[1]],
            [x0[0] * 0.99, x0[1]],
            [x0[0] * 1.01, x0[1]],
            [x0[0] * 1.015, x0[1]],
            [x0[0] * 0.988, x0[1] - 0.05],
            [x0[0] * 1.013, x0[1] - 0.0513],
            [x0[0] * 0.993, x0[1] + 0.0512],
            [x0[0] * 1.0097, x0[1] + 0.0511],
            [0.5, 0.0],
            [0.6, 0.0],
            [0.65, 0.0],
            [0.7, 0.0],
            [0.75, 0.0],
            [0.9, 0.0],
            [1.2, 0.0],
            [1.5, 0.0],
            [3.0, 0.0],
            [2.2 * math.sqrt(mean_r), 0.0],
            [2.1 * math.sqrt(mean_r), 0.0],
            [2.05 * math.sqrt(mean_r), 0.0],
            [2.0 * math.sqrt(mean_r), 0.0],
        ]

    elif method == 30:
        # gValue
        # [S, shipMul, ksi]
        x0 = x0 or [0.70, 1.0, 0.015]
        dimensions = [(0.1, 1.5), (0.4, 2.3), (-0.15, 0.15)]
        x0s = [
            x0,
            [x0[0] * 1.02, x0[1] * 1.02, x0[2] * 1.015],
            [x0[0] * 0.99, x0[1] * 1.01, x0[2] * 1.02],
            [x0[0] * 1.02, x0[1] * 1.0, x0[2] * 0.97],
            [x0[0] * 0.98, x0[1] * 1.0, x0[2] * 0.97],
            [x0[0] * 1.0, x0[1] * 0.98, x0[2] * 0.97],
            [x0[0] * 0.985, x0[1] * 0.98, x0[2]],
            [x0[0] * 0.985, x0[1] * 1.01, x0[2] * 0.99],
            [x0[0] * 1.02, x0[1] * 1.02, -x0[2]],
            [2.15, 0.65, -0.07],
            [2.14, 0.64, -0.09],
            [2.145, 0.655, -0.11],
            [2.0, 0.5953, -0.001],
            [2.22, 0.5492,  -0.15],
            [1.742, 0.4998, -0.022],
            [0.898, 0.286789, -0.13],
            [0.78, 0.9999, -0.022],
            [ 0.6667, 0.107967, 0.02]
        ]
    elif method == 32:
        # gValue
        # [S, shipMul]
        x0 = x0 or [1.0, 1.0]
        dimensions = [(0.1, 1.5), (0.4, 2.3)]

        x0s = [
            x0,
            [x0[0], x0[1] * 1.02],
            [x0[0] * 1.02, x0[1]],
            [x0[0], x0[1] * 0.98],
            [x0[0], x0[1] * 1.055],
            [x0[0] * 0.985, x0[1]],
            [x0[0], x0[1] * 0.991],
            [x0[0] * 1.015, x0[1] * 1.0111],
            [1.8, 1.0],
            [1.7, 1.0],
            [1.4, 1.0],
            [1.1, 1.0],
            [0.8, 1.0],
            [0.8, 0.95],
            [0.7, 0.98],
            [0.60, 1.2],
            [0.55, 0.99],
            [0.51, 0.83],
        ]
    else:
        raise ValueError(f"Bad value {method} for method ")

    res0, sigma0 = fn_with_sigma_(x0)

    print("Initial x = %s" % x0)
    print("Initial value = %g +- %g" % (res0, sigma0))
    fn_noise = max(0.00001, sigma0)

    res = my_optimize(
        fn=fn_,
        optimize_method=optimize_method,
        n_calls=n_calls,
        dimensions=dimensions,
        x0=x0,
        x0s=x0s,
        fn_noise=fn_noise
    )

    print("Args: %s\n Result: %s" % ((method, seed, weeks, trials, mean, sigma, error), res))
    if not dryrun:
        write_point_cmd = "echo %d %d %d %d %s %s %s %s > %s" % (
            method, seed, weeks, trials,
            mean, sigma, error,
            " ".join(map(str, res.x)),
            pt_filename,
        )
        os.system(write_point_cmd)


def results2points(dryrun, filename, fn_name, prefix):
    best = {}
    with open(filename) as file:
        for line in file:
            line = line.rstrip("\r\n")
            # print(f"line = {line}")
            if "result" in line:  # first header line
                continue
            result_and_args = re.split(r"[\r\n\t ]", line)
            result, result_sigma, *xargs = result_and_args
            filename = point_filename(fn_name, prefix, *xargs)
            info = "_".join(str(x) for x in xargs[2:4])
            # info = ""
            print(f"Read {filename}@{info}: {xargs_to_dict(fn_name, xargs)}: result={result}, result_sigma={result_sigma}")
            if best.get(filename, (-1.0, None, None))[0] <= float(result):
                best[filename] = (float(result), float(result_sigma), xargs)
    for filename, (result, result_sigma, xargs) in best.items():
        print(f"Write {filename}: {xargs_to_dict(fn_name, xargs)}: result={result}, result_sigma={result_sigma}")
        if not dryrun:
            with open(filename, "w+") as file:
                file.write(' '.join(xargs) + "\n")
            with open(filename + ".result", "w+") as file:
                file.write("\n".join([str(result), str(result_sigma)]) + "\n")


def generate_cmd(dryrun, filename, fn_name, prefix, weeks, trials):
    with open(filename) as file:
        for line in file:
            line = line.rstrip("\r\n")
            # print(f"line = {line}")
            if "result" in line:  # first header line
                continue
            result_and_args = re.split(r"[\r\n\t ]", line)
            result, result_sigma, *xargs = result_and_args
            # filename = point_filename(fn_name, prefix, *xargs)
            if fn_name == "shipit":
                method, seed, pt_weeks, pt_trials, mean, sigma, error, *_ = xargs
                skip = "# SKIP " if float(pt_weeks) * float(pt_trials) > weeks * trials else ''
                print(
                    f"{skip}python ./optimize.py --prefix {prefix} --fn {fn_name} --weeks {weeks} --method {method} "
                    f"--trials {trials} --mean {mean} --sigma {sigma} --seed {seed} --error {error}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize shipit')
    parser.add_argument("--command", type=str, default="optimize", help="optimize|results2points|generate_cmd")

    # for these arguments default depends on fn; set default=None
    parser.add_argument("--src",    type=str, default=None,    help="input filename (for command=results2points)")
    parser.add_argument("--prefix", type=str, default=None,    help="path prefix to store point data")
    parser.add_argument("--dryrun", action='store_true', help="dry run (no changes in disk)")
    parser.add_argument("--fn",     type=str, default="shipit", help="fn name: shipit or gauss_mab")
    parser.add_argument("--method", type=int, default=10,      help="method of shipping: 1, 2, ...")
    parser.add_argument("--seed",   type=int, default=1,       help="seed for random")
    # --arms only for mab:
    parser.add_argument("--arms",   type=int, default=10,      help="number of arms (for fn=gauss_mab)")
    parser.add_argument("--weeks",  type=int, default=2000000, help="number of weeks")
    parser.add_argument("--trials", type=int, default=25,      help="number of trials")
    parser.add_argument("--mean",   type=str, default="-1",    help="mean value for idea ~ N(mean, sigma)")
    parser.add_argument("--sigma",  type=str, default="1",     help="sigma value for idea ~ N(mean, sigma)")
    parser.add_argument("--error",  type=str, default="8",     help="error of profit measurement for 1 week")
    parser.add_argument("--optimize_method",  type=str, default="minimize", help="minimize, forest_optimize, or gp_minimize")
    parser.add_argument("--n_calls", type=int, default=45,     help="number of calls during optimization")

    args = parser.parse_args()

    if args.prefix is None:
        if args.fn == "shipit":
            args.prefix = "shipit_points/pt_"
        else:
            args.prefix = "mab_points/pt_"

    if "/" in args.prefix and not args.dryrun:
        os.makedirs(args.prefix.rsplit('/', 1)[0], exist_ok=True)

    if args.src is None:
        if args.fn == "shipit":
            args.src = "shipit_results.txt"
        else:
            args.src = "gauss_mab_results.txt"

    if args.command == "optimize":
        if args.fn == "shipit":
            optimize_shipit(
                args.dryrun,
                args.prefix,
                fn=shipit, optimize_method=args.optimize_method, n_calls=args.n_calls,
                method=args.method, seed=args.seed, weeks=args.weeks, trials=args.trials,
                mean=args.mean, sigma=args.sigma, error=args.error,
            )
        elif args.fn == "gauss_mab":
            optimize_mab(
                args.dryrun,
                args.prefix,
                fn=gauss_mab, optimize_method=args.optimize_method, n_calls=args.n_calls,
                method=args.method, seed=args.seed, arms=args.arms, weeks=args.weeks, trials=args.trials,
                mean=args.mean, sigma=args.sigma, error=args.error,
            )
    elif args.command == "results2points":
        results2points(args.dryrun, args.src, args.fn, args.prefix)
    elif args.command == "generate_cmd":
        generate_cmd(args.dryrun, args.src, args.fn, args.prefix, args.weeks, args.trials)
    else:
        raise ValueError(f"Bad command {args.command}")
