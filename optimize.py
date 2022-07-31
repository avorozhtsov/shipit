import os
import numpy as np
from scipy.optimize import minimize
import argparse
import math
from skopt import gp_minimize, forest_minimize, gbrt_minimize, dummy_minimize

def gauss_mab(method, seed, arms, weeks, trials, a, sigma, error, x):
    cmd = "echo %d %d %d %d %d %s %s %s %s | ./gauss_mab" % (
        method, seed, arms, weeks, trials,
        a, sigma, error,
        " ".join(map(str, x))
    )
    res = map(float, os.popen(cmd).read().split("\n")[0:2])
    return (res[0], res[1])

def shippit(method, seed, weeks, trials, a, sigma, error, x):
    cmd = "echo %d %d %d %d %s %s %s %s | ./shipit" % (
        method, seed, weeks, trials,
        a, sigma, error,
        " ".join(map(str, x))
    )
    print(cmd)
    res_str = os.popen(cmd).read()
    print("%s\n" % (res_str,))
    res = list(map(float, res_str.split("\n")[0:2]))
    return (-res[0], res[1])


def optimize_mab(
    prefix, fn, method, seed, arms, weeks, trials, 
    mean, sigma, error, 
    optimize_method="gp_minimize", n_calls=45
):
    fn_ = lambda x: fn(method, seed, arms, weeks, trials, mean, sigma, error, x)[0]
    fn_with_sigma_ = lambda x: fn(method, seed, arms, weeks, trials, mean, sigma, error, x)
    point_file = "%s_%s_m%s_p%0gw%0ga%0gs%0ge%0g.txt" % (prefix, fn.__name__, method, float(arms), float(weeks), float(mean), float(sigma), float(error))
    read_point_cmd = "cat %s 2>/dev/null" % (point_file,)
    print(read_point_cmd)
    point = [float(s) for s in os.popen(read_point_cmd).read().split()]
    x0 = point[8:]
    fn_noise = 0.0045 * math.sqrt(10000.0 / float(trials))
    if x0 is None or len(x0) == 0:
        print("File does not exists or corrupted.")
        x0 = None

    if method in [4, 5, 6]:
        x0 = x0 or [1.2, 12.0, 20.0]
        dimensions = [(0.0, 15.0), (0.0, 19.0), (0.0, 28.0)]
        x0s = [x0, [1.2, 0.0, 0.0], [5.0, 0.0, 0.0], [1.2, 4.0, 12.0], [5.0, 4.0, 12.0], [8.0, 10.0, 20.0]]
    if method in [1]:
        x0 = x0 or [11.0, 0.4]
        dimensions = [(5.0, 15.0), (0.1, 0.5)]
        x0s = [x0, [10.5, 0.4], [11.0, 0.3], [10.5, 0.4], [11.0, 0.32]]
    if method in [2]:
        x0 = x0 or [1.0, 0.0]
        dimensions = [(0.4, 1.5), (0.0, 0.2)]
        x0s = [x0, [0.9, 0.0], [0.9, 0.1], [0.8, 0.0], [0.98, 0], [0.98, 0.1]]
    if method in [3]:
        x0 = x0 or [1.0]
        dimensions = [(0.1, 4)]
        x0s = [x0, [0.3], [0.8], [1.5], [2.8], [3.0], [3.2]]
    if method in [7]:
        x0 = x0 or [1.135, 1.0, 0.325]
        dimensions = [(0.9, 1.5), (0.9, 1.1), (0.29, 0.67)]
        x0s = [x0, [1.15, 1.0, 0.55], [1.15, 1.0, 0.32], [1.13, 1.05, 0.315], [1.14, 1.0, 0.32], [1.13, 1.0, 0.315]]

    res0, sigma0 = fn_with_sigma_(x0)
    print("Initial x = %s" % x0)
    print("Initial value = %6g +- %6g" % (res0, sigma0))

    if optimize_method == "minimize":
        res = minimize(fn_, x0, method='nelder-mead',  options={'xtol': 1e-5, 'disp': True})
    elif optimize_method == "forest_minimize":
        res = forest_minimize(fn_, dimensions, x0=x0s, n_calls=n_calls, n_jobs=3, verbose=True)
    elif optimize_method == "gp_minimize":
        res = gp_minimize(fn_, dimensions, x0=x0s, n_calls=n_calls, n_jobs=1, noise=fn_noise, verbose=True)
    elif optimize_method == "gbrt_minimize":
        res = gbrt_minimize(fn_, dimensions, x0=x0s, n_calls=n_calls, n_jobs=3, verbose=True)
    else:
        print("Bad optimize_method = '%s'. Should be one of [minimize, gp_minimize, forest_minimize, gbrt_minimize]\n" % (optimize_method,))

    print("Args: %s\n Result: %s" % ((method, seed, weeks, a, sigma, error), res))
    write_point_cmd = "echo %d %d %d %d %d %s %s %s %s > %s" % (
        method, seed, arms, weeks, trials,
        a, sigma, error,
        " ".join(map(str, res.x)),
        point_file,
    )
    os.system(write_point_cmd)


def optimize_shipit(
    prefix, fn, method, seed, weeks, trials, 
    mean, sigma, error, 
    optimize_method="gp_minimize", n_calls=45
):
    fn_ = lambda x: fn(method, seed, weeks, trials, mean, sigma, error, x)[0]
    fn_with_sigma_ = lambda x: fn(method, seed, weeks, trials, mean, sigma, error, x)
    # for shipit problem mean < =0, so we use -mean for point_file signature:
    point_file = "%s_%s_m%s_a%0gs%0ge%0g.txt" % (prefix, fn.__name__, method, -float(mean), float(sigma), float(error))
    read_point_cmd = "cat %s 2>/dev/null" % (point_file,)
    print(read_point_cmd)
    point = [float(s) for s in os.popen(read_point_cmd).read().split()]
    x0 = point[7:]

    if x0 is None or len(x0) == 0:
        print("File does not exists or corrupted.")
        x0 = None

    max_test_weeks = max(4, 0.5 + 1.7 * (0.73 + float(error)) * (0.73 + float(error)) - 10)

    if method in [1, 2]:
        # Moss Index
        x0 = x0 or [1.55, 11.5, 0.66, 1.0, 1.6]
        x0 = x0[0:5]
        dimensions = [(0.8, 2.0), (0.1, 15.0), (0.2, 0.9), (0.2, 2.0), (0.0, 2.0)]
        x0s = [x0, [1.5, 12.0, 0.6, 1.0, 0.05], [1.45, 12.4, 0.62, 0.95, 1.0], [1.2, 11.7, 0.55, 1.0, 0.03]]
    elif method == 3:
        # pValue without max_test_weeks
        x0 = x0 or [0.6, 0.95]
        dimensions = [(0.1, 1.0), (0.1, 1.0)]
        x0s = [x0, [0.6, 0.9], [0.5, 0.92], [0.7, 0.95], [0.76, 0.98]]
    elif method == 4:
        # pValue with max_test_weeks
        x0 = x0 or [0.6, 0.95, max_test_weeks]
        dimensions = [(0.1, 1.0), (0.1, 1.0), (1.0, 10000)]
        x0s = [x0, [0.6, 0.9, 1000], [0.5, 0.92, 100], [0.4, 0.8, 100], [0.6, 0.97, 2000], [0.7, 0.99, 2200]]
    elif method == 5:
        # pValue with S
        x0 = x0 or [0.53, 1.8]
        if len(x0) == 3:
            x0 = [x0[0], x0[1]]
        dimensions = [(0.3, 0.8), (1.4, 2.2)]
        x0s = [x0, [0.54, 1.6], [0.54, 1.7], [0.58, 1.6], [0.58, 1.7]]
    elif method == 6:
        # gValue with one params = shipMul
        x0 = x0 or [0.6, 1.6]
        if len(x0) == 1:
            x0 = [x0[0], 1.0]
        dimensions = [(0.4, 0.9), (0.5, 1.8)]
        x0s = [x0, [x0[0], 1.6], [0.6, 1.1], [0.6, 1.6], [0.70, 1.1], [0.70, 1.6]]
    elif method == 7:
        # gValue with 3 params = shipMul, stopMul, ksi
        x0 = x0 or [0.70, 1.0, -0.02]
        dimensions = [(0.4, 0.9), (0.8, 1.3), (-0.2, 0.2)]
        x0s = [ x0 ]
    else: # ???
        x0 = [0.5, 0.5, 0.5, 0.5]

    res0, sigma0 = fn_with_sigma_(x0)

    print("Initial x = %s" % x0)
    print("Initial value = %g +- %g" % (res0, sigma0))
    fn_noise = max(0.00001, sigma0)

    if optimize_method == "minimize":
        res = minimize(fn_, x0, method='nelder-mead',  options={'xtol': 1e-5, 'disp': True})
    elif optimize_method == "forest_minimize":
        res = forest_minimize(fn_, dimensions, x0=x0s, n_calls=n_calls, n_jobs=3, verbose=True)
    elif optimize_method == "gp_minimize":
        res = gp_minimize(fn_, dimensions, x0=x0s, n_calls=n_calls, n_jobs=1, noise=fn_noise, verbose=True)
    elif optimize_method == "gbrt_minimize":
        res = gbrt_minimize(fn_, dimensions, x0=x0s, n_calls=n_calls, n_jobs=3, verbose=True)
    else:
        print("Bad optimize_method = '%s'. Should be one of [minimize, gp_minimize, forest_minimize, gbrt_minimize]\n" % (optimize_method,))

    print("Args: %s\n Result: %s" % ((method, seed, weeks, trials, mean, sigma, error), res))
    write_point_cmd = "echo %d %d %d %d %s %s %s %s > %s" % (
        method, seed, weeks, trials,
        mean, sigma, error,
        " ".join(map(str, res.x)),
        point_file,
    )
    os.system(write_point_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize shipit')
    parser.add_argument("--prefix", type=str, default=None, help="path prefix to store point data")
    parser.add_argument("--fn",     type=str, default="shipit",  help="fn name: shipit or gauss_mab")
    parser.add_argument("--method", type=int, default=2,       help="method of shipping: 1, 2, ...")
    parser.add_argument("--seed",   type=int, default=1,       help="seed for random")
    parser.add_argument("--arms",   type=int, default=10,      help="number of arms (for mab only)")   # only for mab
    parser.add_argument("--weeks",  type=int, default=2000000, help="number of weeks")
    parser.add_argument("--trials", type=int, default=25,      help="number of trials")
    parser.add_argument("--mean",   type=str, default="-1",    help="mean value for idea ~ N(mean, sigma)")
    parser.add_argument("--sigma",  type=str, default="1",     help="sigma value for idea ~ N(mean, sigma)")
    parser.add_argument("--error",  type=str, default="8" ,    help="error of profit measurement for 1 week")
    parser.add_argument("--optimize_method",  type=str, default="minimize", help="method: minimize, forest_optimize, or gp_minimize")
    parser.add_argument("--n_calls", type=int, default=45,     help="number of calls during optimization")

    args = parser.parse_args()

    if args.prefix is None:
        if args.fn == "shipit":
            args.prefix = "shipit_points/point"
        else:
            args.prefix = "mab_points/point"

    os.makedirs(args.prefix.rsplit('/', 1)[0], exist_ok=True)


    if args.fn == "shipit":
        optimize_shipit(
            args.prefix,
            fn=shippit, optimize_method=args.optimize_method, n_calls=args.n_calls,
            method=args.method, seed=args.seed, weeks=args.weeks, trials=args.trials,
            mean=args.mean, sigma=args.sigma, error=args.error,
        )
    elif args.fn == "gauss_mab":
        optimize_mab(
            args.prefix,
            fn=gauss_mab, optimize_method=args.optimize_method, n_calls=args.n_calls,
            method=args.method, seed=args.seed, arms=args.arms, weeks=args.weeks, trials=args.trials,
            mean=args.mean, sigma=args.sigma, error=args.error,
        )

