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
        x0s = [x0, [1.2, 0.0, 0.0], [5.0, 0.0, 0.0], [1.2, 4.0, 12.0], [5.0, 4.0, 12.0], [8.0, 10.0, 20.0]]
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
        x0s = [x0, [1.15, 1.0, 0.55], [1.15, 1.0, 0.32], [1.13, 1.05, 0.315], [1.14, 1.0, 0.32], [1.13, 1.0, 0.315]]
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

    if method == 1:
        # Moss Index
        # S, L, ship_mul, stop_mul, ksi (optional)
        x0 = x0 or [1.55, 11.5, 0.66, 1.0, 1.6]
        x0 = x0[0:5]
        dimensions = [(0.5, 1.8), (8.0, 15.0), (0.4, 0.9), (0.99, 1.01), (0.0, 3.2)]
        x0s = [
            x0,
            [1.5, 12.0, 0.6, 1.0, 0.05], [1.45, 12.4, 0.62, 0.9991, 1.0],
            [1.2, 11.7, 0.55, 1.0, 0.03], [0.76, 12.0, 0.8, 0.999, 2.06],
            [0.8, 12.5, 0.7, 0.993, 1.9], [1.3, 10.5, 0.79, 1.001, 1.73],
            [0.57, 13, 0.67, 1.0, 2.016],  [0.584, 11.3, 0.818, 1.0001, 1.931],
            [0.55, 13.01, 0.6322, 1.0, 2.116]
        ]
    elif method == 2:
        # Moss Index
        # S, L, ksi (optional)
        x0 = x0 or [1.55, 11.5, 1.6]
        x0 = x0[0:3]
        dimensions = [(0.5, 1.8), (8.0, 15.0), (0.0, 3.2)]

        def p1_fn(x):
            return (-1 + math.sqrt(1 + 4 * x * x)) / (- 2 * x)

        x0s = [
            x0,
            [p1_fn(float(mean)), 12.0, 1.5],
            [p1_fn(float(mean)), 11.0, 2.0],
            [p1_fn(float(mean)), 13.0, 2.5],
            [1.5, 12.0, 0.05], [0.8, 12.5, 1.9],
            [1.3, 10.5, 1.73], [0.57, 13,  2.016],
            [0.584, 11.3, 1.931], [0.55, 13.01, 2.116]
        ]
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
        mean_r = - float(mean) / float(sigma)
        x0 = x0 or [0.53, 1.8]
        if len(x0) == 3:
            x0 = [x0[0], x0[1]]
        dimensions = [(0.3, 0.8), (1.4, 2.2)]
        x0s = [
            x0,
            [x0[0], x0[1] * 1.02],
            [x0[0] * 1.02, x0[1]],
            [x0[0], x0[1] * 0.98],
            [x0[0] * 0.985, x0[1]],
            [x0[0] * 0.985, x0[1] * 0.991],
            [x0[0] * 1.015, x0[1] * 1.0111],
            [0.54, 1.6], [0.54, 1.7], [0.58, 1.6], [0.58, 1.7],
            [2.1 * math.sqrt(mean_r), 1.5],
            [2.1 * math.sqrt(mean_r) * 1.02, 1.0],
            [2.1 * math.sqrt(mean_r) * 1.002, 1.55],
            [2.1 * math.sqrt(mean_r) * 0.99, x0[1]],
            [2.1 * math.sqrt(mean_r) * 1.04, x0[1] * 1.015],
            [2.1 * math.sqrt(mean_r) * 0.977, x0[1] * 0.988],
        ]
    elif method == 51:
        mean_r = - float(mean) / float(sigma)
        x0 = x0 or [0.53, 1.8, 0.05]
        dimensions = [(0.3, 0.8), (1.4, 3.0), (-0.3, 5.0)]
        sigma0_fn = lambda b_mean, b_sigma, p_stop : (float(b_mean) + float(p_stop) * float(b_sigma)) / float(p_stop);

        x0s = [
            x0,
            [2.07, 2.07, 0.517],
            [1.5, 1.5, 0.33],
            [2.8, 2.8, 0.14],
            [2.6, 2.6, 0.42],
            [1.74, 1.74, 0.6],
            [1.42, 1.42, 0.36],
            [1.42, 1.42, 1.3],
            [0.644, 0.644, 0.6895],
            [0.93, 0.93, 3.0],
            [x0[0], x0[1], sigma0_fn(mean, sigma, x0[1])],
            [0.5 * (x0[0] + x0[1]), 0.5 * (x0[0] + x0[1]), sigma0_fn(mean, sigma, x0[1])],
            [x0[0], x0[0], sigma0_fn(mean, sigma, x0[0])],
            [x0[0], x0[0], x0[2]],
            [x0[0] * 1.02, x0[1] * 1.023, x0[2] * 1.015],
            [x0[0] * 0.99, x0[1] * 1.011, x0[2] * 1.02],
            [x0[0] * 1.02, x0[1] * 1.0, x0[2] * 0.97],
            [x0[0] * 0.98, x0[1] * 1.0, x0[2] * 0.97],
            [x0[0] * 1.0, x0[1] * 0.985, x0[2] * 0.97],
            [x0[0] * 0.985, x0[1] * 0.99, x0[2]],
            [x0[0] * 0.985, x0[1] * 1.01, x0[2] * 0.99],
            [x0[0] * 1.03, x0[1] * 0.99, x0[2] + 0.04],
            [x0[0] * 1.04, x0[1], x0[2] + 0.1],
            [x0[0] * 0.97, x0[1], x0[2] - 0.081],
            [x0[0] * 0.8, x0[1], x0[2] - 0.11],
            [x0[0] * 0.7, x0[1], x0[2] - 0.12],
            [x0[0], x0[1], 1.3 * x0[2]],
            [x0[0], x0[1], 2 * x0[2]],
            [x0[0], x0[1], 0.6 * x0[2]],
            [2.1 * math.sqrt(mean_r), 1.5, x0[2]],
            [2.1 * math.sqrt(mean_r) * 1.02, 1.0, x0[2]],
            [2.1 * math.sqrt(mean_r) * 1.02, 1.0, 0.2],
            [2.1 * math.sqrt(mean_r) * 0.977, x0[1], 0.2],
            [2.1 * math.sqrt(mean_r) * 0.977, x0[1] * 1.015, 0.2],
            [2.1 * math.sqrt(mean_r) * 0.977, x0[1] * 1.015, 0.6],
            [2.1 * math.sqrt(mean_r) * 0.977, x0[1] * 0.988, x0[2]],
            [0.74, 1.6, -0.1], [0.9, 1.7, -0.2], [0.66, 1.6, -0.05], [1.1, 1.9, -0.3]
        ]
    elif method == 52:
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
    elif method == 6:
        # gValue with one params = shipMul
        x0 = x0 or [0.6, 1.6]
        if len(x0) == 1:
            x0 = [x0[0], 1.0]
        dimensions = [(0.1, 0.9), (0.5, 2.3)]

        x0s = [
            x0,
            [x0[0], x0[1] * 1.02],
            [x0[0] * 1.02, x0[1]],
            [x0[0], x0[1] * 0.98],
            [x0[0] * 0.985, x0[1]],
            [x0[0] * 0.985, x0[1] * 0.991],
            [x0[0] * 1.015, x0[1] * 1.0111],
            [0.6526821085440682, 2.1868917429211416], [0.6303031153970138, 2.104459171320075],
            [0.5988471312356368, 1.9022562284689957], [0.5674642609458018, 1.9728693373756423],
            [0.5531171201640925, 1.9180324695510407], [0.5569467530808754, 1.895735139396715],
            [0.5629, 1.9410868165285864], [0.5309618365251755, 1.7230951098746932],
            [0.49988559005855226, 1.742207664857431], [0.49623357156464115, 1.596026529488439],
            [0.4268137144739725, 1.3473203893983707], [0.23478749790213413, 0.8534030788530234],
            [0.2867894304129259, 0.898966], [0.25711248890352467, 0.5868342211811566],
            [0.1079678469936739, 0.6667052289743889],
        ]

    elif method == 7:
        # gValue with 3 params = shipMul, stopMul, ksi
        x0 = x0 or [0.70, 1.0, 0.015]
        dimensions = [(0.1, 1.2), (0.5, 2.3), (-0.15, 0.15)]
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
            [0.65, 2.15, -0.07], [0.64, 2.14, -0.09],
            [0.655, 2.14, -0.11], [0.64, 2.165, -0.095],
            [0.25, 0.65, -0.02], [0.6482, 2.34, -0.013],
            [0.6482, 2.34, -0.013], [0.62, 2.15, 0.011],
            [0.5954, 2.0, -0.001], [0.5492, 2.22,  -0.15],
            [0.53382, 1.973, 0.0109],  [0.54, 1.95, 0.012],
            [0.5492, 2.22,  -0.15], [0.60, 1.67, 0.001],
            [0.286789, 0.898, -0.13], [0.4268135, 1.34732, 0.033],
            [0.4998, 1.742, -0.022], [0.107967, 0.6667, -0.1],
            [0.12, 0.666, -0.02], [0.107967, 0.6667, 0.02],
            [0.9952, 0.7745, -0.0217225],
            [0.9999, 0.78, -0.022],
        ]
    elif method == 71:
        # gValue with one params = shipMul
        x0 = x0 or [1.0, 1.0]
        if len(x0) == 1:
            x0 = [x0[0], 1.0]
        dimensions = [(0.7, 1.5), (0.4, 2.0)]

        x0s = [
            x0,
            [x0[0], x0[1] * 1.02],
            [x0[0] * 1.02, x0[1]],
            [x0[0], x0[1] * 0.98],
            [x0[0], x0[1] * 1.055],
            [x0[0] * 0.985, x0[1]],
            [x0[0], x0[1] * 0.991],
            [x0[0] * 1.015, x0[1] * 1.0111],
            [1.0, 1.8],
            [1.0, 1.7],
            [1.0, 1.4],
            [1.0, 1.1],
            [1.0, 1.021],
            [1.0, 1.0],
            [1.0, 0.983],
            [1.0, 0.9],
            [1.0, 0.8],
            [1.0, 0.7],
            [1.0, 0.6],
            [1.0, 0.5],
            [1.0, 0.48],
            [1.0, 0.43],
            [0.98, 0.9],
            [0.95, 0.80],
            [0.98, 0.70],
            [1.2, 0.60],
            [0.92, 0.55],
            [0.83, 0.51],
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
    parser.add_argument("--src", type=str, default=None, help="input filename (for command=results2points)")
    parser.add_argument("--prefix", type=str, default=None, help="path prefix to store point data")
    parser.add_argument("--dryrun", action='store_true', help="dry run (no changes in disk)")
    parser.add_argument("--fn",     type=str, default="shipit",  help="fn name: shipit or gauss_mab")
    parser.add_argument("--method", type=int, default=2,       help="method of shipping: 1, 2, ...")
    parser.add_argument("--seed",   type=int, default=1,       help="seed for random")
    parser.add_argument("--arms",   type=int, default=10,      help="number of arms (for fn=gauss_mab)")   # only for mab
    parser.add_argument("--weeks",  type=int, default=2000000, help="number of weeks")
    parser.add_argument("--trials", type=int, default=25,      help="number of trials")
    parser.add_argument("--mean",   type=str, default="-1",    help="mean value for idea ~ N(mean, sigma)")
    parser.add_argument("--sigma",  type=str, default="1",     help="sigma value for idea ~ N(mean, sigma)")
    parser.add_argument("--error",  type=str, default="8",    help="error of profit measurement for 1 week")
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
