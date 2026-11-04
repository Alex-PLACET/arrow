# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Run the Numba benchmarks and print one table, optionally as JSON."""

import argparse
import cProfile
import json
import pathlib
import pstats
import sys
import time

# python/benchmarks/<this file> -> the checkout root, so the runner works
# from any working directory; replacing sys.path[0] also keeps this
# directory (which contains numba.py) from shadowing the numba package
sys.path[0] = str(pathlib.Path(__file__).resolve().parents[2] / "python")

from benchmarks import numba as per_type
from benchmarks import numba_primitives as prims

N = 1_000_000


def _best(cls, kind, param, repeats):
    harness = cls()
    harness.setup(param)
    fn = getattr(harness, f"time_{kind}")
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(param)
        times.append(time.perf_counter() - start)
    return min(times)


def _primitives(repeats):
    rows = {}
    for cls, kind in ((prims.ReadPrimitives, "read"),
                      (prims.BuildPrimitives, "build"),
                      (prims.CallOverhead, "call")):
        for param in cls.params[0]:
            rows[f"{kind} {param}"] = _best(cls, kind, param, repeats)
    return rows


def _per_type(repeats):
    rows = {}
    for cls, kind in ((per_type.NumbaRead, "read"),
                      (per_type.NumbaBuild, "build")):
        for param in cls.params[0]:
            rows[f"{kind} {param}"] = _best(cls, kind, param, repeats)
    return rows


def _print(rows, per_element):
    width = max(len(key) for key in rows)
    header = f"{'benchmark':<{width}}  {'time':>10}"
    if per_element:
        header += f"  {'per element':>12}"
    print(header)
    for key, seconds in sorted(rows.items()):
        unit = "us" if seconds < 1e-3 else "ms"
        shown = seconds * 1e6 if unit == "us" else seconds * 1e3
        line = f"{key:<{width}}  {shown:8.3f} {unit}"
        if per_element:
            line += f"  {seconds / N * 1e9:9.2f} ns"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json")
    parser.add_argument("--only", choices=["primitives", "types", "all"],
                        default="all")
    parser.add_argument("--profile-glue", metavar="PRODUCER",
                        choices=list(prims.CallOverhead.params[0]),
                        help="cProfile a thousand small calls; producer is one "
                             "of pyarrow_dense, polars_series, from_any")
    args = parser.parse_args()

    if args.profile_glue:
        harness = prims.CallOverhead()
        harness.setup(args.profile_glue)
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(1000):
            harness.func(harness.arg)
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(12)
        return

    rows = {}
    if args.only in ("primitives", "all"):
        rows |= _primitives(args.repeats)
    if args.only in ("types", "all"):
        rows |= _per_type(args.repeats)
    _print(rows, per_element=args.only != "types")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
