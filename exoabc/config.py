# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import configparser
from collections import defaultdict
from pkg_resources import resource_filename

__all__ = ["parse_config"]


def parse_config(filename=None, save_to=None):
    config = configparser.ConfigParser()

    # Read the defaults
    dfn = resource_filename(__name__, os.path.join("data", "defaults.ini"))
    config.read(dfn)

    # Read the run-specific configuration
    if filename is not None:
        config.read(filename)

    # Save the full config file that was used.
    if save_to is not None:
        with open(save_to, "w") as f:
            config.write(f)

    # Parse the elements:
    result = defaultdict(dict)
    for section in config:
        if not len(config[section]):
            continue
        for element in config[section]:
            result[section][element] = parse_element(config[section][element])

    return dict(result)


def parse_element(element):
    # Numeric types
    try:
        return int(element)
    except ValueError:
        pass
    try:
        return float(element)
    except ValueError:
        pass

    # Constants
    if element.lower() in ["true", "false"]:
        return bool(element)
    if element.lower() == "finite":
        return "finite"

    # Ranges
    if "," in element:
        a, b = element.split(",")
        try:
            a = float(a)
        except ValueError:
            a = None
        try:
            b = float(b)
        except ValueError:
            b = None
        return (a, b)

    return element
